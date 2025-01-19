from dotenv import load_dotenv
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import requests
import time
import logging
import threading
import json
from flask import Flask, jsonify
import functools

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app for health checks
flask_app = Flask(__name__)

@flask_app.route('/')
def health_check():
    return jsonify({"status": "healthy"})

# Validate required environment variables
required_env_vars = [
    'SLACK_BOT_TOKEN',
    'SLACK_APP_TOKEN',
    'IDEOGRAM_API_KEY',
    'MIDJOURNEY_API_KEY',
    'PUBLIC_CHANNEL_ID'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Print environment check (without exposing tokens)
logger.info("Environment variables check:")
logger.info(f"SLACK_BOT_TOKEN exists: {bool(os.getenv('SLACK_BOT_TOKEN'))}")
logger.info(f"SLACK_APP_TOKEN exists: {bool(os.getenv('SLACK_APP_TOKEN'))}")
logger.info(f"IDEOGRAM_API_KEY exists: {bool(os.getenv('IDEOGRAM_API_KEY'))}")
logger.info(f"MIDJOURNEY_API_KEY exists: {bool(os.getenv('MIDJOURNEY_API_KEY'))}")
logger.info(f"PUBLIC_CHANNEL_ID exists: {bool(os.getenv('PUBLIC_CHANNEL_ID'))}")

try:
    # Initialize the Slack app with additional debugging
    app = App(token=os.environ["SLACK_BOT_TOKEN"])
    logger.info("Slack App initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Slack App: {str(e)}")
    raise

def generate_midjourney_image(prompt):
    """
    Generate images using Midjourney API through UserAPI
    """
    logger.info(f"Attempting to generate Midjourney image with prompt: {prompt}")
    
    api_key = os.environ.get("MIDJOURNEY_API_KEY")
    if not api_key:
        logger.error("MIDJOURNEY_API_KEY is not set in environment variables")
        return None
        
    headers = {
        'api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'prompt': prompt,
        'webhook_type': 'result',
        'is_disable_prefilter': False
    }
    
    try:
        logger.info("Making request to Midjourney API...")
        logger.debug(f"Request headers (excluding auth): Content-Type: {headers['Content-Type']}")
        logger.debug(f"Request data: {data}")
        
        session = requests.Session()
        session.request = functools.partial(session.request, timeout=None)
        
        response = session.post(
            'https://api.userapi.ai/midjourney/v2/imagine',
            headers=headers,
            json=data
        )
        
        logger.info(f"Received response from Midjourney API. Status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Midjourney API error. Status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return None
            
        response_json = response.json()
        logger.info("=== MIDJOURNEY API RESPONSE ===")
        logger.info(json.dumps(response_json, indent=2))
        logger.info("===========================")
        
        if 'hash' in response_json:
            image_result = poll_midjourney_result(response_json['hash'], headers)
            if image_result:
                return [(image_result['url'], image_result.get('enhanced_prompt', prompt))]
        
        logger.error("No valid response from Midjourney API")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return None

def poll_midjourney_result(hash_id, headers, max_attempts=30, delay=10):
    """
    Poll for Midjourney result using the provided hash
    """
    url = f'https://api.userapi.ai/midjourney/v2/result/{hash_id}'
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'completed':
                    return {
                        'url': result['url'],
                        'enhanced_prompt': result.get('enhanced_prompt')
                    }
                elif result.get('status') == 'failed':
                    logger.error(f"Midjourney generation failed: {result.get('error')}")
                    return None
            
            logger.info(f"Generation in progress, attempt {attempt + 1}/{max_attempts}")
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Error polling result: {str(e)}")
            return None
    
    logger.error("Timeout waiting for Midjourney result")
    return None

def generate_ideogram_image(prompt, num_images=5):
    """
    Generate images using Ideogram API
    """
    logger.info(f"Attempting to generate {num_images} images with prompt: {prompt}")
    
    api_key = os.environ.get("IDEOGRAM_API_KEY")
    if not api_key:
        logger.error("IDEOGRAM_API_KEY is not set in environment variables")
        return None
        
    headers = {
        'Api-Key': api_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'image_request': {
            'prompt': prompt,
            'aspect_ratio': 'ASPECT_10_16',
            'model': 'V_2',
            'magic_prompt': 'AUTO',
            'num_images': num_images
        }
    }
    
    try:
        logger.info(f"Making request to Ideogram API...")
        logger.debug(f"Request data: {data}")
        
        response = requests.post(
            'https://api.ideogram.ai/generate',
            headers=headers,
            json=data
        )
        
        logger.info(f"Received response from Ideogram API. Status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Ideogram API error. Status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return None
            
        response_json = response.json()
        logger.info("=== IDEOGRAM API RESPONSE ===")
        logger.info(json.dumps(response_json, indent=2))
        logger.info("===========================")
        
        if 'data' in response_json and response_json['data']:
            image_data = []
            for image_info in response_json['data']:
                if 'url' in image_info and 'prompt' in image_info:
                    image_data.append((image_info['url'], image_info['prompt']))
            
            logger.info(f"Successfully extracted {len(image_data)} images with their enhanced prompts")
            if len(image_data) < num_images:
                logger.warning(f"Requested {num_images} images but only received {len(image_data)}")
            
            return image_data
        else:
            logger.error("No image data found in response")
            logger.debug(f"Full response: {response_json}")
            return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return None

@app.command("/generate")
def handle_generate_command(ack, respond, command):
    logger.info(f"Received command: {command}")
    ack()
    
    command_text = command['text'].strip()
    
    if not command_text:
        respond({
            "text": "Please specify a service (ideogram or midjourney) and a prompt.\nExample: `/generate ideogram a beautiful sunset` or `/generate midjourney a beautiful sunset`",
            "response_type": "ephemeral"
        })
        return
    
    parts = command_text.split()
    
    if len(parts) < 2:
        respond({
            "text": "Please specify both a service (ideogram or midjourney) and a prompt.\nExample: `/generate ideogram a beautiful sunset` or `/generate midjourney a beautiful sunset`",
            "response_type": "ephemeral"
        })
        return
    
    service = parts[0].lower()
    if service not in ['ideogram', 'midjourney']:
        respond({
            "text": "Please specify a valid service: 'ideogram' or 'midjourney'.\nExample: `/generate ideogram a beautiful sunset` or `/generate midjourney a beautiful sunset`",
            "response_type": "ephemeral"
        })
        return
    
    prompt_parts = ' '.join(parts[1:]).split('--n')
    prompt = prompt_parts[0].strip()
    
    if not prompt:
        respond({
            "text": f"Please provide a prompt for {service}.",
            "response_type": "ephemeral"
        })
        return
    
    num_images = 5 if service == 'ideogram' else 1
    
    if len(prompt_parts) > 1 and service == 'ideogram':
        try:
            requested_num = int(prompt_parts[1].strip())
            num_images = min(max(1, requested_num), 5)
        except ValueError:
            pass
    elif len(prompt_parts) > 1 and service == 'midjourney':
        try:
            requested_num = int(prompt_parts[1].strip())
            if requested_num != 1:
                respond({
                    "text": "Note: Midjourney always generates 1 image regardless of the --n parameter.",
                    "response_type": "ephemeral"
                })
        except ValueError:
            pass
    
    try:
        # Get channel IDs
        public_channel_id = os.environ['PUBLIC_CHANNEL_ID']
        current_channel_id = command['channel_id']
        
        # Determine visibility mode
        is_public = current_channel_id == public_channel_id
        
        # Tell user we're working on it (always ephemeral)
        respond({
            "text": f"Working on generating image{'s' if num_images > 1 else ''} using {service}...",
            "response_type": "ephemeral"
        })
        
        # Generate images based on selected service
        result = generate_midjourney_image(prompt) if service == 'midjourney' else generate_ideogram_image(prompt, num_images)
        
        if result:
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🎨 Generated {len(result)} image{'s' if len(result) > 1 else ''} using {service.title()}",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*📝 Original Prompt:*\n```" + prompt + "```"
                    }
                },
                {
                    "type": "divider"
                }
            ]
            
            for i, (image_url, image_prompt) in enumerate(result, 1):
                blocks.extend([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*✨ Enhanced Prompt for Image {i}:*\n```{image_prompt}```"
                        }
                    },
                    {
                        "type": "image",
                        "title": {
                            "type": "plain_text",
                            "text": f"Generated Image {i}"
                        },
                        "image_url": image_url,
                        "alt_text": f"AI generated image {i}"
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"🔗 <{image_url}|Click here to download image {i}>"
                            }
                        ]
                    }
                ])
                
                if i < len(result):
                    blocks.append({
                        "type": "divider"
                    })
            
            response_payload = {
                "blocks": blocks,
                "unfurl_links": False,
                "unfurl_media": False,
                "response_type": "in_channel" if is_public else "ephemeral",
                "replace_original": True
            }
            
            respond(response_payload)
        else:
            error_msg = f"Sorry, I couldn't generate the image{'s' if num_images > 1 else ''} using {service}. Please try again."
            logger.error(error_msg)
            respond({
                "text": error_msg,
                "response_type": "ephemeral",
                "replace_original": True
            })
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        respond({
            "text": "An error occurred. Please try again or contact support if the problem persists.",
            "response_type": "ephemeral",
            "replace_original": True
        })

def run_slack_app():
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

if __name__ == "__main__":
    try:
        slack_thread = threading.Thread(target=run_slack_app)
        slack_thread.start()
        logger.info("⚡️ Socket Mode Handler initialized successfully")
        logger.info("⚡️ Slack bot is starting up...")
        
        port = int(os.environ.get("PORT", 8080))
        flask_app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start the bot: {str(e)}")
        raise