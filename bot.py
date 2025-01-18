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
logger.info(f"PUBLIC_CHANNEL_ID exists: {bool(os.getenv('PUBLIC_CHANNEL_ID'))}")
logger.info(f"PUBLIC_CHANNEL_ID value: {os.getenv('PUBLIC_CHANNEL_ID')}")

try:
    # Initialize the Slack app with additional debugging
    app = App(token=os.environ["SLACK_BOT_TOKEN"])
    logger.info("Slack App initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Slack App: {str(e)}")
    raise

# Handle the /generate command
@app.command("/generate")
def handle_generate_command(ack, respond, command):
    logger.info(f"Received command: {command}")
    # Acknowledge command received
    ack()
    
    # Parse command text for prompt and number of images
    command_text = command['text']
    parts = command_text.split('--n')
    
    prompt = parts[0].strip()
    if not prompt:
        respond("Please provide a prompt for image generation.")
        return
        
    num_images = 5  # default to 5 images
    
    if len(parts) > 1:
        try:
            requested_num = int(parts[1].strip())
            num_images = min(max(1, requested_num), 5)  # limit between 1 and 5 images
        except ValueError:
            num_images = 5  # keep default of 5 if invalid input
    
    try:
        # Get channel IDs
        public_channel_id = os.environ['PUBLIC_CHANNEL_ID']  # Will raise KeyError if not set
        current_channel_id = command['channel_id']
        
        # Determine visibility mode
        is_public = current_channel_id == public_channel_id
        visibility_mode = "public" if is_public else "private"
        logger.info(f"Channel visibility check - Current: {current_channel_id}, Public: {public_channel_id}, Mode: {visibility_mode}")
        
        # Tell user we're working on it
        initial_response = respond(f"Working on generating {num_images} images for: '{prompt}'...")
        
        # Generate images with Ideogram
        result = generate_ideogram_image(prompt, num_images)
        
        if result:
            # result is now a list of tuples (url, prompt)
            ideogram_images = result
            logger.info(f"Successfully generated {len(ideogram_images)} images")
            
            # Create blocks for Slack message with both prompts
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🎨 Generated {len(ideogram_images)} images",
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
            
            # Add each image with its enhanced prompt and download link
            for i, (image_url, image_prompt) in enumerate(ideogram_images, 1):
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
                                "text": f"<{image_url}|🔗 Download Image {i}>"
                            }
                        ]
                    }
                ])
                
                # Only add divider if it's not the last image
                if i < len(ideogram_images):
                    blocks.append({
                        "type": "divider"
                    })
            
            # Set response visibility based on channel
            response_payload = {
                "blocks": blocks,
                "unfurl_links": False,  # Disable link previews
                "unfurl_media": False,  # Disable media previews
                "response_type": "in_channel" if is_public else "ephemeral",
                "replace_original": True
            }
            
            # Update the initial response with the complete message
            respond(response_payload)
            
        else:
            error_msg = "Sorry, I couldn't generate the images. Please try again."
            logger.error(error_msg)
            respond({
                "text": error_msg,
                "replace_original": True
            })
            
    except KeyError as e:
        error_msg = "Missing required environment variable: PUBLIC_CHANNEL_ID"
        logger.error(error_msg)
        respond({
            "text": "Configuration error. Please contact the administrator.",
            "replace_original": True
        })
    except Exception as e:
        error_msg = f"Error processing command: {str(e)}"
        logger.error(error_msg)
        respond({
            "text": "An error occurred while processing your request. Please try again.",
            "replace_original": True
        })

def generate_ideogram_image(prompt, num_images=5):
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
        logger.info(f"Making request to Ideogram API for {num_images} images...")
        logger.debug(f"Request headers (excluding auth): Content-Type: {headers['Content-Type']}")
        logger.debug(f"Request data: {data}")
        
        # Create a session with custom timeout settings
        session = requests.Session()
        session.request = functools.partial(session.request, timeout=None)  # Disable timeout
        
        response = session.post(
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

def run_slack_app():
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

# Start both Flask and Slack apps
if __name__ == "__main__":
    try:
        # Start Slack app in a separate thread
        slack_thread = threading.Thread(target=run_slack_app)
        slack_thread.start()
        logger.info("⚡️ Socket Mode Handler initialized successfully")
        logger.info("⚡️ Slack bot is starting up...")
        
        # Start Flask app
        port = int(os.environ.get("PORT", 8080))
        flask_app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start the bot: {str(e)}")
        raise