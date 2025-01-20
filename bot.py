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
import tempfile
import shutil
from pathlib import Path

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
    'MIDJOURNEY_CHANNEL_ID',
    'MIDJOURNEY_ACCOUNT_HASH',
    'MIDJOURNEY_TOKEN',
    'PUBLIC_CHANNEL_ID'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Initialize Slack app
try:
    app = App(token=os.environ["SLACK_BOT_TOKEN"])
    logger.info("Slack App initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Slack App: {str(e)}")
    raise

def poll_midjourney_result(hash_id, api_key, max_attempts=60, delay=10):
    """
    Poll for Midjourney result using the provided hash
    """
    headers = {
        'api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    base_url = 'https://api.userapi.ai'
    status_url = f'{base_url}/midjourney/v2/status?hash={hash_id}'
    
    logger.info(f"Starting to poll for results with hash: {hash_id}")
    
    for attempt in range(max_attempts):
        try:
            status_response = requests.get(status_url, headers=headers)
            if status_response.status_code != 200:
                logger.error(f"Failed to get status. Status code: {status_response.status_code}")
                logger.error(f"Response content: {status_response.text}")
                time.sleep(delay)
                continue

            status_data = status_response.json()
            status = status_data.get('status')
            progress = status_data.get('progress', 0)
            
            logger.info(f"Status: {status}, Progress: {progress}%")
            
            if status == 'done':
                result = status_data.get('result', {})
                if result and result.get('url'):
                    return {
                        'url': result['url'],
                        'proxy_url': result.get('proxy_url'),
                        'enhanced_prompt': status_data.get('prompt'),
                        'type': status_data.get('type')
                    }
                    
            elif status == 'failed':
                logger.error(f"Generation failed: {status_data.get('status_reason')}")
                return None
                
            elif status in ['queued', 'progress']:
                if attempt % 3 == 0:  # Log every third attempt to reduce noise
                    logger.info(f"Generation in progress ({attempt + 1}/{max_attempts}): {progress}%")
                time.sleep(delay)
                continue
                
            else:
                logger.warning(f"Unknown status: {status}")
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error polling result: {str(e)}")
            time.sleep(delay)
    
    logger.error("Timeout waiting for result")
    return None

def upscale_midjourney_image(hash_id, choice, api_key, max_attempts=5, poll_max_attempts=45, poll_delay=8):
    """
    Upscale a specific image variation from Midjourney with improved polling
    """
    logger.info(f"Attempting to upscale variation {choice} from hash: {hash_id}")
    
    headers = {
        'api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'hash': hash_id,
        'choice': choice,
        'webhook_type': 'result'
    }
    
    for attempt in range(max_attempts):
        try:
            logger.info(f"Upscale attempt {attempt + 1}/{max_attempts} for variation {choice}")
            upscale_response = requests.post(
                'https://api.userapi.ai/midjourney/v2/upscale',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if upscale_response.status_code != 200:
                logger.error(f"Upscale request failed: {upscale_response.status_code}")
                logger.error(f"Response content: {upscale_response.text}")
                if attempt < max_attempts - 1:
                    time.sleep(10)
                continue
                
            upscale_data = upscale_response.json()
            if 'hash' not in upscale_data:
                logger.error("No hash in upscale response")
                if attempt < max_attempts - 1:
                    time.sleep(10)
                continue
            
            upscale_result = poll_midjourney_result(
                upscale_data['hash'], 
                api_key, 
                max_attempts=poll_max_attempts, 
                delay=poll_delay
            )
            
            if upscale_result and 'url' in upscale_result:
                return upscale_result['url']
            
            logger.warning(f"Failed to get upscale result, attempt {attempt + 1}")
            if attempt < max_attempts - 1:
                time.sleep(10)
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout during upscale attempt {attempt + 1}")
            if attempt < max_attempts - 1:
                time.sleep(10)
                continue
        except Exception as e:
            logger.error(f"Error during upscale attempt {attempt + 1}: {str(e)}")
            if attempt < max_attempts - 1:
                time.sleep(10)
                continue
            
    logger.error(f"Failed to upscale variation {choice} after {max_attempts} attempts")
    return None

def generate_midjourney_image(prompt):
    """
    Generate Midjourney images with automatic upscaling for each variation
    """
    logger.info(f"Generating Midjourney image for prompt: {prompt}")
    
    api_key = os.getenv("MIDJOURNEY_API_KEY")
    if not api_key:
        logger.error("MIDJOURNEY_API_KEY is not set")
        return None
    
    headers = {
        'api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'prompt': prompt,
        'webhook_type': 'result',
        'is_disable_prefilter': False,
        'channel_id': os.getenv('MIDJOURNEY_CHANNEL_ID'),
        'account_hash': os.getenv('MIDJOURNEY_ACCOUNT_HASH'),
        'token': os.getenv('MIDJOURNEY_TOKEN')
    }
    
    try:
        logger.info("Initiating Midjourney image generation...")
        response = requests.post(
            'https://api.userapi.ai/midjourney/v2/imagine',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to generate initial images: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return None
            
        response_data = response.json()
        logger.info(f"Initial generation response: {json.dumps(response_data, indent=2)}")
        
        if 'hash' not in response_data:
            logger.error("No hash in response")
            return None
            
        logger.info("Waiting for initial generation to complete...")
        initial_result = poll_midjourney_result(response_data['hash'], api_key)
        if not initial_result:
            logger.error("Failed to get initial generation result")
            return None
            
        upscaled_results = []
        total_variations = 4
        
        for i in range(1, total_variations + 1):
            logger.info(f"=== Processing variation {i}/{total_variations} ===")
            
            if i > 1:
                time.sleep(15)
            
            upscaled_url = upscale_midjourney_image(
                response_data['hash'],
                i,
                api_key,
                max_attempts=5,
                poll_max_attempts=45,
                poll_delay=8
            )
            
            if upscaled_url:
                upscaled_results.append((
                    upscaled_url,
                    f"{prompt} (Variation {i})"
                ))
                logger.info(f"Successfully upscaled variation {i}")
            else:
                logger.warning(f"Failed to upscale variation {i}, continuing with next variation")
        
        if upscaled_results:
            logger.info(f"Successfully generated {len(upscaled_results)} upscaled images")
            return upscaled_results
        else:
            logger.error("No successful upscales")
            return None
        
    except Exception as e:
        logger.error(f"Error in generate_midjourney_image: {str(e)}")
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
            'magic_prompt_option': 'AUTO',
            'num_images': num_images
        }
    }
    
    try:
        logger.info(f"Making request to Ideogram API...")
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
            return None
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return None

def generate_ideogram_recreation(image_file_content, prompt=None):
    """
    Generate recreations of an image using Ideogram Remix API
    """
    logger.info("Starting image recreation with Ideogram")
    
    api_key = os.environ.get('IDEOGRAM_API_KEY')
    if not api_key:
        logger.error("IDEOGRAM_API_KEY not found")
        return None
    
    try:
        headers = {
            'Api-Key': api_key,
            'Accept': 'application/json'
        }
        
        # Prepare the multipart/form-data request
        files = {
            'image_file': ('image.png', image_file_content, 'image/png')
        }
        
        # Prepare the remix request data
        request_data = {
            'prompt': prompt if prompt else "Recreate this image with creative variations",
            'aspect_ratio': 'ASPECT_10_16',
            'model': 'V_2',
            'magic_prompt_option': 'AUTO'
        }
        
        # Create proper multipart form data
        data = {
            'image_request': json.dumps(request_data)
        }
        
        logger.info("Making request to Ideogram Remix API...")
        logger.debug(f"Request data: {json.dumps(request_data, indent=2)}")
        
        response = requests.post(
            'https://api.ideogram.ai/remix',
            headers=headers,
            data=data,
            files=files,
            timeout=None
        )
        
        logger.info(f"Remix API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Failed to generate recreations. Status: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return None
        
        response_json = response.json()
        logger.info("=== REMIX API RESPONSE ===")
        logger.info(json.dumps(response_json, indent=2))
        logger.info("=========================")
        
        if 'data' in response_json and response_json['data']:
            image_data = []
            for image_info in response_json['data']:
                if 'url' in image_info:
                    # Use enhanced prompt if available, otherwise use original or default
                    image_prompt = (
                        image_info.get('enhanced_prompt') or 
                        image_info.get('prompt') or 
                        prompt or 
                        "Recreation variation"
                    )
                    image_data.append((image_info['url'], image_prompt))
            
            logger.info(f"Successfully generated {len(image_data)} recreations")
            return image_data
        else:
            logger.error("No image data in response")
            return None
        
    except Exception as e:
        logger.error(f"Error in generate_ideogram_recreation: {str(e)}")
        return None
@app.command("/generate")
def handle_generate_command(ack, respond, command, client):
    """
    Handle the /generate slash command
    """
    logger.info(f"Received command: {command}")
    ack()
    
    command_text = command['text'].strip()
    
    if not command_text:
        respond({
            "text": "Please specify a service and parameters:\n" +
                   "1. Text to image: `/generate [ideogram|midjourney] your prompt`\n" +
                   "2. Image recreation: `/generate recreation` (with an image file)",
            "response_type": "ephemeral"
        })
        return
    
    parts = command_text.split()
    service = parts[0].lower()
    
    if service not in ['ideogram', 'midjourney', 'recreation']:
        respond({
            "text": "Please specify a valid service: 'ideogram', 'midjourney', or 'recreation'",
            "response_type": "ephemeral"
        })
        return

    try:
        # Get channel IDs
        public_channel_id = os.environ['PUBLIC_CHANNEL_ID']
        current_channel_id = command['channel_id']
        is_public = current_channel_id == public_channel_id

        if service == 'recreation':
            # Open modal for file upload
            try:
                client.views_open(
                    trigger_id=command["trigger_id"],
                    view={
                        "type": "modal",
                        "callback_id": "recreation_upload_modal",
                        "title": {
                            "type": "plain_text",
                            "text": "Upload Image for Recreation",
                            "emoji": True
                        },
                        "submit": {
                            "type": "plain_text",
                            "text": "Generate",
                            "emoji": True
                        },
                        "blocks": [
                            {
                                "type": "input",
                                "block_id": "image_block",
                                "element": {
                                    "type": "file_input",
                                    "action_id": "file_input",
                                    "filetypes": ["png", "jpg", "jpeg"]
                                },
                                "label": {
                                    "type": "plain_text",
                                    "text": "Select an image to recreate",
                                    "emoji": True
                                }
                            },
                            {
                                "type": "input",
                                "block_id": "prompt_block",
                                "optional": True,
                                "element": {
                                    "type": "plain_text_input",
                                    "action_id": "prompt_input",
                                    "placeholder": {
                                        "type": "plain_text",
                                        "text": "Optional: Add guidance for the recreation"
                                    }
                                },
                                "label": {
                                    "type": "plain_text",
                                    "text": "Recreation Prompt",
                                    "emoji": True
                                }
                            }
                        ]
                    }
                )
            except Exception as e:
                logger.error(f"Error opening modal: {str(e)}")
                respond({
                    "text": "Failed to open upload dialog. Please try again.",
                    "response_type": "ephemeral"
                })
            return
        else:
            # Handle text-to-image generation
            prompt = ' '.join(parts[1:])
            if not prompt:
                respond({
                    "text": f"Please provide a prompt for {service}.",
                    "response_type": "ephemeral"
                })
                return
            
            # Tell user we're working on it
            initial_response = respond({
                "text": f"Working on generating images using {service}...\nThis might take a few minutes, especially for Midjourney with upscaling.",
                "response_type": "ephemeral"
            })
            
            # Generate images based on selected service
            result = generate_midjourney_image(prompt) if service == 'midjourney' else generate_ideogram_image(prompt)
            
            if result:
                # Create blocks for each image
                blocks = [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"🎨 Generated {len(result)} images using {service.title()}",
                            "emoji": True
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*📝 Original Prompt:*\n```{prompt}```"
                        }
                    }
                ]
                
                # Add each image as a separate message block
                for i, (image_url, enhanced_prompt) in enumerate(result, 1):
                    blocks.extend([
                        {
                            "type": "divider"
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*✨ Enhanced Prompt for Image {i}:*\n```{enhanced_prompt}```"
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
                                    "text": f"🔗 <{image_url}|Click to download image {i}>"
                                }
                            ]
                        }
                    ])
                
                # Send the response with appropriate visibility
                response_payload = {
                    "blocks": blocks,
                    "text": f"Generated {len(result)} images using {service.title()}",
                    "unfurl_links": False,
                    "unfurl_media": False,
                    "response_type": "in_channel" if is_public else "ephemeral",
                    "replace_original": True
                }
                
                # Log channel information
                logger.info(f"Sending response - Channel: {current_channel_id}, Public: {public_channel_id}, Is Public: {is_public}")
                
                respond(response_payload)
                logger.info(f"Successfully sent {len(result)} images to Slack")
                
            else:
                error_msg = f"Sorry, I couldn't generate the images using {service}. Please try again."
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
            "text": "An unexpected error occurred. Please try again or contact support if the problem persists.",
            "response_type": "ephemeral",
            "replace_original": True
        })
@app.view("recreation_upload_modal")
def handle_recreation_submission(ack, body, view, client):
    """
    Handle the submission of the recreation upload modal
    """
    ack()
    
    try:
        user_id = body["user"]["id"]
        prompt = view["state"]["values"]["prompt_block"]["prompt_input"].get("value", "")
        
        # Get file from the input
        try:
            file_data = view["state"]["values"]["image_block"]["file_input"]["files"][0]
            
        except (KeyError, IndexError) as e:
            logger.error(f"Error accessing file data: {str(e)}")
            client.chat_postEphemeral(
                channel=user_id,
                user=user_id,
                text="⚠️ No image file was uploaded. Please try again."
            )
            return
            
        # Send initial status message
        client.chat_postEphemeral(
            channel=user_id,
            user=user_id,
            text="🔄 Processing your image for recreation..."
        )
        
        try:
            # Get the file URL and download it
            file_info = client.files_info(file=file_data)
            file_url = file_info["file"]["url_private"]
            
            # Download file with proper authorization
            headers = {"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"}
            response = requests.get(file_url, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to download file: {response.status_code}")
            
            # Get the image content
            image_content = response.content
            
            # Update status
            client.chat_postEphemeral(
                channel=user_id,
                user=user_id,
                text="🎨 Generating recreations... This may take a minute."
            )
            
            # Generate recreations using Ideogram
            result = generate_ideogram_recreation(image_content, prompt)
            
            if not result:
                raise Exception("Failed to generate recreations")
            
            # Get public channel ID and determine where to post
            public_channel_id = os.environ.get('PUBLIC_CHANNEL_ID')
            target_channel = public_channel_id if public_channel_id else user_id
            is_public = bool(public_channel_id)
            
            # Create and send message blocks for results
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🎨 Image Recreation Results",
                        "emoji": True
                    }
                }
            ]
            
            if prompt:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*📝 Recreation Prompt:*\n```{prompt}```"
                    }
                })
            
            blocks.append({"type": "divider"})
            
            # Add each recreation result
            for i, (image_url, enhanced_prompt) in enumerate(result, 1):
                blocks.extend([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*✨ Recreation {i} Enhanced Prompt:*\n```{enhanced_prompt}```"
                        }
                    },
                    {
                        "type": "image",
                        "title": {
                            "type": "plain_text",
                            "text": f"Recreation {i}"
                        },
                        "image_url": image_url,
                        "alt_text": f"AI recreation {i}"
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"🔗 <{image_url}|Click to download recreation {i}>"
                            }
                        ]
                    }
                ])
                
                if i < len(result):
                    blocks.append({"type": "divider"})
            
            # Send the results
            client.chat_postMessage(
                channel=target_channel,
                blocks=blocks,
                text="Image Recreation Results",
                unfurl_links=False,
                unfurl_media=False
            )
            
            # If sent to public channel, notify user
            if is_public:
                client.chat_postEphemeral(
                    channel=user_id,
                    user=user_id,
                    text="✅ Recreation results have been posted in the public channel!"
                )
            
        except Exception as e:
            logger.error(f"Error processing recreation: {str(e)}")
            client.chat_postEphemeral(
                channel=user_id,
                user=user_id,
                text="❌ Failed to process the recreation. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Error in recreation submission handler: {str(e)}")
        if body and "user" in body:
            client.chat_postEphemeral(
                channel=body["user"]["id"],
                user=body["user"]["id"],
                text="❌ An error occurred. Please try again."
            )
def run_slack_app():
    """
    Run the Slack app in socket mode
    """
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

# Main entry point
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