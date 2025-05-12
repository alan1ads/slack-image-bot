from dotenv import load_dotenv
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.webhook import WebhookClient
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
import traceback
import base64
from typing import Optional, Dict, Any, Tuple, List
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify environment variables
required_vars = ['SLACK_BOT_TOKEN', 'SLACK_APP_TOKEN', 'IDEOGRAM_API_KEY']
for var in required_vars:
    if not os.environ.get(var):
        logger.error(f"Missing required environment variable: {var}")
    else:
        logger.debug(f"Found environment variable: {var}")

# Verify API key is loaded
api_key = os.environ.get('IDEOGRAM_API_KEY')
if not api_key:
    logger.error("IDEOGRAM_API_KEY not found in environment variables")
else:
    logger.info("IDEOGRAM_API_KEY found in environment variables")

# Initialize Flask app for health checks
flask_app = Flask(__name__)

@flask_app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "slack_connected": hasattr(app, "client") and app.client is not None,
        "version": "1.0.0",
        "timestamp": time.time()
    })

@flask_app.route('/ping')
def ping():
    return "pong", 200

# Validate required environment variables
required_env_vars = [
    'SLACK_BOT_TOKEN',
    'SLACK_APP_TOKEN',
    'IDEOGRAM_API_KEY',
    'MIDJOURNEY_API_KEY',
    'MIDJOURNEY_CHANNEL_ID',
    'MIDJOURNEY_ACCOUNT_HASH',
    'MIDJOURNEY_TOKEN',
    'PUBLIC_CHANNEL_ID',
    'OPENAI_API_KEY'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Initialize Slack app
try:
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    if not slack_bot_token:
        logger.error("SLACK_BOT_TOKEN environment variable is empty or not set")
        raise ValueError("SLACK_BOT_TOKEN is not set properly")
    
    logger.info(f"Initializing Slack app with token: {slack_bot_token[:5]}...{slack_bot_token[-5:] if len(slack_bot_token) > 10 else ''}")
    app = App(token=slack_bot_token)
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

def generate_ideogram_image(prompt, num_images=5, magic_prompt="AUTO"):
    """
    Generate images using Ideogram v3 API
    """
    logger.info(f"Attempting to generate {num_images} images with prompt: {prompt} (Magic Prompt: {magic_prompt})")
    
    api_key = os.environ.get("IDEOGRAM_API_KEY")
    if not api_key:
        logger.error("IDEOGRAM_API_KEY is not set in environment variables")
        return None
        
    headers = {
        'Api-Key': api_key
    }
    
    # Prepare multipart form-data
    form_data = {
        'prompt': (None, prompt, 'text/plain'),
        'num_images': (None, str(num_images), 'text/plain'),
        'magic_prompt': (None, magic_prompt, 'text/plain'),
        'rendering_speed': (None, 'DEFAULT', 'text/plain')
    }
    
    try:
        logger.info(f"Making request to Ideogram v3 API...")
        response = requests.post(
            'https://api.ideogram.ai/v1/ideogram-v3/generate',
            headers=headers,
            files=form_data
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

def generate_ideogram_recreation(image_data, prompt=None, magic_prompt="ON"):
    """
    Generate image remixes using Ideogram v3 API
    """
    api_key = os.environ.get('IDEOGRAM_API_KEY')
    if not api_key:
        raise ValueError("IDEOGRAM_API_KEY not set")

    try:
        headers = {'Api-Key': api_key}
        
        # Ensure prompt is not None
        if not prompt:
            prompt = "Default prompt if none provided"

        # Prepare multipart form data for v3 API
        form_data = {
            'image': ('image.png', image_data, 'image/png'),
            'prompt': (None, prompt, 'text/plain'),
            'magic_prompt': (None, magic_prompt, 'text/plain'),
            'rendering_speed': (None, 'DEFAULT', 'text/plain'),
            'num_images': (None, '5', 'text/plain'),
            'image_weight': (None, '50', 'text/plain')
        }
        
        response = requests.post(
            'https://api.ideogram.ai/v1/ideogram-v3/remix',
            headers=headers,
            files=form_data
        )
        
        response.raise_for_status()
        response_json = response.json()
        
        if 'data' in response_json and response_json['data']:
            image_data = []
            for item in response_json['data']:
                if 'url' in item:
                    # Each item should have its own enhanced prompt when magic_prompt is ON
                    # If no enhanced prompt is provided, fall back to the original prompt
                    enhanced_prompt = item.get('prompt')
                    if not enhanced_prompt and magic_prompt.upper() == "ON":
                        # Log warning if we're not getting unique prompts as expected
                        logger.warning("Magic prompt enabled but no unique prompt received from API")
                    final_prompt = enhanced_prompt if enhanced_prompt else prompt
                    image_data.append((item['url'], final_prompt))
                    logger.info(f"Processed remix with prompt: {final_prompt}")
            
            return image_data
            
        return None
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Error in generate_ideogram_recreation: {str(e)}")
        return None

def generate_openai_image(prompt, num_images=5, size="auto"):
    """
    Generate images using OpenAI's GPT-Image-1 model
    """
    logger.info(f"Generating OpenAI GPT-Image-1 image for prompt: {prompt}, requesting {num_images} images")
    
    # Force requesting exactly 5 images
    num_images = 5
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set in environment variables")
        return None
    
    try:
        # Initialize OpenAI client without proxy settings
        import httpx
        client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(
                proxies=None,
                timeout=120.0  # Increase timeout to 2 minutes
            )
        )
        
        # Call the OpenAI API to generate images
        logger.info(f"Sending request to OpenAI API with size={size}, n={num_images}")
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=num_images,
            size=size
        )
        
        logger.info(f"Received response from OpenAI API with {len(response.data)} images")
        
        # Process the generated images
        image_data = []
        for i, image in enumerate(response.data):
            # OpenAI returns base64 encoded images in b64_json field
            if hasattr(image, 'b64_json') and image.b64_json:
                # Store the raw base64 data - no formatting or URI prefix
                raw_base64 = image.b64_json
                
                # Add enhanced prompt
                enhanced_prompt = f"{prompt}\n\nGenerated with OpenAI GPT-Image-1"
                
                # Store the raw base64 data and enhanced prompt
                image_data.append((raw_base64, enhanced_prompt))
                logger.info(f"Successfully processed image {i+1}")
            else:
                logger.warning(f"Image {i+1} doesn't have b64_json data")
        
        if image_data:
            logger.info(f"Successfully processed {len(image_data)} images from OpenAI")
            return image_data
        else:
            logger.error("No images were successfully processed")
            return None
            
    except Exception as e:
        logger.error(f"Error in generate_openai_image: {str(e)}")
        traceback.print_exc()
        return None

def post_openai_images_to_slack(client, command, prompt, result, respond):
    """
    Special handler for posting OpenAI images to Slack
    """
    try:
        # Determine if this should be ephemeral or in-channel
        is_public_channel = command['channel_id'] == os.environ.get('PUBLIC_CHANNEL_ID')
        
        logger.info(f"Posting OpenAI images to channel {command['channel_id']} (public: {is_public_channel})")
        
        # Create header blocks
        header_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text", 
                    "text": f"🎨 Generated {len(result)} images using OpenAI",
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
        
        # For public channels use chat_postMessage, for private use respond with ephemeral
        if is_public_channel:
            # Public channel - post message publicly
            client.chat_postMessage(
                channel=command['channel_id'],
                blocks=header_blocks,
                text=f"Generated {len(result)} images using OpenAI"
            )
        else:
            # Private channel - post as ephemeral
            respond({
                "blocks": header_blocks,
                "text": f"Generated {len(result)} images using OpenAI",
                "response_type": "ephemeral",
                "replace_original": True
            })
        
        # Process images and upload files
        for i, (base64_data, enhanced_prompt) in enumerate(result, 1):
            try:
                # Create a temporary file to hold the image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    # Decode the base64 data - correctly handling the raw base64 string
                    # Make sure we're working with just the base64 data, not a data URI
                    if isinstance(base64_data, str) and base64_data.startswith('data:'):
                        # Extract just the base64 part from the data URI
                        base64_content = base64_data.split(',')[1]
                    else:
                        # Use the data as-is if it's already just base64
                        base64_content = base64_data
                    
                    # Decode the base64 content
                    try:
                        image_bytes = base64.b64decode(base64_content)
                        temp_file.write(image_bytes)
                        temp_file_path = temp_file.name
                        logger.info(f"Successfully wrote image {i} to temporary file {temp_file_path}")
                    except Exception as decode_error:
                        logger.error(f"Error decoding base64 for image {i}: {str(decode_error)}")
                        continue
                
                # Upload the file to Slack
                upload_result = client.files_upload(
                    channels=command['channel_id'],
                    file=temp_file_path,
                    filename=f"openai-image-{i}.png",
                    title=f"OpenAI Generated Image {i}",
                    initial_comment=f"*✨ Enhanced Prompt for Image {i}:*\n```{enhanced_prompt}```"
                )
                
                # Remove the temporary file after uploading
                os.unlink(temp_file_path)
                
                logger.info(f"Successfully uploaded image {i} to Slack")
                
            except Exception as e:
                logger.error(f"Error uploading image {i}: {str(e)}")
        
        # Update initial message
        respond({
            "text": f"✅ Successfully generated {len(result)} images with OpenAI",
            "response_type": "ephemeral",
            "replace_original": True
        })
        
        logger.info(f"Successfully sent {len(result)} OpenAI images to Slack")
        return True
    except Exception as e:
        logger.error(f"Error posting OpenAI images: {str(e)}")
        traceback.print_exc()  # Add full stack trace for better debugging
        respond({
            "text": f"Error displaying images: {str(e)}",
            "response_type": "ephemeral",
            "replace_original": True
        })
        return False

@app.command("/generate")
def handle_generate_command(ack, respond, command, client):
    """Handle the /generate slash command"""
    ack()
    
    command_text = command.get('text', '').strip()
    original_channel_id = command.get('channel_id')
    user_id = command.get('user_id')
    
    print(f"Debug - Command received:")
    print(f"Channel ID: {original_channel_id}")
    print(f"User ID: {user_id}")
    print(f"Command text: {command_text}")
    
    if not command_text:
        respond({
            "text": "Please specify a service and parameters:\n" +
                   "1. Text generation: `/generate [ideogram|midjourney|openai] [on|off|auto] your prompt`\n" +
                   "2. Image remix: `/generate ideogram-remix [on|off|auto]` (attach an image) [optional prompt]",
            "response_type": "ephemeral"
        })
        return
    
    parts = command_text.split()
    service = parts[0].lower()
    
    if service not in ['ideogram', 'midjourney', 'openai', 'ideogram-remix']:
        respond({
            "text": "Please specify a valid service: 'ideogram', 'midjourney', 'openai', or 'ideogram-remix'",
            "response_type": "ephemeral"
        })
        return

    try:
        # Parse magic prompt option
        magic_prompt = "AUTO"  # default
        prompt_start = 1
        
        if len(parts) > 1 and parts[1].lower() in ['on', 'off', 'auto']:
            magic_prompt = parts[1].upper()
            prompt_start = 2
            
        prompt = " ".join(parts[prompt_start:]) if len(parts) > prompt_start else ""

        if service == 'ideogram-remix':
            try:
                # Store the response_url for later use
                private_metadata = json.dumps({
                    "channel_id": command.get("channel_id"),
                    "user_id": command.get("user_id"),
                    "response_url": command.get("response_url"),
                    "magic_prompt": magic_prompt,
                    "command_context": {
                        "channel_id": command.get("channel_id"),
                        "user_id": command.get("user_id"),
                        "response_url": command.get("response_url")
                    }
                })
                
                result = client.views_open(
                    trigger_id=command["trigger_id"],
                    view={
                        "type": "modal",
                        "callback_id": "recreation_upload_modal",
                        "private_metadata": private_metadata,
                        "title": {
                            "type": "plain_text",
                            "text": "Remix Image",
                            "emoji": True
                        },
                        "submit": {
                            "type": "plain_text",
                            "text": "Generate Remixes",
                            "emoji": True
                        },
                        "blocks": [
                            {
                                "type": "input",
                                "block_id": "image_block",
                                "element": {
                                    "type": "file_input",
                                    "action_id": "file_input",
                                    "filetypes": ["png", "jpg", "jpeg", "webp"]
                                },
                                "label": {
                                    "type": "plain_text",
                                    "text": "Upload an image",
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
                                        "text": "Optional: Add a prompt to guide the remix"
                                    }
                                },
                                "label": {
                                    "type": "plain_text",
                                    "text": "Prompt",
                                    "emoji": True
                                }
                            }
                        ]
                    }
                )
            except Exception as e:
                # Send error to original channel
                respond({
                    "text": f"Failed to open upload dialog: {str(e)}",
                    "response_type": "ephemeral"
                })
        else:
            # Handle other services (ideogram, midjourney, openai)
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
            
            # Handle OpenAI separately to avoid block size limits
            if service == 'openai':
                # Generate OpenAI images
                result = generate_openai_image(prompt)
                if result:
                    # Use the special handler for OpenAI images
                    success = post_openai_images_to_slack(client, command, prompt, result, respond)
                    if not success:
                        error_msg = "Sorry, I couldn't display the images from OpenAI. Please try again."
                        logger.error(error_msg)
                        respond({
                            "text": error_msg,
                            "response_type": "ephemeral",
                            "replace_original": True
                        })
                    return  # Exit after handling OpenAI images
                else:
                    error_msg = "Sorry, I couldn't generate the images using OpenAI. Please try again."
                    logger.error(error_msg)
                    respond({
                        "text": error_msg,
                        "response_type": "ephemeral",
                        "replace_original": True
                    })
                    return  # Exit after error handling
                
            # Handle other services (ideogram, midjourney) with the original code
            if service == 'midjourney':
                result = generate_midjourney_image(prompt)
            else:
                result = generate_ideogram_image(prompt, magic_prompt=magic_prompt)
            
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
                    "response_type": "in_channel" if command['channel_id'] == os.environ.get('PUBLIC_CHANNEL_ID') else "ephemeral",
                    "replace_original": True
                }
                
                # Log channel information
                logger.info(f"Sending response - Channel: {command['channel_id']}, Public: {os.environ.get('PUBLIC_CHANNEL_ID')}")
                
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
        logger.error(f"Error in command handler: {str(e)}")
        respond({
            "text": "An error occurred. Please try again.",
            "response_type": "ephemeral"
        })

@app.view("recreation_upload_modal")
def handle_recreation_submission(ack, body, client):
    try:
        ack()
        
        view = body["view"]
        metadata = json.loads(view["private_metadata"])
        magic_prompt = metadata.get("magic_prompt", "ON")  # Default to ON for remixes
        
        files = view["state"]["values"]["image_block"]["file_input"]["files"]
        user_prompt = view["state"]["values"]["prompt_block"]["prompt_input"].get("value")
        
        send_slack_response(metadata["response_url"], "Working on generating remixes...", channel_id=metadata["channel_id"])
        
        # Get the image data
        file_info = client.files_info(file=files[0]["id"])
        headers = {"Authorization": f"Bearer {os.environ.get('SLACK_BOT_TOKEN')}"}
        image_data = download_slack_image(file_info["file"]["url_private"], headers)
        
        # Get description of the image using Describe API
        description_data = get_image_description(image_data)
        base_description = description_data.get('descriptions', [{}])[0].get('text', 'No description available')
        logger.info(f"Base description: {base_description}")
        
        # Use user prompt if provided, otherwise use the base description
        prompt_to_use = user_prompt if user_prompt else base_description
        
        # Generate remixes using the Remix API
        remix_results = generate_ideogram_recreation(
            image_data=image_data,
            prompt=prompt_to_use,
            magic_prompt=magic_prompt
        )
        
        if not remix_results:
            raise ValueError("Failed to generate remixes")
            
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🎨 Generated Remixes", "emoji": True}
            }
        ]

        # Display results with their unique prompts
        for i, (image_url, prompt) in enumerate(remix_results, 1):
            blocks.extend([
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*✨ Remix {i}:*\n```{prompt}```"}
                },
                {
                    "type": "image",
                    "title": {"type": "plain_text", "text": f"Remix {i}"},
                    "image_url": image_url,
                    "alt_text": f"Generated remix {i}"
                }
            ])

        send_slack_response(
            metadata["response_url"],
            "Generated remixes",
            blocks=blocks,
            channel_id=metadata["channel_id"]
        )
        
        logger.info("Successfully sent remixes to Slack")

    except Exception as e:
        logger.error(f"Error in handle_recreation_submission: {str(e)}")
        if metadata and metadata.get("response_url"):
            send_slack_response(
                metadata["response_url"],
                f"Sorry, something went wrong: {str(e)}",
                channel_id=metadata["channel_id"]
            )

def download_slack_image(url: str, headers: Dict[str, str]) -> bytes:
    """Download image from Slack's private URL"""
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.content

def get_image_description(image_data: bytes) -> Dict[str, Any]:
    """Get description of an image using Ideogram Describe API"""
    api_key = os.environ.get("IDEOGRAM_API_KEY")
    if not api_key:
        raise ValueError("IDEOGRAM_API_KEY not set")

    headers = {'Api-Key': api_key}
    files = {'image_file': ('image.png', image_data, 'image/png')}
    
    logger.info("Making request to Ideogram Describe API...")
    response = requests.post(
        'https://api.ideogram.ai/describe',
        headers=headers,
        files=files
    )
    response.raise_for_status()
    description_data = response.json()
    logger.info(f"Description API response: {json.dumps(description_data, indent=2)}")
    return description_data

def send_slack_response(response_url: str, text: str, blocks: Optional[List[Dict]] = None, channel_id: Optional[str] = None) -> None:
    """Send response back to Slack"""
    data = {
        "text": text,
        "response_type": "in_channel" if channel_id == os.environ.get('PUBLIC_CHANNEL_ID') else "ephemeral",
        "replace_original": True,
        "unfurl_links": False,
        "unfurl_media": False
    }
    if blocks:
        data["blocks"] = blocks
    
    response = requests.post(response_url, json=data)
    response.raise_for_status()

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
        slack_thread.daemon = True  # Make thread exit when main thread exits
        slack_thread.start()
        logger.info("⚡️ Socket Mode Handler initialized successfully")
        logger.info("⚡️ Slack bot is starting up...")
        
        # Start Flask app
        port = int(os.environ.get("PORT", 8080))
        logger.info(f"Starting Flask app on port {port}")
        flask_app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start the bot: {str(e)}")
        traceback.print_exc()  # Print full traceback for better error diagnosis
        raise