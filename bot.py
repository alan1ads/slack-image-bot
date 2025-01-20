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
                timeout=30  # Add timeout
            )
            
            if upscale_response.status_code != 200:
                logger.error(f"Upscale request failed: {upscale_response.status_code}")
                logger.error(f"Response content: {upscale_response.text}")
                if attempt < max_attempts - 1:
                    time.sleep(10)  # Longer delay between retry attempts
                continue
                
            upscale_data = upscale_response.json()
            if 'hash' not in upscale_data:
                logger.error("No hash in upscale response")
                if attempt < max_attempts - 1:
                    time.sleep(10)
                continue
            
            # Poll for the upscaled result with its own timeout
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
        # Initial image generation
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
            
        # Wait for initial generation to complete
        logger.info("Waiting for initial generation to complete...")
        initial_result = poll_midjourney_result(response_data['hash'], api_key)
        if not initial_result:
            logger.error("Failed to get initial generation result")
            return None
            
        # Store results for each upscaled image
        upscaled_results = []
        total_variations = 4
        
        # Upscale each variation with improved error handling and delays
        for i in range(1, total_variations + 1):
            logger.info(f"=== Processing variation {i}/{total_variations} ===")
            
            # Add delay between upscale attempts to avoid rate limiting
            if i > 1:
                time.sleep(15)  # Increased delay between variations
            
            upscaled_url = upscale_midjourney_image(
                response_data['hash'],
                i,
                api_key,
                max_attempts=5,  # Increased max attempts
                poll_max_attempts=45,  # Longer polling for upscale
                poll_delay=8  # Adjusted poll delay
            )
            
            if upscaled_url:
                upscaled_results.append((
                    upscaled_url,
                    f"{prompt} (Variation {i})"
                ))
                logger.info(f"Successfully upscaled variation {i}")
            else:
                logger.warning(f"Failed to upscale variation {i}, continuing with next variation")
        
        # Return results if we have any successful upscales
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
            'magic_prompt': 'AUTO',
            'num_images': num_images
        }
    }
    
    try:
        logger.info(f"Making request to Ideogram API for {num_images} images...")
        logger.debug(f"Request headers (excluding auth): Content-Type: {headers['Content-Type']}")
        logger.debug(f"Request data: {data}")
        
        session = requests.Session()
        session.request = functools.partial(session.request, timeout=None)
        
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

def generate_ideogram_recreation(image_url, prompt=None, num_images=4):
    """
    Generate Ideogram images based on an uploaded image
    """
    logger.info(f"Generating Ideogram recreation from image: {image_url}")
    logger.info(f"With prompt: {prompt if prompt else 'No prompt provided'}")
    
    api_key = os.getenv("IDEOGRAM_API_KEY")
    if not api_key:
        logger.error("IDEOGRAM_API_KEY is not set")
        return None
        
    headers = {
        'Api-Key': api_key,
        'Content-Type': 'application/json'
    }
    
    data = {
        'image_recreation_request': {
            'image_url': image_url,
            'model': 'V_2',  # Using latest model
            'num_images': num_images
        }
    }
    
    # Add optional prompt if provided
    if prompt:
        data['image_recreation_request']['prompt'] = prompt
    
    try:
        logger.info(f"Making request to Ideogram API for {num_images} recreations...")
        logger.debug(f"Request data: {data}")
        
        session = requests.Session()
        session.request = functools.partial(session.request, timeout=None)
        
        response = session.post(
            'https://api.ideogram.ai/recreation',
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
                if 'url' in image_info:
                    # Use original prompt if no enhanced prompt is provided
                    image_prompt = image_info.get('prompt', prompt if prompt else "Image recreation")
                    image_data.append((image_info['url'], image_prompt))
            
            logger.info(f"Successfully extracted {len(image_data)} recreated images")
            return image_data
        else:
            logger.error("No image data found in response")
            return None
            
    except Exception as e:
        logger.error(f"Error generating recreation: {str(e)}")
        return None

def upload_to_storage(file_path):
    """
    Upload a file to Render's disk storage and create a temporary URL
    """
    try:
        # Create a directory in Render's writable filesystem if it doesn't exist
        upload_dir = "/opt/render/project/src/uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        # Generate unique filename
        filename = f"{int(time.time())}_{os.path.basename(file_path)}"
        destination = os.path.join(upload_dir, filename)
        
        # Copy file to uploads directory
        shutil.copy2(file_path, destination)
        
        # Get the base URL from environment variable
        base_url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:8080')
        
        # Create and return the public URL
        public_url = f"{base_url}/uploads/{filename}"
        logger.info(f"File uploaded successfully. Public URL: {public_url}")
        
        return public_url
        
    except Exception as e:
        logger.error(f"Failed to upload file to storage: {str(e)}")
        raise

@app.command("/generate")
def handle_generate_command(ack, respond, command, client):
    """
    Handle the /generate slash command with file upload support
    """
    logger.info(f"Received command: {command}")
    ack()
    
    command_text = command['text'].strip()
    
    if not command_text:
        respond({
            "text": "Please specify a service and parameters:\n" +
                   "1. Text generation: `/generate [ideogram|midjourney] your prompt`\n" +
                   "2. Image recreation: `/generate ideogram-recreation` (attach an image) [optional prompt]",
            "response_type": "ephemeral"
        })
        return
    
    parts = command_text.split()
    service = parts[0].lower()
    
    if service not in ['ideogram', 'midjourney', 'ideogram-recreation']:
        respond({
            "text": "Please specify a valid service: 'ideogram', 'midjourney', or 'ideogram-recreation'",
            "response_type": "ephemeral"
        })
        return

    try:
        # Get channel IDs
        public_channel_id = os.environ['PUBLIC_CHANNEL_ID']
        current_channel_id = command['channel_id']
        is_public = current_channel_id == public_channel_id

        if service == 'ideogram-recreation':
            # Open a modal for file upload
            client.views_open(
                trigger_id=command["trigger_id"],
                view={
                    "type": "modal",
                    "callback_id": "image_upload_modal",
                    "title": {
                        "type": "plain_text",
                        "text": "Upload Image",  # Shortened title
                        "emoji": True
                    },
                    "submit": {
                        "type": "plain_text",
                        "text": "Generate",
                        "emoji": True
                    },
                    "close": {
                        "type": "plain_text",
                        "text": "Cancel",
                        "emoji": True
                    },
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "Upload an image to recreate using Ideogram AI"
                            }
                        },
                        {
                            "type": "input",
                            "block_id": "image_block",
                            "label": {
                                "type": "plain_text",
                                "text": "Image",
                                "emoji": True
                            },
                            "element": {
                                "type": "file_input",
                                "action_id": "image_input",
                                "filetypes": ["png", "jpg", "jpeg", "gif"]
                            }
                        },
                        {
                            "type": "input",
                            "block_id": "prompt_block",
                            "optional": True,
                            "label": {
                                "type": "plain_text",
                                "text": "Optional prompt",
                                "emoji": True
                            },
                            "element": {
                                "type": "plain_text_input",
                                "action_id": "prompt_input",
                                "placeholder": {
                                    "type": "plain_text",
                                    "text": "Enter a prompt to guide the recreation",
                                    "emoji": True
                                }
                            }
                        }
                    ]
                }
            )
            return
        else:
            # Handle regular text-to-image generation
            prompt = ' '.join(parts[1:])
            
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
                            "text": f"*Original Prompt:*\n```{prompt}```"
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
                                "text": f"*Image {i} of {len(result)}*\n```{enhanced_prompt}```"
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
                    "text": f"Generated {len(result)} images using {service.title()}",  # Fallback text
                    "unfurl_links": False,
                    "unfurl_media": False,
                    "response_type": "in_channel" if is_public else "ephemeral",
                    "replace_original": True
                }
                
                # Log channel information
                logger.info(f"Sending response - Channel ID: {current_channel_id}, Public Channel: {public_channel_id}, Is Public: {is_public}")
                
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
            "response_type": "ephemeral"
        })

@app.view("image_upload_modal")
def handle_image_upload_submission(ack, body, view, client):
    """
    Handle the submission of the image upload modal
    """
    # Acknowledge the view submission immediately
    ack()
    
    try:
        # Get user ID and prompt
        user_id = body["user"]["id"]
        prompt = view["state"]["values"]["prompt_block"]["prompt_input"]["value"]
        
        # Get the file from the modal
        files = view["state"]["values"]["image_block"]["image_input"].get("files", [])
        
        if not files:
            client.chat_postEphemeral(
                channel=user_id,
                user=user_id,
                text="⚠️ No image was uploaded. Please try again with an image."
            )
            return

        # Send initial status message
        client.chat_postEphemeral(
            channel=user_id,
            user=user_id,
            text="📥 Processing your uploaded image..."
        )

        # Wait a moment for Slack to process the file
        time.sleep(2)

        # Get file info
        file_id = files[0]
        
        try:
            # Get the file using files.info with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    file_info = client.files_info(file=file_id)
                    if file_info and file_info.get('ok'):
                        break
                    time.sleep(2)  # Wait between retries
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)

            if not file_info or not file_info.get('ok'):
                raise Exception("Could not retrieve file information")

            file_url = file_info["file"]["url_private"]
            
            # Download the file
            headers = {"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"}
            file_response = requests.get(file_url, headers=headers)
            
            if file_response.status_code != 200:
                raise Exception(f"Failed to download file. Status code: {file_response.status_code}")

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(file_response.content)
                temp_file_path = temp_file.name

            try:
                # Upload to storage
                public_url = upload_to_storage(temp_file_path)
                
                # Update status
                client.chat_postEphemeral(
                    channel=user_id,
                    user=user_id,
                    text="🎨 Starting image recreation... This may take a few moments."
                )
                
                # Generate recreations
                result = generate_ideogram_recreation(public_url, prompt)
                
                if result:
                    # Get channel for posting
                    public_channel_id = os.environ.get('PUBLIC_CHANNEL_ID')
                    target_channel = public_channel_id if public_channel_id else user_id
                    
                    # Send results
                    for i, (image_url, enhanced_prompt) in enumerate(result, 1):
                        blocks = [
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Recreation {i} of {len(result)}*\n```{enhanced_prompt}```"
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
                                        "text": f"🔗 <{image_url}|Download image {i}>"
                                    }
                                ]
                            }
                        ]
                        
                        client.chat_postMessage(
                            channel=target_channel,
                            blocks=blocks,
                            text=f"Generated recreation {i} of {len(result)}",
                            unfurl_links=False,
                            unfurl_media=False
                        )
                        time.sleep(1)
                    
                    logger.info(f"Successfully sent {len(result)} recreated images")
                    
                else:
                    client.chat_postEphemeral(
                        channel=user_id,
                        user=user_id,
                        text="❌ Failed to generate recreations. Please try again."
                    )
                    
            finally:
                # Cleanup
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {str(e)}")
                    
        except Exception as e:
            logger.error(f"File handling error: {str(e)}")
            client.chat_postEphemeral(
                channel=user_id,
                user=user_id,
                text="❌ Failed to process the uploaded file. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Error in image upload submission: {str(e)}")
        if body and "user" in body:
            client.chat_postEphemeral(
                channel=body["user"]["id"],
                user=body["user"]["id"],
                text="❌ An error occurred. Please try again or contact support."
            )

def run_slack_app():
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

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