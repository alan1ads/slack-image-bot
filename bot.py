from dotenv import load_dotenv
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import requests
import logging
import threading
import json
from flask import Flask, jsonify

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

# Print environment check (without exposing tokens)
logger.info("Environment variables check:")
logger.info(f"SLACK_BOT_TOKEN exists: {bool(os.getenv('SLACK_BOT_TOKEN'))}")
logger.info(f"SLACK_APP_TOKEN exists: {bool(os.getenv('SLACK_APP_TOKEN'))}")
logger.info(f"IDEOGRAM_API_KEY exists: {bool(os.getenv('IDEOGRAM_API_KEY'))}")

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
    num_images = 5  # default to 5 images
    
    if len(parts) > 1:
        try:
            requested_num = int(parts[1].strip())
            num_images = min(max(1, requested_num), 5)  # limit between 1 and 5 images
        except ValueError:
            num_images = 5  # keep default of 5 if invalid input
    
    # Tell user we're working on it
    respond(f"Working on generating {num_images} images for: '{prompt}'...")
    
    try:
        # Generate images with Ideogram
        result = generate_ideogram_image(prompt, num_images)
        
        if result:
            ideogram_images, enhanced_prompt = result
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
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*✨ Ideogram's Magic Prompt:*\n```" + enhanced_prompt + "```"
                    }
                },
                {
                    "type": "divider"
                }
            ]
            
            # Add each image and its download link
            for i, image_url in enumerate(ideogram_images, 1):
                blocks.extend([
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
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"<{image_url}|📥 Download Image {i}>"
                        }
                    }
                ])
            
            respond({"blocks": blocks})
        else:
            logger.error("Failed to generate images")
            respond("Sorry, I couldn't generate the images. Please try again.")
            
    except Exception as e:
        logger.error(f"Error in command handler: {str(e)}")
        respond(f"Error: {str(e)}")

def generate_ideogram_image(prompt, num_images=5):
    """
    Generate images using Ideogram API
    
    Args:
        prompt (str): The prompt for image generation
        num_images (int): Number of images to generate (default: 5)
    
    Returns:
        tuple: (list of image URLs, enhanced prompt if available)
    """
    logger.info(f"Attempting to generate {num_images} images with prompt: {prompt}")
    
    # Check if we have the API key
    api_key = os.environ.get("IDEOGRAM_API_KEY")
    if not api_key:
        logger.error("IDEOGRAM_API_KEY is not set in environment variables")
        return None
        
    headers = {
        'Api-Key': os.environ.get("IDEOGRAM_API_KEY"),
        'Content-Type': 'application/json'
    }
    
    data = {
        'image_request': {
            'prompt': prompt,
            'aspect_ratio': 'ASPECT_10_16',
            'model': 'V_2',
            'magic_prompt': 'AUTO',  # Changed from magic_prompt_option
            'num_images': num_images,  # Keep using num_images
            'upscale': True
        }
    }
    
    try:
        logger.info(f"Making request to Ideogram API for {num_images} images...")
        logger.debug(f"Request headers (excluding auth): Content-Type: {headers['Content-Type']}")
        logger.debug(f"Request data: {data}")
        
        response = requests.post(
            'https://api.ideogram.ai/generate',
            headers=headers,
            json=data
        )
        
        # Log the complete response for debugging
        logger.info("=== IDEOGRAM API RESPONSE ===")
        logger.info(f"Status Code: {response.status_code}")
        response_json = response.json()
        logger.info("Response Content:")
        logger.info(json.dumps(response_json, indent=2))
        logger.info("===========================")
        
        if response.status_code != 200:
            logger.error(f"Ideogram API error. Status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return None
            
        # Get the response content and check each possible location for the magic prompt
        response_json = response.json()
        enhanced_prompts = []
        
        logger.info("Checking for magic prompts in response...")
        
        # Collect all unique prompts from the data array
        if 'data' in response_json and response_json['data']:
            for image_data in response_json['data']:
                if 'prompt' in image_data:
                    if image_data['prompt'] not in enhanced_prompts:
                        enhanced_prompts.append(image_data['prompt'])
        
        if not enhanced_prompts:
            logger.info("No magic prompts found in response")
            enhanced_prompt_text = "_Auto-enhancement active, but enhanced prompt not visible in API response_"
        else:
            logger.info(f"Found {len(enhanced_prompts)} magic prompts")
            enhanced_prompt_text = "\n\n".join([f"Image {i+1}:\n{prompt}" for i, prompt in enumerate(enhanced_prompts)])
        
        if 'data' in response_json and response_json['data']:
            image_urls = []
            for image_data in response_json['data']:
                if 'url' in image_data:
                    image_urls.append(image_data['url'])
            
            logger.info(f"Successfully extracted {len(image_urls)} image URLs from response")
            if len(image_urls) < num_images:
                logger.warning(f"Requested {num_images} images but only received {len(image_urls)}")
            
            # Return both image URLs and enhanced prompts
            return image_urls, enhanced_prompt_text
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