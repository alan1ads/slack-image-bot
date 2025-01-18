from dotenv import load_dotenv
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import requests
import time
import logging
import threading
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
    num_images = 4  # default to 4 images
    
    if len(parts) > 1:
        try:
            requested_num = int(parts[1].strip())
            num_images = min(max(1, requested_num), 4)  # still limit between 1 and 4 images
        except ValueError:
            num_images = 4  # keep default of 4 if invalid input
    
    # Tell user we're working on it
    respond(f"Working on generating images for: '{prompt}'...")
    
    try:
        # Generate images with Ideogram
        ideogram_images = generate_ideogram_image(prompt, num_images)
        
        if ideogram_images:
            logger.info(f"Successfully generated {len(ideogram_images)} images")
            
            # Create blocks for Slack message
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Here are your {len(ideogram_images)} generated images for: *{prompt}*"
                    }
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
                            "text": f"<{image_url}|Download Image {i}>"
                        }
                    }
                ])
            
            respond({"blocks": blocks})
        else:
            logger.error("Failed to generate image")
            respond("Sorry, I couldn't generate an image. Please try again.")
            
    except Exception as e:
        logger.error(f"Error in command handler: {str(e)}")
        respond(f"Error: {str(e)}")

def generate_ideogram_image(prompt, num_images=4):
    """
    Generate images using Ideogram API
    
    Args:
        prompt (str): The prompt for image generation
        num_images (int): Number of images to generate (default: 4)
    
    Returns:
        list: List of image URLs
    """
    logger.info(f"Attempting to generate image with prompt: {prompt}")
    
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
            'magic_prompt_option': 'AUTO'
        },
        'num_images': num_images
    }
    
    try:
        logger.info("Making request to Ideogram API...")
        logger.debug(f"Request headers (excluding auth): Content-Type: {headers['Content-Type']}")
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
        logger.debug(f"Response JSON structure: {list(response_json.keys())}")
        
        # Check if 'data' key exists and contains image information
        if 'data' in response_json and response_json['data']:
            image_urls = [image['url'] for image in response_json['data']]
            logger.info(f"Successfully extracted {len(image_urls)} image URLs from response")
            return image_urls
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