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
    
    # Get the prompt from the command
    prompt = command['text']
    
    # Tell user we're working on it
    respond(f"Working on generating images for: '{prompt}'...")
    
    try:
        # Generate image with Ideogram
        ideogram_image = generate_ideogram_image(prompt)
        
        if ideogram_image:
            logger.info("Successfully generated image")
            # Post the image back to Slack with a direct download link
            respond({
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Here's your generated image for: *{prompt}*"
                        }
                    },
                    {
                        "type": "image",
                        "title": {
                            "type": "plain_text",
                            "text": "Generated Image"
                        },
                        "image_url": ideogram_image,
                        "alt_text": "AI generated image"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"<{ideogram_image}|Download Image>"
                        }
                    }
                ]
            })
        else:
            logger.error("Failed to generate image")
            respond("Sorry, I couldn't generate an image. Please try again.")
            
    except Exception as e:
        logger.error(f"Error in command handler: {str(e)}")
        respond(f"Error: {str(e)}")

def generate_ideogram_image(prompt):
    """
    Generate an image using Ideogram API
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
        }
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
            image_url = response_json['data'][0]['url']
            logger.info("Successfully extracted image URL from response")
            return image_url
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