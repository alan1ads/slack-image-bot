from dotenv import load_dotenv
import os

print("Starting test...")
load_dotenv()

print("\nChecking environment variables:")
print(f"SLACK_BOT_TOKEN: {'Present' if os.getenv('SLACK_BOT_TOKEN') else 'Missing'}")
print(f"SLACK_APP_TOKEN: {'Present' if os.getenv('SLACK_APP_TOKEN') else 'Missing'}")
print(f"IDEOGRAM_API_KEY: {'Present' if os.getenv('IDEOGRAM_API_KEY') else 'Missing'}")

print("\nActual values (first 5 characters only, for security):")
if os.getenv('SLACK_BOT_TOKEN'):
    print(f"SLACK_BOT_TOKEN starts with: {os.getenv('SLACK_BOT_TOKEN')[:5]}...")
if os.getenv('SLACK_APP_TOKEN'):
    print(f"SLACK_APP_TOKEN starts with: {os.getenv('SLACK_APP_TOKEN')[:5]}...")
if os.getenv('IDEOGRAM_API_KEY'):
    print(f"IDEOGRAM_API_KEY starts with: {os.getenv('IDEOGRAM_API_KEY')[:5]}...")