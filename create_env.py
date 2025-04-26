import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')

# Create the .env file
print(f"Creating .env file at: {env_path}")

# Write the template to the file
with open(env_path, 'w') as f:
    f.write("SLACK_BOT_TOKEN=xoxb-your-bot-token\n")
    f.write("SLACK_APP_TOKEN=xapp-your-app-token\n")
    f.write("IDEOGRAM_API_KEY=your-ideogram-key\n")

print("\nFile created successfully!")
print("Please edit the .env file and replace the placeholder values with your actual tokens.")
print("\nTo verify the file exists, this script will now try to read it:")

# Verify the file exists and can be read
try:
    with open(env_path, 'r') as f:
        content = f.read()
        print("\nFile contents (showing structure but not actual tokens):")
        for line in content.splitlines():
            key = line.split('=')[0]
            print(f"{key}=***")
except Exception as e:
    print(f"\nError reading file: {str(e)}")