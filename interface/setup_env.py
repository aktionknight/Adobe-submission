#!/usr/bin/env python3
"""
Setup script to create the .env file for the Adobe 1B Interface.
This script will prompt you for your Adobe PDF Embed API Client ID.
"""

import os
import sys

def main():
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    
    print("Adobe 1B Interface Environment Setup")
    print("=" * 40)
    print()
    print("You need an Adobe PDF Embed API Client ID to use this application.")
    print("Get one from: https://www.adobe.com/go/dcsdks_credentials")
    print()
    
    # Check if .env already exists
    if os.path.exists(env_file):
        print(f"⚠️  .env file already exists at: {env_file}")
        overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    # Get Adobe API key
    adobe_api = input("Enter your Adobe PDF Embed API Client ID: ").strip()
    
    if not adobe_api:
        print("❌ Adobe API key is required!")
        return
    
    # Create .env content
    env_content = f"""# Adobe PDF Embed API Client ID
ADOBE_API={adobe_api}

# Server configuration
HOST=127.0.0.1
PORT=8000
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ .env file created successfully at: {env_file}")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start the server: python start_server.py")
        print("3. Open your browser to: http://127.0.0.1:8000/")
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        print(f"Please create it manually at: {env_file}")
        print("Content should be:")
        print(f"ADOBE_API={adobe_api}")

if __name__ == "__main__":
    main()
