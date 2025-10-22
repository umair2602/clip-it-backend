#!/usr/bin/env python3
"""
Start ngrok HTTPS tunnel for TikTok OAuth
Simple script to start ngrok with the correct path
"""

import subprocess
import time
import requests
import os


def start_ngrok():
    """Start ngrok tunnel"""
    print("🚀 Starting ngrok HTTPS tunnel...")

    # ngrok path from winget installation
    ngrok_path = os.path.join(
        os.environ["LOCALAPPDATA"],
        "Microsoft",
        "WinGet",
        "Packages",
        "Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe",
        "ngrok.exe",
    )

    print(f"📋 ngrok path: {ngrok_path}")

    try:
        # Start ngrok in background
        cmd = [ngrok_path, "http", "8000"]
        print(f"📋 Command: {' '.join(cmd)}")

        # Start the process
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for ngrok to start
        print("⏳ Waiting for ngrok to start...")
        time.sleep(5)

        # Check if process is running
        if process.poll() is None:
            print("✅ ngrok tunnel started successfully!")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ ngrok failed to start")
            print(f"📄 stdout: {stdout}")
            print(f"📄 stderr: {stderr}")
            return None

    except Exception as e:
        print(f"❌ Error starting ngrok: {e}")
        return None


def get_ngrok_url():
    """Get the public HTTPS URL from ngrok"""
    print("\n🔗 Getting ngrok public URL...")

    try:
        # ngrok provides a local API to get the public URL
        response = requests.get("http://localhost:4040/api/tunnels", timeout=5)

        if response.status_code == 200:
            data = response.json()
            tunnels = data.get("tunnels", [])

            for tunnel in tunnels:
                if tunnel.get("proto") == "https":
                    public_url = tunnel.get("public_url")
                    if public_url:
                        print(f"✅ HTTPS URL: {public_url}")
                        return public_url

            print("❌ No HTTPS tunnel found")
            return None
        else:
            print(f"❌ Failed to get tunnel info: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error getting ngrok URL: {e}")
        return None


def update_env_file(https_url):
    """Update .env file with HTTPS redirect URI"""
    print(f"\n⚙️  Updating .env file...")

    callback_url = f"{https_url}/tiktok/callback/"

    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"📁 Found existing .env file")

        # Read current content
        with open(env_file, "r") as f:
            content = f.read()

        # Update TIKTOK_REDIRECT_URI
        if "TIKTOK_REDIRECT_URI=" in content:
            # Replace existing value
            import re

            new_content = re.sub(
                r"TIKTOK_REDIRECT_URI=.*",
                f"TIKTOK_REDIRECT_URI={callback_url}",
                content,
            )

            with open(env_file, "w") as f:
                f.write(new_content)

            print(f"✅ Updated .env file with new redirect URI")
        else:
            # Add new line
            with open(env_file, "a") as f:
                f.write(f"\nTIKTOK_REDIRECT_URI={callback_url}\n")

            print(f"✅ Added TIKTOK_REDIRECT_URI to .env file")
    else:
        print(f"📁 Creating new .env file")

        # Create new .env file

        print(f"✅ Created .env file with HTTPS redirect URI")

    return callback_url


def main():
    """Main function"""
    print("🔒 Start ngrok HTTPS Tunnel for TikTok OAuth")
    print("=" * 50)

    # Start ngrok
    ngrok_process = start_ngrok()
    if not ngrok_process:
        print("❌ Failed to start ngrok tunnel.")
        return 1

    try:
        # Wait a bit more for ngrok to fully start
        print("⏳ Waiting for ngrok to fully start...")
        time.sleep(3)

        # Get the public HTTPS URL
        https_url = get_ngrok_url()
        if not https_url:
            print("❌ Failed to get ngrok URL.")
            return 1

        # Update environment variables
        callback_url = update_env_file(https_url)

        print(f"\n🎉 ngrok tunnel is running!")
        print(f"📋 Your HTTPS URL: {https_url}")
        print(f"📋 Callback URL: {callback_url}")

        print(f"\n🌐 Next Steps:")
        print(f"1. Update TikTok Developer Console:")
        print(f"   Redirect URI: {callback_url}")
        print(f"2. Restart your FastAPI server")
        print(f"3. Test with: python quick_tiktok_test.py")

        print(f"\n⚠️  Keep this terminal open to keep ngrok running!")
        print(f"   Press Ctrl+C to stop ngrok when done testing")

        # Keep the process running
        try:
            ngrok_process.wait()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping ngrok tunnel...")
            ngrok_process.terminate()
            ngrok_process.wait()
            print("✅ ngrok tunnel stopped")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
