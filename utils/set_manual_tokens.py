"""
Script to manually set YouTube PO token and visitor data
"""

import os
import sys

def set_tokens(po_token, visitor_data):
    """
    Save the provided PO token and visitor data to the tokens file
    """
    # Get the path to the tokens file
    tokens_file = os.path.join(os.path.dirname(__file__), "youtube_tokens.py")
    
    # Write the tokens to the file
    with open(tokens_file, "w") as f:
        f.write('"""\n')
        f.write('Store YouTube PO token and visitor data for authentication\n')
        f.write('Manually set by set_manual_tokens.py\n')
        f.write('"""\n\n')
        f.write(f'PO_TOKEN = "{po_token}"\n')
        f.write(f'VISITOR_DATA = "{visitor_data}"\n')
    
    print(f"Tokens saved to {tokens_file}")
    return True

if __name__ == "__main__":
    # Check if tokens are provided as arguments
    if len(sys.argv) != 3:
        print("Usage: python set_manual_tokens.py <po_token> <visitor_data>")
        print("\nHow to get these values:")
        print("1. Open an incognito/private browser window")
        print("2. Go to any YouTube video")
        print("3. Open developer tools (F12)")
        print("4. Go to Network tab and filter for 'player'")
        print("5. Play the video")
        print("6. Find a request to 'youtubei/v1/player'")
        print("7. In the request payload, find:")
        print("   - poToken in serviceIntegrityDimensions.poToken")
        print("   - visitorData in context.client.visitorData")
        sys.exit(1)
    
    # Get the tokens from the arguments
    po_token = sys.argv[1]
    visitor_data = sys.argv[2]
    
    # Set the tokens
    set_tokens(po_token, visitor_data) 