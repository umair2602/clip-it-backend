"""
Script to generate YouTube PO tokens using Node.js
Requires the 'youtube-po-token-generator' npm package to be installed:
npm install -g youtube-po-token-generator
"""

import subprocess
import json
import logging
import os
import sys
import shutil

def find_npx():
    """
    Find the npx executable in various locations
    """
    # Try direct path first
    npx_path = shutil.which("npx")
    if npx_path:
        return npx_path
    
    # Check common nvm locations
    possible_paths = [
        # RunPod nvm installation paths
        "/opt/conda/bin/npx",
        "/usr/local/bin/npx",
        "/usr/bin/npx",
        # NVM paths
        os.path.expanduser("~/.nvm/current/bin/npx"),
        os.path.expanduser("~/.nvm/versions/node/*/bin/npx"),
    ]
    
    for path in possible_paths:
        if "*" in path:
            # Handle wildcard paths
            import glob
            for match in glob.glob(path):
                if os.path.exists(match) and os.access(match, os.X_OK):
                    return match
        elif os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    # If we can't find npx directly, try to find node and run npx through it
    node_path = shutil.which("node")
    if node_path:
        node_dir = os.path.dirname(node_path)
        npx_in_node_dir = os.path.join(node_dir, "npx")
        if os.path.exists(npx_in_node_dir) and os.access(npx_in_node_dir, os.X_OK):
            return npx_in_node_dir
    
    # If all else fails, try running npm to find where npx might be
    try:
        npm_path = shutil.which("npm")
        if npm_path:
            result = subprocess.run(
                [npm_path, "bin", "-g"],
                capture_output=True,
                text=True,
                check=True
            )
            npm_bin_path = result.stdout.strip()
            npx_in_npm_bin = os.path.join(npm_bin_path, "npx")
            if os.path.exists(npx_in_npm_bin) and os.access(npx_in_npm_bin, os.X_OK):
                return npx_in_npm_bin
    except Exception as e:
        logging.warning(f"Error finding npx through npm: {e}")
    
    return None

def generate_po_token():
    """
    Generate a YouTube PO token using the Node.js package
    Returns a tuple of (po_token, visitor_data) or (None, None) if failed
    """
    try:
        logging.info("Generating YouTube PO token using Node.js")
        
        # Find npx
        npx_path = find_npx()
        if not npx_path:
            logging.error("Could not find npx executable. Make sure Node.js and npm are properly installed.")
            logging.info("Attempting to run with npm directly...")
            
            # Try using npm directly if npx isn't found
            try:
                result = subprocess.run(
                    ["npm", "exec", "--", "youtube-po-token-generator"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError:
                logging.error("Failed to run with npm exec. Installing package globally...")
                
                # Try installing the package globally and then running it
                try:
                    subprocess.run(
                        ["npm", "install", "-g", "youtube-po-token-generator"],
                        check=True
                    )
                    result = subprocess.run(
                        ["npm", "exec", "--", "youtube-po-token-generator"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to install and run package: {e}")
                    return None, None
        else:
            logging.info(f"Found npx at: {npx_path}")
            # Run the Node.js command to generate a token
            result = subprocess.run(
                [npx_path, "youtube-po-token-generator"],
                capture_output=True,
                text=True,
                check=True
            )
        
        # Parse the output as JSON
        output = json.loads(result.stdout)
        
        # Extract the token and visitor data
        po_token = output.get("poToken")
        visitor_data = output.get("visitorData")
        
        if not po_token or not visitor_data:
            logging.error("Failed to extract PO token or visitor data from output")
            return None, None
            
        logging.info("Successfully generated PO token")
        
        # Update the tokens file
        tokens_file = os.path.join(os.path.dirname(__file__), "youtube_tokens.py")
        with open(tokens_file, "w") as f:
            f.write('"""\n')
            f.write('Store YouTube PO token and visitor data for authentication\n')
            f.write('Auto-generated by generate_token.py\n')
            f.write('"""\n\n')
            f.write(f'PO_TOKEN = "{po_token}"\n')
            f.write(f'VISITOR_DATA = "{visitor_data}"\n')
            
        logging.info(f"Updated tokens file: {tokens_file}")
        return po_token, visitor_data
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Node.js command: {e}")
        logging.error(f"Output: {e.stderr}")
        return None, None
    except Exception as e:
        logging.error(f"Error generating PO token: {e}")
        return None, None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    po_token, visitor_data = generate_po_token()
    if po_token and visitor_data:
        print(f"PO Token: {po_token[:20]}...")
        print(f"Visitor Data: {visitor_data[:20]}...")
        print("Tokens have been saved to youtube_tokens.py")
    else:
        print("Failed to generate tokens")
        sys.exit(1) 