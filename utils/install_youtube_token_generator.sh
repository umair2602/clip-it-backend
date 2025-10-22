#!/bin/bash
# Script to install the YouTube PO token generator package

# Print current Node.js and npm versions
echo "Node.js version:"
node -v
echo "npm version:"
npm -v

# Make sure we're using the nvm-installed Node.js
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install the package globally
echo "Installing youtube-po-token-generator package..."
npm install -g youtube-po-token-generator

# Verify installation
echo "Verifying installation..."
npm list -g youtube-po-token-generator

echo "Installation complete. You can now run the generate_token.py script." 