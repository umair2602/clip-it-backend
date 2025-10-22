"""
Store YouTube PO token and visitor data for authentication

How to get these values:
1. Open an incognito/private browser window
2. Go to any YouTube video
3. Open developer tools (F12)
4. Go to Network tab and filter for "player"
5. Play the video
6. Find a request to "youtubei/v1/player"
7. In the request payload, find:
   - poToken in serviceIntegrityDimensions.poToken
   - visitorData in context.client.visitorData
8. Copy these values below

Note: These tokens expire periodically, so you may need to update them
if downloads start failing again.
"""

# Example values (replace with your actual values):
# PO_TOKEN = "AJYo2UhXJKqYKWUTxpWbNQbNNAFhKpyRQDfCUoKe-xHEJFBuLTCfCRvnAFJxe0vHBCRRPkwYOBLNgXG3..."
# VISITOR_DATA = "CgtoWlpfLVBQc0lXOCiQs..."

# Replace these values with the ones you extracted from your browser
PO_TOKEN = "your_po_token_here"
VISITOR_DATA = "your_visitor_data_here" 