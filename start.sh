#!/bin/bash

# Railway startup script
# This script determines whether to start the API or worker based on environment

if [ "$RAILWAY_SERVICE_NAME" = "worker" ]; then
    echo "Starting worker service..."
    python worker.py
else
    echo "Starting API service..."
    python app.py
fi 