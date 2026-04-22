#!/bin/bash

echo "🚀 Setting up VoIP Scam Detection Integration"

# Install Python dependencies for VoIP integration
echo "📦 Installing VoIP integration dependencies..."
pip install -r requirements_scam.txt

# Make the detector script executable
chmod +x voip_scam_detector.py

echo "✅ VoIP integration setup complete!"
echo ""
echo "Next steps:"
echo "1. Start your scam detection API server first"
echo "2. Start the VoIP signaling server: node server.js"
echo "3. Run the scam detector: python voip_scam_detector.py"
echo "4. Start your Flutter VoIP app"
echo ""
echo "The system will automatically detect and analyze call recordings!"
