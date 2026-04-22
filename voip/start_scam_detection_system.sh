#!/bin/bash

echo "🚀 Starting VoIP Scam Detection System"
echo "======================================"

# Function to check if a port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        echo "✅ Port $1 is available"
        return 0
    fi
}

# Check required ports
echo "🔍 Checking ports..."
check_port 8000  # Scam Detection API
check_port 3000  # VoIP Signaling Server

echo ""
echo "📦 Installing dependencies..."

# Install Python dependencies
if [ ! -f "requirements_scam.txt" ]; then
    echo "Creating requirements_scam.txt..."
    cat > requirements_scam.txt << EOF
watchdog==3.0.0
requests==2.31.0
EOF
fi

pip install -r requirements_scam.txt

# Install Node.js dependencies
cd signaling_server
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi
cd ..

echo ""
echo "🎯 Starting servers..."
echo ""
echo "Please open 4 separate terminals and run these commands:"
echo ""
echo "📍 Terminal 1 - Scam Detection API:"
echo "   cd /home/antonyshane/linux/INTER_NIT/Inter_NIT_cybersec_v2"
echo "   python main.py"
echo ""
echo "📍 Terminal 2 - VoIP Signaling Server:"
echo "   cd /home/antonyshane/voip/signaling_server"
echo "   node server.js"
echo ""
echo "📍 Terminal 3 - Scam Detection Bridge:"
echo "   cd /home/antonyshane/voip/signaling_server"
echo "   python ../voip_scam_detector.py"
echo ""
echo "📍 Terminal 4 - Flutter VoIP App:"
echo "   cd /home/antonyshane/voip"
echo "   flutter run"
echo ""
echo "🔗 URLs:"
echo "   Scam Detection API: http://localhost:8000"
echo "   VoIP Signaling: http://localhost:3000"
echo ""
echo "✅ Setup complete! Start the terminals in order above."
