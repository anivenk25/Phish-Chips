#!/bin/bash

# Test VoIP Server Deployment
echo "🧪 Testing VoIP Server Deployment"
echo "================================="

if [ -z "$1" ]; then
    echo "Usage: $0 <server-url>"
    echo "Example: $0 https://your-app-name.up.railway.app"
    echo "Example: $0 https://your-app-name.onrender.com"
    exit 1
fi

SERVER_URL=$1

echo "🔍 Testing server: $SERVER_URL"
echo ""

# Test root endpoint
echo "📡 Testing root endpoint..."
curl -s "$SERVER_URL" | jq . || echo "JSON parsing failed"
echo ""

# Test health endpoint
echo "🏥 Testing health endpoint..."
curl -s "$SERVER_URL/health" | jq . || echo "JSON parsing failed"
echo ""

# Test users endpoint
echo "👥 Testing users endpoint..."
curl -s "$SERVER_URL/users" | jq . || echo "JSON parsing failed"
echo ""

# Test WebSocket connection (basic check)
echo "🔌 Testing WebSocket availability..."
curl -I "$SERVER_URL/socket.io/" 2>/dev/null | head -1

echo ""
echo "✅ Basic tests completed!"
echo "💡 For full testing, use the Flutter app to make actual calls"
