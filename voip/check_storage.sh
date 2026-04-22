#!/bin/bash

# Check VoIP Recording Storage Status
echo "📁 VoIP Recording Storage Status"
echo "================================"

if [ -z "$1" ]; then
    echo "Usage: $1 <your-railway-app-url>"
    echo "Example: $0 https://your-app-abc123.up.railway.app"
    echo ""
    echo "This script checks where your call recordings are being stored."
    exit 1
fi

RAILWAY_URL=$1

echo "🔍 Checking storage info at: $RAILWAY_URL"
echo ""

# Check storage information
echo "📊 Storage Information:"
echo "====================="
curl -s "$RAILWAY_URL/storage-info" | jq . 2>/dev/null || echo "Error: Could not fetch storage info"

echo ""
echo ""

# Check analysis stats
echo "📈 Analysis Statistics:"
echo "======================"
curl -s "$RAILWAY_URL/analysis-stats" | jq . 2>/dev/null || echo "Error: Could not fetch analysis stats"

echo ""
echo ""

echo "💡 Storage Recommendations:"
echo "=========================="
echo "• ⚠️  If storage_type shows 'Ephemeral': Recordings will be lost on restart"
echo "• ✅ If storage_type shows 'Persistent': Recordings are permanently stored"
echo "• 🔧 To add persistent storage: Go to Railway dashboard → Add Volume"
echo "• 📏 Volume mount path: /app/recordings"
echo "• 💰 Cost: ~$0.25/GB/month"
