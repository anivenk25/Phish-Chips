#!/bin/bash

# VoIP Signaling Server - Railway Deployment Script (FREE)
echo "🚀 VoIP Signaling Server - Railway Deployment (FREE)"
echo "=================================================="

echo "Railway offers $5/month in free credits - perfect for VoIP apps!"
echo "✅ No sleeping issues - always available for mobile data calls"
echo ""

# Check if git is initialized
cd signaling_server
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    git add .
    git commit -m "VoIP signaling server for Railway deployment"
    git branch -M main
fi

echo "🚂 Railway Deployment Instructions:"
echo "=================================="
echo ""
echo "1. 📋 Go to: https://railway.app"
echo "2. 🔐 Sign up/login with GitHub"
echo "3. ➕ Click 'Start a New Project'"
echo "4. 📁 Select 'Deploy from GitHub repo'"
echo "5. 🔗 Choose your VoIP signaling server repo"
echo "6. 🚀 Railway auto-deploys your Node.js app!"
echo ""

echo "💡 If you don't have a GitHub repo yet:"
echo "   git remote add origin https://github.com/yourusername/voip-server.git"
echo "   git push -u origin main"
echo ""

echo "📡 After deployment:"
echo "• Your app will be at: https://your-app-name.up.railway.app"
echo "• Update lib/main.dart with this URL"
echo "• Rebuild APK: flutter build apk --release"
echo ""

echo "✅ Benefits of Railway for VoIP:"
echo "• 🏠 WiFi: Auto-detects local server"
echo "• 📱 Mobile Data: Uses Railway cloud server"
echo "• ⚡ No sleeping (unlike free Heroku alternatives)"
echo "• 💰 $5 free credits (lasts months for VoIP)"
echo ""

read -p "Press Enter to continue with Railway deployment guide..."

echo ""
echo "🔧 Next Steps After Railway Deployment:"
echo "======================================="
echo "1. Copy your Railway app URL"
echo "2. Replace 'https://your-app-name.up.railway.app' in lib/main.dart"
echo "3. Run: flutter build apk --release"
echo "4. Install APK and test with mobile data! 📱"
