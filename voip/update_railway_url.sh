#!/bin/bash

# Update Flutter App with Railway URL
echo "🔧 Update Flutter App with Railway URL"
echo "====================================="

echo "After deploying to Railway, follow these steps:"
echo ""

read -p "Enter your Railway app URL (e.g., https://your-app-abc123.up.railway.app): " RAILWAY_URL

if [ -z "$RAILWAY_URL" ]; then
    echo "❌ No URL provided. Please run this script again with your Railway URL."
    exit 1
fi

echo ""
echo "🔄 Updating lib/main.dart with Railway URL..."

# Update the Flutter app with the Railway URL
sed -i "s|https://your-app-name\.up\.railway\.app|$RAILWAY_URL|g" lib/main.dart

echo "✅ Updated lib/main.dart with Railway URL: $RAILWAY_URL"
echo ""

echo "🏗️ Building new APK with Railway support..."
flutter build apk --release

echo ""
echo "✅ APK built successfully!"
echo "📱 Your VoIP app now supports:"
echo "   • 🏠 WiFi: Auto-detects local server"
echo "   • 📱 Mobile Data: Uses Railway cloud server ($RAILWAY_URL)"
echo ""
echo "🔄 Install the new APK and test mobile data calls!"
echo "📍 APK location: build/app/outputs/flutter-apk/app-release.apk"
