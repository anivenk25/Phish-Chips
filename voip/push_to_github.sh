#!/bin/bash

# GitHub Push Script for VoIP App
echo "🚀 Pushing VoIP App to GitHub"
echo "============================="

echo "📋 Your VoIP project is ready to push!"
echo ""

read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter your repository name (e.g., voip-app-flutter): " REPO_NAME

echo ""
echo "🔗 Setting up remote repository..."

# Add GitHub remote
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

echo "📤 Pushing to GitHub..."

# Push to GitHub
git push -u origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo "📍 Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "🚂 Next step: Deploy to Railway"
echo "1. Go to https://railway.app"
echo "2. Sign in with GitHub"
echo "3. Deploy your signaling_server folder"
echo "4. Update lib/main.dart with Railway URL"
echo "5. Rebuild APK for mobile data support!"
