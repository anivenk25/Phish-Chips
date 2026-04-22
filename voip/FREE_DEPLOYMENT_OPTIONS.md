# Railway Deployment Guide for VoIP Signaling Server (FREE)

## 🚂 Railway - Perfect for VoIP Apps!

### Why Railway is Best for VoIP:
- ✅ **$5/month in free credits** (enough for small VoIP apps)
- ✅ **No sleeping** (always available for mobile data calls)
- ✅ **Easy deployment** like Heroku
- ✅ **Excellent for Node.js** applications
- ✅ **Fast global CDN** for better call quality

## Quick Deployment Steps

### 1. Create GitHub Repository (if needed)
```bash
cd signaling_server
git init
git add .
git commit -m "VoIP signaling server"
git branch -M main
git remote add origin https://github.com/yourusername/voip-signaling-server.git
git push -u origin main
```

### 2. Deploy on Railway
1. 🌐 Go to **https://railway.app**
2. 🔐 **Sign up/login** with GitHub
3. ➕ Click **"Start a New Project"**
4. 📁 Select **"Deploy from GitHub repo"**
5. 🔗 Choose your **voip-signaling-server** repository
6. 🚀 Railway automatically detects Node.js and deploys!

### 3. Get Your App URL
- Railway provides a URL like: `https://your-app-name.up.railway.app`
- Copy this URL for your Flutter app

### 4. Update Flutter App
Replace the Railway URL in `lib/main.dart`:
```dart
const String cloudServerUrl = 'https://your-app-name.up.railway.app';
```

### 5. Test Mobile Data Calls
```bash
flutter build apk --release
```
Install the APK and test calls with mobile data! 📱

## How It Works

### 📶 Network Detection:
- **WiFi**: App auto-detects local signaling server
- **Mobile Data**: App automatically uses Railway cloud server
- **Fallback**: If local fails, uses Railway as backup

### 🎯 Perfect for VoIP:
- No sleeping issues (unlike free alternatives)
- Always available for incoming calls
- Fast response times for WebRTC signaling
- Handles file uploads for call recordings

## Cost Breakdown
- **Railway**: $5/month free credit
  - Typically uses ~$1-2/month for VoIP signaling
  - Free credits last 2-3 months
  - Perfect for development and testing

## Alternative Free Options

### Render.com (100% Free but with limitations)
- ⚠️ **Sleeps after 15 min** (bad for VoIP)
- May miss incoming calls
- Good for testing only

### Fly.io (Good Free Tier)
- 3 free VMs with no sleeping
- More complex setup
- Good alternative if you need more control

## Railway Advantages for VoIP
1. **Always On**: No sleeping = no missed calls
2. **Fast Deployment**: GitHub integration
3. **Auto Scaling**: Handles traffic spikes
4. **Built-in Metrics**: Monitor your app
5. **Custom Domains**: Professional URLs
6. **Environment Variables**: Easy configuration

## After Deployment Checklist
- [ ] Railway app deployed successfully
- [ ] App URL copied
- [ ] `lib/main.dart` updated with Railway URL
- [ ] APK rebuilt with `flutter build apk --release`
- [ ] Mobile data calling tested
- [ ] WiFi local server still works

## Troubleshooting
- **Can't connect on mobile data**: Check Railway app URL in main.dart
- **Local WiFi not working**: Make sure local server is running
- **Calls dropping**: Railway free tier has enough resources for VoIP

Railway is the perfect solution for your VoIP app - reliable, affordable, and always available! 🚀
