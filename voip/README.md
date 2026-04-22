# VoIP Flutter App

Cross-platform VoIP client with WebRTC for real-time voice calls and automatic call recording with scam detection upload.

## Setup

```bash
# Install dependencies
flutter pub get

# Run on connected device/emulator
flutter run
```

## How It Works

1. App generates a random 4-digit user ID on launch
2. Auto-discovers the signaling server by scanning the local network
3. Uses WebRTC for peer-to-peer voice calls via Socket.IO signaling
4. Records calls and uploads chunks to the signaling server for AI analysis

## Structure

```
lib/
├── main.dart          # Dialer UI, incoming call screen, call screen
└── voip_service.dart  # WebRTC connection, recording, server upload

signaling_server/
├── server.js          # Node.js + Socket.IO + Multer upload + scam API bridge
├── Dockerfile
└── package.json
```

## Signaling Server

```bash
cd signaling_server
npm install
node server.js    # Runs on port 3000
```

The signaling server handles WebRTC signaling, stores call recordings, and notifies the backend API (`POST /force-analyze/{filename}`) for scam analysis.
