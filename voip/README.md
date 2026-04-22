# VoIP Flutter App

A professional VoIP (Voice over Internet Protocol) application built with Flutter and WebRTC.

## Features

✅ **Real-time Voice Calls** - High-quality P2P voice communication  
✅ **Auto Server Discovery** - Automatically detects signaling server  
✅ **Call Recording** - Automatic recording with server upload  
✅ **Professional UI** - Modern incoming call interface with animations  
✅ **Global Connectivity** - Works anywhere with internet connection  
✅ **Low Latency** - WebRTC P2P for minimal delay  

## Architecture

- **Frontend**: Flutter with WebRTC
- **Backend**: Node.js signaling server with Socket.IO
- **Communication**: Direct P2P after signaling
- **Recording**: Chunked uploads to server

## Quick Start

### 1. Start Signaling Server
```bash
cd signaling_server
npm install
node server.js
```

### 2. Run Flutter App
```bash
flutter pub get
flutter run
```

### 3. Make Calls
- Generate random 4-digit user ID automatically
- Enter target user ID to call
- Enjoy high-quality voice calls!

## Project Structure

```
voip/
├── lib/
│   ├── main.dart          # Main UI and dialer
│   └── voip_service.dart  # WebRTC service and recording
├── signaling_server/
│   ├── server.js          # Node.js signaling server
│   ├── package.json       # Server dependencies
│   └── recordings/        # Call recordings storage
├── android/               # Android platform code
└── pubspec.yaml          # Flutter dependencies
```

## Technologies

- **Flutter**: Cross-platform mobile framework
- **WebRTC**: Real-time peer-to-peer communication
- **Socket.IO**: WebSocket signaling
- **Node.js**: Signaling server
- **Multer**: File upload handling

## Deployment

Currently configured for local development. For global access:

1. Deploy signaling server to cloud (Heroku, AWS, etc.)
2. Update server URL in `lib/main.dart`
3. Add authentication for security
4. Distribute app to users

## Recording Feature

- Automatic recording during calls
- Chunked uploads every 30 seconds
- Server storage in `signaling_server/recordings/`
- WAV format with timestamps
