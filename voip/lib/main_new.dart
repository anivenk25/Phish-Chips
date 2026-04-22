import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:permission_handler/permission_handler.dart';
import 'voip_service.dart';
import 'dart:math';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VoIP Dialer',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const DialerPage(),
    );
  }
}

class DialerPage extends StatefulWidget {
  const DialerPage({super.key});

  @override
  State<DialerPage> createState() => _DialerPageState();
}

class _DialerPageState extends State<DialerPage> {
  String _phoneNumber = '';
  final VoIPService _voipService = VoIPService();
  String _currentUserId = '';
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initializeVoIP();
  }

  Future<void> _initializeVoIP() async {
    // Request permissions
    await _requestPermissions();
    
    // Generate a random user ID for this session
    _currentUserId = 'user_${Random().nextInt(10000)}';
    
    // Initialize VoIP service
    await _voipService.initialize(
      serverUrl: 'http://localhost:3000', // Change this to your server IP
      userId: _currentUserId,
    );

    // Set up callbacks
    _voipService.onIncomingCall = (String fromUser) {
      _showIncomingCallDialog(fromUser);
    };

    _voipService.onCallEnded = () {
      if (mounted) {
        Navigator.of(context).popUntil((route) => route.isFirst);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Call ended')),
        );
      }
    };

    _voipService.onError = (String error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $error')),
        );
      }
    };

    setState(() {
      _isInitialized = true;
    });
  }

  Future<void> _requestPermissions() async {
    await [
      Permission.microphone,
      Permission.camera,
    ].request();
  }

  void _showIncomingCallDialog(String fromUser) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Text('Incoming Call'),
        content: Text('Call from: $fromUser'),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              _voipService.rejectIncomingCall();
            },
            child: const Text('Reject'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.of(context).pop();
              _voipService.acceptIncomingCall();
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => CallingPage(
                    phoneNumber: fromUser,
                    isIncomingCall: true,
                  ),
                ),
              );
            },
            child: const Text('Accept'),
          ),
        ],
      ),
    );
  }

  void _addDigit(String digit) {
    setState(() {
      if (_phoneNumber.length < 15) {
        _phoneNumber += digit;
      }
    });
  }

  void _deleteDigit() {
    setState(() {
      if (_phoneNumber.isNotEmpty) {
        _phoneNumber = _phoneNumber.substring(0, _phoneNumber.length - 1);
      }
    });
  }

  void _callNumber() {
    if (_phoneNumber.isNotEmpty && _isInitialized) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => CallingPage(
            phoneNumber: _phoneNumber,
            isIncomingCall: false,
          ),
        ),
      );
    }
  }

  Widget _buildDialButton(String digit) {
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: ElevatedButton(
          onPressed: () => _addDigit(digit),
          child: Text(digit, style: const TextStyle(fontSize: 24)),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('VoIP Dialer'),
        backgroundColor: Colors.deepPurple,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: Icon(
              _isInitialized ? Icons.wifi : Icons.wifi_off,
              color: _isInitialized ? Colors.green : Colors.red,
            ),
            onPressed: null,
          ),
        ],
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Text(
                  'Your ID: $_currentUserId',
                  style: const TextStyle(fontSize: 14, color: Colors.grey),
                ),
                const SizedBox(height: 8),
                Text(
                  _phoneNumber.isEmpty ? 'Enter user ID to call' : _phoneNumber,
                  style: const TextStyle(fontSize: 32, letterSpacing: 2),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),
          for (var row in [
            ['1', '2', '3'],
            ['4', '5', '6'],
            ['7', '8', '9'],
            ['*', '0', '#'],
          ])
            Row(children: row.map(_buildDialButton).toList()),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(
                icon: const Icon(Icons.backspace),
                onPressed: _deleteDigit,
                iconSize: 32,
              ),
              const SizedBox(width: 40),
              ElevatedButton.icon(
                onPressed: (_phoneNumber.isNotEmpty && _isInitialized) 
                    ? _callNumber 
                    : null,
                icon: const Icon(Icons.call),
                label: const Text('Call'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 12,
                  ),
                  textStyle: const TextStyle(fontSize: 20),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          if (!_isInitialized)
            const CircularProgressIndicator()
          else
            const Text(
              'Ready to make calls',
              style: TextStyle(color: Colors.green),
            ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _voipService.disconnect();
    super.dispose();
  }
}

class CallingPage extends StatefulWidget {
  final String phoneNumber;
  final bool isIncomingCall;
  
  const CallingPage({
    super.key, 
    required this.phoneNumber,
    required this.isIncomingCall,
  });

  @override
  State<CallingPage> createState() => _CallingPageState();
}

class _CallingPageState extends State<CallingPage> {
  final VoIPService _voipService = VoIPService();
  RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  RTCVideoRenderer _remoteRenderer = RTCVideoRenderer();
  bool _isMuted = false;
  bool _isSpeakerOn = false;
  bool _isConnected = false;

  @override
  void initState() {
    super.initState();
    _initializeRenderers();
    _setupVoIPCallbacks();
    
    if (!widget.isIncomingCall) {
      // Outgoing call
      _voipService.startCall(widget.phoneNumber);
    }
  }

  void _initializeRenderers() async {
    await _localRenderer.initialize();
    await _remoteRenderer.initialize();
  }

  void _setupVoIPCallbacks() {
    _voipService.onLocalStream = (MediaStream stream) {
      setState(() {
        _localRenderer.srcObject = stream;
      });
    };

    _voipService.onRemoteStream = (MediaStream stream) {
      setState(() {
        _remoteRenderer.srcObject = stream;
        _isConnected = true;
      });
    };

    _voipService.onCallEnded = () {
      if (mounted) {
        Navigator.of(context).pop();
      }
    };

    _voipService.onError = (String error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Call error: $error')),
        );
      }
    };
  }

  void _toggleMute() {
    setState(() {
      _isMuted = !_isMuted;
      _voipService.toggleMicrophone();
    });
  }

  void _toggleSpeaker() {
    setState(() {
      _isSpeakerOn = !_isSpeakerOn;
      _voipService.toggleSpeaker(_isSpeakerOn);
    });
  }

  void _endCall() {
    _voipService.endCall();
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            // Header
            Container(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Text(
                    widget.phoneNumber,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _isConnected 
                        ? 'Connected' 
                        : (widget.isIncomingCall ? 'Incoming call...' : 'Calling...'),
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 16,
                    ),
                  ),
                ],
              ),
            ),
            
            // Video area
            Expanded(
              child: Stack(
                children: [
                  // Remote video (full screen)
                  if (_remoteRenderer.srcObject != null)
                    RTCVideoView(_remoteRenderer, mirror: false)
                  else
                    Container(
                      width: double.infinity,
                      height: double.infinity,
                      color: Colors.grey[900],
                      child: const Center(
                        child: Icon(
                          Icons.person,
                          size: 100,
                          color: Colors.white54,
                        ),
                      ),
                    ),
                  
                  // Local video (small overlay)
                  if (_localRenderer.srcObject != null)
                    Positioned(
                      top: 20,
                      right: 20,
                      child: Container(
                        width: 120,
                        height: 160,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: Colors.white, width: 2),
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: RTCVideoView(_localRenderer, mirror: true),
                        ),
                      ),
                    ),
                ],
              ),
            ),
            
            // Controls
            Container(
              padding: const EdgeInsets.all(20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  // Mute button
                  CircleAvatar(
                    radius: 30,
                    backgroundColor: _isMuted ? Colors.red : Colors.white24,
                    child: IconButton(
                      icon: Icon(
                        _isMuted ? Icons.mic_off : Icons.mic,
                        color: Colors.white,
                        size: 28,
                      ),
                      onPressed: _toggleMute,
                    ),
                  ),
                  
                  // End call button
                  CircleAvatar(
                    radius: 35,
                    backgroundColor: Colors.red,
                    child: IconButton(
                      icon: const Icon(
                        Icons.call_end,
                        color: Colors.white,
                        size: 32,
                      ),
                      onPressed: _endCall,
                    ),
                  ),
                  
                  // Speaker button
                  CircleAvatar(
                    radius: 30,
                    backgroundColor: _isSpeakerOn ? Colors.blue : Colors.white24,
                    child: IconButton(
                      icon: Icon(
                        _isSpeakerOn ? Icons.volume_up : Icons.volume_down,
                        color: Colors.white,
                        size: 28,
                      ),
                      onPressed: _toggleSpeaker,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _localRenderer.dispose();
    _remoteRenderer.dispose();
    super.dispose();
  }
}
