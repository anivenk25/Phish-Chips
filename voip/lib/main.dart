import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;
import 'package:connectivity_plus/connectivity_plus.dart';
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

  Future<String> _getServerUrl() async {
    List<String> possibleIPs = [];

    // Step 1: Check network connectivity and get network info
    try {
      var connectivityResult = await Connectivity().checkConnectivity();
      print('📶 Network type: $connectivityResult');

      if (connectivityResult == ConnectivityResult.none) {
        print('⚠️ No network connection detected');
        return 'http://localhost:3000';
      }

      // If on mobile data, try cloud server first
      if (connectivityResult == ConnectivityResult.mobile) {
        print('📱 Mobile data detected - trying Railway cloud server first');

        // Try Railway cloud server first for mobile data
        const String cloudServerUrl = 'https://your-app-name.up.railway.app';
        try {
          final response = await http
              .get(Uri.parse(cloudServerUrl))
              .timeout(const Duration(seconds: 5));

          if (response.statusCode == 200 &&
              (response.body.contains('VoIP') ||
                  response.body.contains('signaling'))) {
            print('✅ Found Railway VoIP server at: $cloudServerUrl');
            return cloudServerUrl;
          }
        } catch (e) {
          print('☁️ Railway cloud server not accessible: $e');
        }
      }
    } catch (e) {
      print('Connectivity check failed: $e');
    }

    // Step 2: Try to detect current device's network automatically
    try {
      // Attempt to get external IP for better network detection context
      await http
          .get(Uri.parse('https://api.ipify.org?format=json'))
          .timeout(const Duration(seconds: 3));

      // Continue with local network scanning regardless of external IP result
    } catch (e) {
      // External IP detection failed, continue with local scanning
      print('External IP detection failed, scanning local networks: $e');
    }

    // Step 3: Comprehensive network scanning
    // Common network ranges - ordered by probability
    List<String> networkBases = [
      '192.168.1', // Most common home networks
      '192.168.0', // Common router default
      '192.168.2', // Some routers use this
      '192.168.4', // Some mobile hotspots
      '10.0.0', // Corporate/some home networks
      '10.0.1', // Alternative corporate range
      '172.16.0', // Private networks
      '172.20.10', // iPhone hotspot default
      '192.168.43', // Android hotspot default
      '192.168.137', // Windows mobile hotspot default
    ];

    // For each network base, scan the most likely IPs
    for (String base in networkBases) {
      // First try the most common server/router IPs
      possibleIPs.addAll([
        '$base.1', // Most common router IP
        '$base.254', // Alternative router IP
        '$base.100', // Common server IP
        '$base.10', // Common server IP
        '$base.2', // Sometimes used for servers
        '$base.5', // Sometimes used for servers
      ]);

      // Then scan a broader range for development servers
      for (int i = 3; i <= 50; i++) {
        if (i != 10 && i != 100 && i != 254) {
          // Skip already added IPs
          possibleIPs.add('$base.$i');
        }
      }
    }

    // Add localhost variants
    possibleIPs.addAll([
      '127.0.0.1', // Localhost IPv4
      '0.0.0.0', // All interfaces
    ]);

    // Remove duplicates while preserving order
    possibleIPs = possibleIPs.toSet().toList();

    print('🔍 Scanning ${possibleIPs.length} possible server locations...');
    print('📋 Priority networks: ${networkBases.take(3).join(', ')}...');

    // Test each IP to see if the signaling server is running
    int testedCount = 0;
    for (String ip in possibleIPs) {
      testedCount++;
      if (testedCount % 20 == 0) {
        print('📊 Tested $testedCount/${possibleIPs.length} locations...');
      }

      try {
        final response = await http
            .get(Uri.parse('http://$ip:3000'))
            .timeout(
              const Duration(milliseconds: 1200),
            ); // Fast timeout for scanning

        if (response.statusCode == 200) {
          // Check if it's actually our VoIP server by looking for expected response
          if (response.body.contains('VoIP') ||
              response.body.contains('signaling')) {
            print('✅ Found VoIP server at: http://$ip:3000');
            print('📝 Server response: ${response.body.substring(0, 100)}...');
            return 'http://$ip:3000';
          } else {
            print('📍 Found HTTP server at $ip:3000 but not VoIP server');
          }
        } else if (response.statusCode == 404) {
          // Server is responding but might not have root endpoint - could still be our server
          print('� Found server at: http://$ip:3000 (testing further...)');

          // Try to ping a VoIP-specific endpoint
          try {
            final testResponse = await http
                .get(Uri.parse('http://$ip:3000/socket.io/'))
                .timeout(const Duration(milliseconds: 500));
            if (testResponse.statusCode == 400 ||
                testResponse.body.contains('socket.io')) {
              print('✅ Confirmed VoIP server at: http://$ip:3000');
              return 'http://$ip:3000';
            }
          } catch (e) {
            // Continue searching
          }
        }
      } catch (e) {
        // Server not found on this IP, continue silently
        continue;
      }
    }

    print(
      '⚠️ No VoIP server found after scanning ${possibleIPs.length} locations',
    );
    print('💡 Make sure your signaling server is running on port 3000');

    // Final fallback: try Railway cloud server if local network fails
    const String cloudServerUrl = 'https://your-app-name.up.railway.app';
    try {
      final response = await http
          .get(Uri.parse(cloudServerUrl))
          .timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        print('☁️ Using Railway cloud fallback server: $cloudServerUrl');
        return cloudServerUrl;
      }
    } catch (e) {
      print('☁️ Railway cloud fallback server not accessible: $e');
    }

    print('🔄 Using localhost fallback - server may be on this device');

    // Final fallback to localhost for desktop testing
    return 'http://localhost:3000';
  }

  Future<void> _initializeVoIP() async {
    // Request permissions
    await _requestPermissions();

    // Generate a random numeric user ID for this session
    _currentUserId =
        '${Random().nextInt(9000) + 1000}'; // 4-digit number (1000-9999)

    // Auto-detect server URL
    String serverUrl = await _getServerUrl();
    print('Using server URL: $serverUrl');

    // Initialize VoIP service
    await _voipService.initialize(serverUrl: serverUrl, userId: _currentUserId);

    // Set up callbacks
    _voipService.onIncomingCall = (String fromUser) {
      _showIncomingCallDialog(fromUser);
    };

    _voipService.onCallEnded = () {
      if (mounted) {
        Navigator.of(context).popUntil((route) => route.isFirst);
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Call ended')));
      }
    };

    _voipService.onError = (String error) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $error')));
      }
    };

    setState(() {
      _isInitialized = true;
    });
  }

  Future<void> _requestPermissions() async {
    await [Permission.microphone, Permission.camera].request();
  }

  void _showIncomingCallDialog(String fromUser) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => IncomingCallPage(
          callerName: fromUser,
          onAccept: () {
            _voipService.acceptIncomingCall();
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (context) =>
                    CallingPage(phoneNumber: fromUser, isIncomingCall: true),
              ),
            );
          },
          onReject: () {
            _voipService.rejectIncomingCall();
            Navigator.pop(context);
          },
        ),
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
          builder: (context) =>
              CallingPage(phoneNumber: _phoneNumber, isIncomingCall: false),
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
                  style: const TextStyle(
                    fontSize: 16,
                    color: Colors.blue,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _phoneNumber.isEmpty ? 'Enter ID to call' : _phoneNumber,
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

    // Don't override onCallEnded here - let the main DialerPage handle it

    _voipService.onError = (String error) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Call error: $error')));
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
    // Navigate back to the dialer page
    Navigator.of(context).popUntil((route) => route.isFirst);
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
                        : (widget.isIncomingCall
                              ? 'Incoming call...'
                              : 'Calling...'),
                    style: const TextStyle(color: Colors.white70, fontSize: 16),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.fiber_manual_record,
                        color: Colors.red,
                        size: 12,
                      ),
                      const SizedBox(width: 4),
                      const Text(
                        'Recording • Scam Analysis',
                        style: TextStyle(
                          color: Colors.red,
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
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
                    backgroundColor: _isSpeakerOn
                        ? Colors.blue
                        : Colors.white24,
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

// Incoming Call Page - Full screen call acceptance interface
class IncomingCallPage extends StatefulWidget {
  final String callerName;
  final VoidCallback onAccept;
  final VoidCallback onReject;

  const IncomingCallPage({
    super.key,
    required this.callerName,
    required this.onAccept,
    required this.onReject,
  });

  @override
  State<IncomingCallPage> createState() => _IncomingCallPageState();
}

class _IncomingCallPageState extends State<IncomingCallPage>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat();

    _pulseAnimation = Tween<double>(begin: 0.8, end: 1.2).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black87,
      body: SafeArea(
        child: Column(
          children: [
            // Top section with caller info
            Expanded(
              flex: 3,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text(
                    'Incoming Call',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 18,
                      fontWeight: FontWeight.w300,
                    ),
                  ),
                  const SizedBox(height: 20),

                  // Animated avatar
                  AnimatedBuilder(
                    animation: _pulseAnimation,
                    builder: (context, child) {
                      return Transform.scale(
                        scale: _pulseAnimation.value,
                        child: Container(
                          width: 150,
                          height: 150,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.blue.shade400,
                            boxShadow: [
                              BoxShadow(
                                color: Colors.blue.withOpacity(0.3),
                                blurRadius: 20,
                                spreadRadius: 10,
                              ),
                            ],
                          ),
                          child: const Icon(
                            Icons.person,
                            size: 80,
                            color: Colors.white,
                          ),
                        ),
                      );
                    },
                  ),

                  const SizedBox(height: 30),

                  // Caller name
                  Text(
                    widget.callerName,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 32,
                      fontWeight: FontWeight.w300,
                    ),
                  ),

                  const SizedBox(height: 10),

                  const Text(
                    'VoIP Call',
                    style: TextStyle(color: Colors.white60, fontSize: 16),
                  ),
                ],
              ),
            ),

            // Bottom section with action buttons
            Expanded(
              flex: 1,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  // Reject button
                  GestureDetector(
                    onTap: widget.onReject,
                    child: Container(
                      width: 70,
                      height: 70,
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.red,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black26,
                            blurRadius: 10,
                            offset: Offset(0, 4),
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.call_end,
                        color: Colors.white,
                        size: 32,
                      ),
                    ),
                  ),

                  // Accept button
                  GestureDetector(
                    onTap: widget.onAccept,
                    child: Container(
                      width: 70,
                      height: 70,
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.green,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black26,
                            blurRadius: 10,
                            offset: Offset(0, 4),
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.call,
                        color: Colors.white,
                        size: 32,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 50),
          ],
        ),
      ),
    );
  }
}
