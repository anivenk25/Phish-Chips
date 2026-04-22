const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    // Use Railway volume mount path if available, otherwise local folder
    const uploadDir = process.env.RAILWAY_VOLUME_MOUNT_PATH || './recordings';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `recording_${Date.now()}_${file.originalname}`);
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit for Railway
  }
});

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// Recording upload endpoint for scam analysis
app.post('/upload-recording', upload.single('recording'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No recording file provided' });
    }
    
    const { timestamp, type, analysis_priority, chunk_interval, call_duration } = req.body;
    
    // Log with enhanced metadata for scam analysis
    console.log(`📞 SCAM ANALYSIS UPLOAD:`);
    console.log(`   File: ${req.file.filename}`);
    console.log(`   Size: ${req.file.size} bytes`);
    console.log(`   Timestamp: ${timestamp}`);
    console.log(`   Type: ${type}`);
    console.log(`   Priority: ${analysis_priority}`);
    console.log(`   Interval: ${chunk_interval}`);
    console.log(`   Duration: ${call_duration}`);
    console.log(`   ⚡ Ready for real-time fraud detection`);
    
    // Notify scam detection system of new file
    notifyScamDetectionSystem(req.file.filename, req.file.path, {
      timestamp,
      type,
      analysis_priority,
      chunk_interval,
      call_duration,
      size: req.file.size
    });
    
    res.status(200).json({ 
      message: 'Recording uploaded for scam analysis',
      filename: req.file.filename,
      size: req.file.size,
      analysis_status: 'queued',
      priority: analysis_priority
    });
  } catch (error) {
    console.error('Error uploading recording for analysis:', error);
    res.status(500).json({ error: 'Failed to upload recording for analysis' });
  }
});

// Function to notify scam detection system
async function notifyScamDetectionSystem(filename, filepath, metadata) {
  try {
    // Check if scam detection API is available
    const axios = require('axios').default;
    const scamApiUrl = process.env.SCAM_API_URL || 'http://localhost:8000';
    
    // Notify the scam detection system
    await axios.post(`${scamApiUrl}/force-analyze/${filename}`, {
      metadata: metadata
    });
    
    console.log(`🔔 Notified scam detection system: ${filename}`);
  } catch (error) {
    // Silently fail if scam detection system is not available
    // File monitoring will pick it up anyway
    console.log(`📝 Scam detection notification failed (will be picked up by file monitor): ${error.message}`);
  }
}

// Health check endpoint
app.get('/', (req, res) => {
  res.json({ 
    status: 'VoIP Signaling Server with Scam Analysis', 
    timestamp: new Date().toISOString(),
    features: ['real_time_recording', 'scam_detection', '10_second_chunks']
  });
});

// Scam analysis status endpoint
app.get('/analysis-stats', (req, res) => {
  const recordingsDir = process.env.RAILWAY_VOLUME_MOUNT_PATH || './recordings';
  
  try {
    const files = fs.readdirSync(recordingsDir);
    const totalRecordings = files.length;
    const recentFiles = files.filter(file => {
      const filePath = `${recordingsDir}/${file}`;
      const stats = fs.statSync(filePath);
      const fileAge = Date.now() - stats.mtime.getTime();
      return fileAge < (24 * 60 * 60 * 1000); // Last 24 hours
    });
    
    res.json({
      status: 'Scam Analysis Active',
      total_recordings: totalRecordings,
      recent_recordings: recentFiles.length,
      upload_frequency: '10_seconds',
      analysis_ready: true,
      storage_path: recordingsDir,
      storage_type: process.env.RAILWAY_VOLUME_MOUNT_PATH ? 'Railway Volume (Persistent)' : 'Local Directory (Ephemeral)',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.json({
      status: 'Analysis Ready',
      total_recordings: 0,
      upload_frequency: '10_seconds',
      storage_path: recordingsDir,
      storage_type: process.env.RAILWAY_VOLUME_MOUNT_PATH ? 'Railway Volume (Persistent)' : 'Local Directory (Ephemeral)',
      timestamp: new Date().toISOString()
    });
  }
});

// Storage information endpoint
app.get('/storage-info', (req, res) => {
  const recordingsDir = process.env.RAILWAY_VOLUME_MOUNT_PATH || './recordings';
  const isVolumeMount = !!process.env.RAILWAY_VOLUME_MOUNT_PATH;
  
  try {
    const files = fs.readdirSync(recordingsDir);
    let totalSize = 0;
    const fileDetails = files.map(file => {
      const filePath = `${recordingsDir}/${file}`;
      const stats = fs.statSync(filePath);
      totalSize += stats.size;
      return {
        name: file,
        size: `${(stats.size / 1024 / 1024).toFixed(2)} MB`,
        created: stats.birthtime.toISOString(),
        modified: stats.mtime.toISOString()
      };
    });
    
    res.json({
      storage_path: recordingsDir,
      storage_type: isVolumeMount ? 'Railway Volume (Persistent)' : 'Local Directory (Ephemeral)',
      persistent: isVolumeMount,
      total_files: files.length,
      total_size: `${(totalSize / 1024 / 1024).toFixed(2)} MB`,
      warning: !isVolumeMount ? 'Files will be lost on app restart/redeploy' : null,
      recommendation: !isVolumeMount ? 'Add Railway Volume for persistent storage' : 'Storage is persistent',
      files: fileDetails.slice(0, 10), // Show last 10 files
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.json({
      storage_path: recordingsDir,
      storage_type: isVolumeMount ? 'Railway Volume (Persistent)' : 'Local Directory (Ephemeral)',
      persistent: isVolumeMount,
      error: 'Unable to read storage directory',
      warning: !isVolumeMount ? 'Files will be lost on app restart/redeploy' : null,
      recommendation: !isVolumeMount ? 'Add Railway Volume for persistent storage' : 'Storage is persistent',
      timestamp: new Date().toISOString()
    });
  }
});

// Store connected users
const connectedUsers = new Map();

io.on('connection', (socket) => {
  console.log('User connected:', socket.id);
  
  // Register user
  const userId = socket.handshake.query.userId;
  if (userId) {
    connectedUsers.set(userId, socket.id);
    console.log(`User ${userId} registered with socket ${socket.id}`);
  }

  // Handle call request
  socket.on('call-request', (data) => {
    const targetUserId = data.target;
    const targetSocketId = connectedUsers.get(targetUserId);
    
    if (targetSocketId) {
      io.to(targetSocketId).emit('call-request', {
        from: userId,
        offer: data.offer
      });
      console.log(`Call request from ${userId} to ${targetUserId}`);
    } else {
      socket.emit('user-not-found', { targetUserId });
    }
  });

  // Handle answer
  socket.on('answer', (data) => {
    // Find who called this user
    socket.broadcast.emit('answer', {
      answer: data.answer
    });
  });

  // Handle ICE candidates
  socket.on('ice-candidate', (data) => {
    socket.broadcast.emit('ice-candidate', {
      candidate: data.candidate
    });
  });

  // Handle call rejection
  socket.on('reject-call', () => {
    socket.broadcast.emit('call-rejected');
  });

  // Handle call end
  socket.on('end-call', () => {
    socket.broadcast.emit('call-ended');
  });

  // Handle disconnect
  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
    // Remove from connected users
    for (const [userId, socketId] of connectedUsers.entries()) {
      if (socketId === socket.id) {
        connectedUsers.delete(userId);
        break;
      }
    }
  });
});

// REST endpoint to get connected users
app.get('/users', (req, res) => {
  const users = Array.from(connectedUsers.keys());
  res.json({ users });
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'OK', connectedUsers: connectedUsers.size });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 VoIP signaling server running on port ${PORT}`);
  console.log(`🌐 Environment: ${process.env.NODE_ENV || 'development'}`);
  if (process.env.PORT) {
    console.log(`☁️ Heroku deployment detected`);
  } else {
    console.log(`🏠 Local development mode`);
    console.log(`WebSocket endpoint: ws://localhost:${PORT}`);
  }
});
