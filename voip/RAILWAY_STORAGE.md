# Railway Volume Setup for Persistent Recording Storage

## Why Railway Volumes?
- ✅ Persistent storage that survives deployments
- ✅ Easy to set up in Railway dashboard
- ✅ Cost-effective for small files
- ✅ No code changes needed

## Setup Steps:

### 1. Add Volume in Railway Dashboard
1. Go to your Railway project dashboard
2. Click on your service
3. Go to "Variables" tab
4. Add volume:
   - **Mount Path**: `/app/recordings`
   - **Size**: 1GB (or as needed)

### 2. Update Server Path (Optional)
If you want to be explicit about the volume path, update server.js:

```javascript
const uploadDir = process.env.RAILWAY_VOLUME_MOUNT_PATH || './recordings';
```

### 3. Environment Variable
Railway will automatically set the volume mount path, but you can also set:
- **RECORDINGS_PATH**: `/app/recordings`

## Benefits:
- 📁 Recordings persist between deployments
- 🔄 Automatic backup by Railway
- 📈 Scalable storage
- 💰 Pay only for what you use

## Cost:
- Railway volumes: ~$0.25/GB/month
- 1GB typically handles hundreds of call recordings

## Alternative: Cloud Storage (Advanced)
For production apps, consider:
- AWS S3
- Google Cloud Storage  
- Cloudinary
- Railway PostgreSQL with file references
