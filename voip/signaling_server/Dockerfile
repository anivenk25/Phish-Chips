FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install --production

# Copy app code
COPY . .

# Ensure recordings directory exists
RUN mkdir -p /recordings

# Environment variables
ENV RAILWAY_VOLUME_MOUNT_PATH=/recordings
ENV SCAM_API_URL=http://backend:8000

EXPOSE 3000

# Start the signaling server
CMD ["node", "server.js"]