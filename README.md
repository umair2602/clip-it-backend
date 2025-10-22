# Clip-It Backend API

A production-ready FastAPI backend for AI-powered video clipping with user authentication, S3 integration, and single-table database architecture.

## 🚀 Features

- **User Authentication**: JWT-based authentication system
- **Video Upload**: Direct file upload with progress tracking
- **YouTube Integration**: Download and process YouTube videos
- **S3 Storage**: Cloud storage for videos and clips
- **AI Processing**: Automatic transcription, content analysis, and clip generation
- **Single-Table Architecture**: Efficient MongoDB design with embedded arrays
- **Real-time Status**: Background task processing with status updates

## 🏗️ Architecture

### Database Structure
- **Users Collection**: Contains user data with embedded videos array


### API Endpoints

#### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Token refresh

#### Video Management
- `POST /upload` - Direct file upload
- `POST /youtube-download` - YouTube video download
- `GET /user/history/videos` - User video history
- `GET /user/history/clips` - User clip history
- `GET /status/{task_id}` - Processing status

#### S3 Integration
- `POST /s3-upload-url` - Generate presigned upload URLs
- `POST /register-s3-upload` - Register uploaded files

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- MongoDB
- AWS S3 (for file storage)
- FFmpeg (for video processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd clip-it-backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_clean.txt
   ```

3. **Environment Configuration**
   ```bash
   cp env_example.txt .env
   # Edit .env with your configuration
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## 📁 Project Structure

```
├── app.py                 # Main FastAPI application
├── config.py             # Configuration settings
├── requirements_clean.txt # Python dependencies
├── .gitignore           # Git ignore patterns
├── services/            # Business logic services
│   ├── auth.py         # Authentication service
│   ├── video_service.py # Video processing service
│   └── user_video_service.py # Single-table operations
├── routes/              # API route definitions
│   ├── auth.py         # Authentication routes
│   └── video.py        # Video management routes
├── models/              # Data models
│   ├── user.py         # User and video models
│   └── video.py        # Legacy video models
├── utils/               # Utility functions
│   ├── s3_storage.py   # S3 integration
│   ├── s3_validator.py # URL validation
│   └── youtube_downloader.py # YouTube download
├── database/            # Database connection
│   └── connection.py   # MongoDB connection
├── static/              # Static assets
├── uploads/             # Temporary upload directory
└── outputs/             # Processed video outputs
```

## 🔧 Configuration

### Environment Variables

```env
# Database
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=clipit_db

# JWT Authentication
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AWS S3
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

## 🚀 Deployment

### Railway Deployment
```bash
# Railway will automatically detect and deploy
railway up
```

### Docker Deployment
```bash
docker build -t clipit-backend .
docker run -p 8000:8000 clipit-backend
```

## 📊 API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔒 Security Features

- JWT-based authentication
- CORS protection
- Input validation
- S3 URL validation
- Error handling and logging

## 🎯 Production Ready

- ✅ Clean codebase (removed all debug files)
- ✅ Single-table database architecture
- ✅ Comprehensive error handling
- ✅ Authentication integration
- ✅ S3 integration
- ✅ Background task processing
- ✅ Real-time status updates
- ✅ Production deployment ready

## 📝 License

This project is licensed under the MIT License.

---

**Status**: Production Ready ✅  
**Last Updated**: August 2025 