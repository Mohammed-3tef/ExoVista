# 🚀 Cosmic Hunter - AI Exoplanet Detection System

A comprehensive AI-powered system for detecting and analyzing exoplanets using machine learning and a beautiful space-themed interface.

## 🌟 Features

- **AI-Powered Detection**: XGBoost model with 92.7% AUC accuracy
- **Space-Themed UI**: Beautiful starfield background with Orbitron font
- **Manual Analysis**: Input 8 physical features for single exoplanet analysis
- **Batch Processing**: Upload CSV files for bulk analysis
- **Real-time Results**: Interactive dashboards with confidence metrics
- **AI Chatbot**: Educational assistant for exoplanet science
- **Feature Importance**: Understand which features matter most

## 🏗️ Architecture

### Phase 1: Frontend ✅
- Space-themed UI with starfield background
- Manual scan form (8 physical features)
- Batch CSV upload with drag & drop
- Single result dashboard with classification badges
- Batch result dashboard with summary statistics
- Floating AI chatbot interface

### Phase 2: Machine Learning Model ✅
- **Data Processing**: Loaded and merged 3 Kepler datasets
- **Model Training**: XGBoost binary classifier
- **Target**: `koi_disposition` (CONFIRMED vs FALSE POSITIVE/CANDIDATE)
- **Features**: 8 key physical features
- **Performance**: 92.7% AUC, optimized for recall
- **Artifacts**: Saved model, scaler, and feature importance

### Phase 3: Backend API ✅
- **FastAPI Backend**: RESTful API with CORS support
- **Endpoints**:
  - `POST /api/predict/single` - Single exoplanet prediction
  - `POST /api/predict/batch` - Batch CSV processing
  - `POST /api/chat` - AI chatbot (placeholder for Gemini)
  - `GET /api/model/info` - Model information
  - `GET /api/features/descriptions` - Feature descriptions

### Phase 4: Gemini AI Integration ✅
- **Google Gemini API**: Integrated for educational responses
- **Context-aware chatbot**: Uses analysis results for personalized responses
- **Fallback system**: Graceful degradation when API is unavailable
- **Educational focus**: Specialized in exoplanet science and detection methods

### Phase 5: Frontend-Backend Integration ✅
- **Real API integration**: All endpoints connected to backend
- **Enhanced error handling**: User-friendly error messages and validation
- **Loading states**: Visual feedback during processing
- **Success notifications**: Confirmation messages for completed actions
- **Typing indicators**: Real-time chat feedback
- **Form validation**: Client-side validation with helpful error messages

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Google Gemini API key (optional, for enhanced chatbot)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nasa-project-final
   ```

2. **Install Python dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib fastapi uvicorn python-multipart google-generativeai python-dotenv requests
   ```

3. **Set up Gemini AI (Optional)**
   ```bash
   # Copy the environment template
   cp server/env_example.txt server/.env
   
   # Edit server/.env and add your Gemini API key
   # Get your API key from: https://makersuite.google.com/app/apikey
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Easy Startup (Recommended)**
   ```bash
   python start_system.py
   ```
   This will start both servers and open your browser automatically.

5. **Manual Startup (Alternative)**
   ```bash
   # Terminal 1 - Backend
   cd server
   python main.py
   
   # Terminal 2 - Frontend  
   cd client
   python -m http.server 3000 --directory public
   ```

6. **Access the Application**
   - Frontend: `http://localhost:3000`
   - Backend API: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`

## 📊 Model Performance

- **AUC Score**: 92.7%
- **Cross-validation AUC**: 92.17% (±1.12%)
- **Features Used**: 8 physical characteristics
- **Training Data**: 9,201 exoplanet candidates
- **Class Balance**: 29.8% confirmed exoplanets

### Feature Importance
1. **Planetary Radius** (23.4%) - Most important indicator
2. **Insolation Flux** (16.0%) - Stellar energy received
3. **Radius Ratio** (14.8%) - Planet-to-star size ratio
4. **Equilibrium Temperature** (11.6%) - Planet temperature
5. **Transit Depth** (9.8%) - Light dimming during transit
6. **Stellar Temperature** (8.5%) - Host star temperature
7. **Transit Duration** (8.3%) - Length of transit event
8. **Impact Parameter** (7.7%) - Transit geometry

## 🔬 Required Features

For exoplanet analysis, the system requires these 8 features:

| Feature | Description | Range | Units |
|---------|-------------|-------|-------|
| `koi_ror` | Planet-to-star radius ratio | 0.001 - 1.0 | Dimensionless |
| `koi_impact` | Impact parameter | 0.0 - 1.0 | Dimensionless |
| `koi_depth` | Transit depth | 0 - 1000 | Parts per million |
| `koi_prad` | Planetary radius | 0.1 - 50.0 | Earth radii |
| `koi_teq` | Equilibrium temperature | 100 - 5000 | Kelvin |
| `koi_duration` | Transit duration | 0.1 - 50.0 | Hours |
| `koi_insol` | Insolation flux | 0.1 - 10000 | Stellar flux units |
| `koi_steff` | Stellar temperature | 2000 - 10000 | Kelvin |

## 📁 Project Structure

```
nasa-project-final/
├── ml/                          # Machine Learning
│   ├── train_model.py          # Model training script
│   ├── model_utils.py          # Model utilities and inference
│   ├── exoplanet_model.pkl     # Trained XGBoost model
│   ├── scaler.pkl              # Feature scaler
│   ├── feature_importance.csv  # Feature importance rankings
│   └── model_results.png       # Training visualizations
├── server/                      # Backend API
│   ├── main.py                 # FastAPI application
│   └── requirements.txt        # Python dependencies
├── client/                      # Frontend
│   ├── public/
│   │   ├── index.html          # Main HTML file
│   │   ├── styles.css          # Space-themed CSS
│   │   └── app.js              # Frontend JavaScript
│   └── package.json            # Frontend dependencies
├── *.csv                       # Kepler datasets
└── README.md                   # This file
```

## 🎯 Usage Examples

### Single Exoplanet Analysis
1. Navigate to the "Manual Scan" tab
2. Fill in the 8 required features
3. Click "Analyze Exoplanet"
4. View results with classification and confidence

### Batch Analysis
1. Navigate to the "Batch Upload" tab
2. Upload a CSV file with the required columns
3. View summary statistics and individual results

### AI Chatbot
1. Click the floating chat icon (🤖)
2. Ask questions about exoplanets or the analysis
3. Get educational responses about exoplanet science

## 🔧 API Endpoints

### Single Prediction
```bash
curl -X POST "http://localhost:8000/api/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "koi_ror": 0.02,
    "koi_impact": 0.3,
    "koi_depth": 400.0,
    "koi_prad": 1.5,
    "koi_teq": 1200.0,
    "koi_duration": 3.5,
    "koi_insol": 1000.0,
    "koi_steff": 5500.0
  }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/api/predict/batch" \
  -F "file=@exoplanet_data.csv"
```

## 🎨 UI Features

- **Starfield Animation**: Animated background with twinkling stars
- **Orbitron Font**: Futuristic typography for space theme
- **Gradient Effects**: Cosmic color schemes and glowing elements
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Elements**: Hover effects and smooth transitions
- **Loading States**: Visual feedback during processing
- **Error Handling**: User-friendly error messages

## 🧪 Testing

### Run Complete System Test
```bash
python test_complete_system.py
```
This will test all components:
- ML model functionality
- Backend API endpoints
- Frontend accessibility
- Integration between components

### Test Individual Components
```bash
# Test ML model only
cd ml && python model_utils.py

# Test backend only
cd server && python main.py
# Then visit http://localhost:8000/docs

# Test frontend only
cd client && python -m http.server 3000 --directory public
# Then visit http://localhost:3000
```

## 🚧 Development Status

- ✅ **Phase 1**: Frontend UI complete
- ✅ **Phase 2**: ML model trained and saved
- ✅ **Phase 3**: Backend API implemented
- ✅ **Phase 4**: Gemini AI integration complete
- ✅ **Phase 5**: Full integration and testing complete

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NASA Kepler Mission** for the exoplanet datasets
- **Google Fonts** for the Orbitron font
- **FastAPI** for the backend framework
- **XGBoost** for the machine learning model

---

**Cosmic Hunter** - Exploring the universe, one exoplanet at a time! 🌌✨
