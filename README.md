# Kepler Exoplanet Analysis Dashboard

A FastAPI web application for analyzing Kepler exoplanet data with comprehensive visualizations and machine learning analysis.

## Features

- **Heatmap Analysis**: Disposition provenance vs disposition correlation
- **Count Plot**: Distribution of dispositions by fit type  
- **Donut Chart**: Overall disposition distribution visualization
- **Machine Learning**: Stacking classifier with accuracy metrics and confusion matrix

## Quick Start

### Option 1: Using the Batch File (Windows)
```bash
# Double-click start_server.bat or run in command prompt:
start_server.bat
```

### Option 2: Using the Python Script
```bash
python start_server.py
```

### Option 3: Manual Setup
```bash
# Install dependencies
pip install fastapi uvicorn jinja2 pandas numpy matplotlib seaborn scikit-learn bokeh hvplot

# Start the server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Usage

1. Open your browser and go to: http://127.0.0.1:8000
2. The application will automatically load the existing Kepler dataset
3. Click "Run Complete Analysis" to generate all visualizations
4. Optionally upload your own CSV file with Kepler data format

## Required Data Format

Your CSV should contain these columns for full functionality:
- `koi_disposition`: Planet disposition (CONFIRMED, CANDIDATE, FALSE POSITIVE)
- `koi_disp_prov`: Disposition provenance 
- `koi_fittype`: Fit type information

## Troubleshooting

### Page Not Loading?
1. Make sure all dependencies are installed
2. Check that you're running from the correct directory
3. Verify the dataset file exists in `app/data/current_dataset.csv`
4. Check the terminal for error messages

### Missing Visualizations?
1. Ensure your CSV has the required columns
2. Check that the data contains valid values
3. Look at the browser console for JavaScript errors

### Server Won't Start?
1. Make sure port 8000 is not in use
2. Try running: `pip install --upgrade fastapi uvicorn`
3. Check Python version (requires Python 3.7+)

## File Structure
```
├── app/
│   ├── main.py              # FastAPI application
│   ├── model_loader.py      # Model loading utilities
│   ├── models/
│   │   └── kepler_model.py  # Kepler analysis model
│   ├── templates/
│   │   └── index.html       # Web interface
│   └── data/
│       └── current_dataset.csv  # Kepler dataset
├── start_server.bat         # Windows startup script
├── start_server.py          # Python startup script
└── requirements.txt         # Python dependencies
```

## Development

To modify the analysis:
1. Edit `app/models/kepler_model.py` for new visualizations
2. Update `app/templates/index.html` for UI changes
3. Modify `app/main.py` for API endpoints

## License

This project is for educational and research purposes.