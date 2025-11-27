# Land Cover Segmentation App 🛰️

A Flask web application for satellite/aerial image segmentation using traditional XGBoost machine learning approach.

## Features ✨

- **Drag & Drop Interface**: Easy image upload with visual feedback
- **Ultra-Fast Processing**: Optimized prediction algorithm with parallel processing
- **Interactive Dashboard**: Real-time visualization of land cover percentages
- **6 Land Cover Classes**: Urban, Agriculture, Rangeland, Forest, Water, Barren
- **MongoDB Integration**: Optional database storage for prediction history
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start 🚀

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start MongoDB (Optional)
```bash
# Windows
mongod

# Or use MongoDB Atlas cloud service
```

### 3. Run the Application
```bash
python run_app.py
```

### 4. Access the App
Open your browser and go to: `http://localhost:5000`

## Project Structure 📁

```
Segmantation/
├── app.py                 # Main Flask application
├── run_app.py            # Startup script with checks
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── test_prediction.py    # Test script
├── models/
│   ├── xgboost_model.pkl # Trained XGBoost model
│   └── scaler.pkl        # Feature scaler
├── templates/
│   └── index.html        # Web interface
├── static/               # CSS, JS, images
└── uploads/              # Uploaded images storage
```

## How It Works 🔬

1. **Image Upload**: User drags/drops or selects satellite image
2. **Preprocessing**: Image is divided into 64x64 patches with 32-pixel stride
3. **Feature Extraction**: Color statistics and texture features extracted from each patch
4. **Parallel Processing**: Features extracted using all CPU cores
5. **Prediction**: XGBoost model classifies each patch
6. **Reconstruction**: Patches reassembled into full segmentation map
7. **Post-processing**: Median filter applied for smoothing
8. **Visualization**: Results displayed with statistics and charts

## Supported Formats 📷

- **Input**: PNG, JPG, JPEG (max 16MB)
- **Output**: Segmentation overlay with color-coded classes

## Land Cover Classes 🌍

| Class | Color | Description |
|-------|-------|-------------|
| Urban | White | Buildings, roads, urban areas |
| Agriculture | Yellow | Farmland, crops |
| Rangeland | Magenta | Grassland, pastures |
| Forest | Green | Trees, woodland |
| Water | Blue | Rivers, lakes, oceans |
| Barren | Cyan | Desert, bare soil |

## Performance ⚡

- **Processing Speed**: ~2-10 seconds per image (depending on size)
- **Optimization**: Vectorized operations, parallel processing
- **Memory Efficient**: Streaming processing for large images

## API Endpoints 🔌

- `GET /` - Main web interface
- `POST /upload` - Upload and process image
- `GET /history` - Get prediction history
- `GET /prediction/<id>` - Get specific prediction
- `GET /statistics` - Get global statistics
- `GET /health` - Health check

## Configuration ⚙️

Edit `config.py` to customize:
- Model paths
- Processing parameters (patch size, stride)
- MongoDB connection
- Upload limits

## Troubleshooting 🔧

### Common Issues:

1. **Model files not found**
   - Ensure `xgboost_model.pkl` and `scaler.pkl` are in `models/` folder

2. **MongoDB connection error**
   - App works without MongoDB, but history won't be saved
   - Install and start MongoDB service

3. **Slow processing**
   - Increase `stride` parameter in config for faster processing
   - Reduce image size before upload

4. **Memory errors**
   - Reduce `patch_size` or increase `stride`
   - Close other applications to free RAM

## Development 👨‍💻

### Test the prediction function:
```bash
python test_prediction.py
```

### Run in debug mode:
```bash
python app.py
```

## Next Steps 🎯

- [ ] Add U-Net deep learning model comparison
- [ ] Implement batch processing
- [ ] Add export functionality (GeoTIFF, shapefile)
- [ ] Integrate with mapping libraries
- [ ] Add user authentication
- [ ] Deploy to cloud platforms

## License 📄

This project is for educational purposes as part of a land cover segmentation comparison study.