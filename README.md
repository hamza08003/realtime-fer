# Real-Time Facial Emotion Recognition Web Application

A real-time facial emotion recognition web application built with Streamlit and PyTorch. This application uses a deep learning model to detect and classify emotions from webcam feed or uploaded images.

## Features

- Real-time emotion detection from webcam feed
- Image upload and emotion analysis
- Support for multiple emotion classes
- Beautiful and intuitive user interface
- Results visualization and analysis
- Batch processing capabilities

## Prerequisites

- Python 3.10 or higher
- Conda (for environment management)
- Webcam (for real-time emotions detection)
- CUDA-capable GPU (optional, for faster inference)

## Installation

### 1. Install Conda

1. Download Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://www.anaconda.com/download)
2. Run the installer and follow the installation instructions
3. Verify installation by opening a terminal/command prompt and running:
   
   ```bash
   conda --version
   ```

### 2. Clone the Repository

```bash
git clone https://github.com/hamza08003/realtime-fer
cd realtime-fer
```

### 3. Create and Activate Conda Environment

```bash
conda env create -f environment.yml
conda activate realtime_fer
```

### 4. Install Dependencies

If you prefer using pip instead of conda:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you have activated the conda environment:
   
   ```bash
   conda activate realtime_fer
   ```

3. Start the Streamlit application:
   
   ```bash
   streamlit run app.py
   ```

5. The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Real-time Detection**:
   - Click on the "Start Camera" button
   - Allow camera access when prompted
   - The application will detect and display emotions in real-time

2. **Image Upload**:
   - Use the file uploader to select an image
   - The application will process the image and display the detected emotions

3. **Batch Processing**:
   - Upload multiple images using the batch processing feature
   - Download results as a ZIP file

## Model Information

The application uses a pre-trained deep learning model based on the AffectNet dataset, capable of detecting 7 basic emotions:
- Happy
- Sad
- Angry
- Surprised
- Fear
- Disgust
- Neutral

## Troubleshooting

1. **Camera Access Issues**:
   - Ensure your browser has permission to access the camera
   - Check if another application is using the camera

2. **Model Loading Errors**:
   - Verify that the model file exists in the `trained_model` directory
   - Check your internet connection for first-time model download

3. **CUDA/GPU Issues**:
   - The application will automatically fall back to CPU if CUDA is not available
   - For optimal performance, ensure you have the correct CUDA version installed
