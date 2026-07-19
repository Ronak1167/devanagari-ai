# Devanagari AI - Handwritten Character Recognition

A beautiful, interactive web application that allows users to draw Devanagari characters (Hindi alphabets and numerals) on an HTML5 canvas and predicts the character in real-time using a deep Convolutional Neural Network (CNN).

## Features
* **Real-time Prediction**: Draws instant predictions with top-3 confidence scores as soon as you hit the predict button.
* **Smart Image Preprocessing**: Employs advanced bilinear downscaling and binarization. This allows users to draw with elegant, thin strokes on the frontend while the backend automatically reconstructs the thick strokes expected by the CNN.
* **Modern UI**: A sleek, dark-mode glassmorphism interface.
* **High Accuracy**: Powered by a custom PyTorch CNN that achieves **~98.85% validation accuracy** on the dataset.

## Tech Stack
* **Frontend**: HTML5, Vanilla JavaScript, CSS (Custom Design System)
* **Backend**: Python, Flask
* **Machine Learning**: PyTorch, Torchvision, Pillow (PIL), NumPy

## The Dataset
This project uses the **UCI Devanagari Handwritten Character Dataset**, which consists of 92,000 images (78,200 training and 13,800 testing) spanning 46 classes (36 consonants and 10 numerals). 

## Project Structure
* `app.py`: The Flask web server. Handles routing and complex image preprocessing (cropping, padding, scaling, thresholding).
* `models/`: Directory containing the saved model weights (`devanagari_cnn_model.pth`) and mapping file (`cnn_class_names.json`).
* `scripts/`: Directory containing utility scripts (`train_cnn.py` and `download_data.py`).
* `requirements.txt`: Python package dependencies.
* `static/` & `templates/`: Frontend web assets.
* `data/`: Extracted dataset directory.

---

## How to Run on a New System

### 1. Prerequisites
Make sure you have [Python 3.8+](https://www.python.org/) installed on your system.

### 2. Setup the Environment
It is highly recommended to use a virtual environment. Open your terminal or command prompt and run:

```bash
# Navigate to the project directory
cd devanagari_project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Flask development server:

```bash
python app.py
```

### 5. Open the Web App
Open your favorite web browser and navigate to:
`http://127.0.0.1:5000`

---

## (Optional) Training the Model Yourself
If you want to train the model from scratch on your own machine:

1. **Download the Dataset:**
   ```bash
   python scripts/download_data.py
   ```
2. **Run the Training Script:**
   ```bash
   python scripts/train_cnn.py
   ```
   *This will take some time depending on your CPU/GPU. It will automatically save the best performing model into the `models/` directory.*
