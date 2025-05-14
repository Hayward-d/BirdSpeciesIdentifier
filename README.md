# 🐦 Bird Species Identifier

This is a Flask-based web application that uses a deep learning model to identify bird species from user-uploaded images. It predicts the top 3 most likely bird species out of 200 using a pre-trained model.

---

## 🚀 How to Run

### 1. Install Dependencies

Before running the application, install the required Python packages:

```bash
pip install torch torchvision torchaudio
pip install matplotlib scikit-learn pandas flask flask-cors pillow
```

[Download the AI model with this link](https://drive.google.com/file/d/17mY2A6YePYUDcM1asu5xpU_sfbwHHFYX/view?usp=sharing)
Then put the model into the ../BirdSpeciesIdentifer directory.

### 2. Launch the Application

Navigate to the `Project` directory:

```bash
cd ../BirdSpeciesIdentifier/Project
```

Then run the application:

```bash
python app.py
```

Once it's running, click the local URL printed in the terminal. You’ll be able to upload your own bird photos through the web interface.

The app will display three possible species predictions based on the uploaded image.

### 📷 Sample Images

Sample images from the 200 bird species are provided under the directory:

```
../BirdSpeciesIdentifier/SamplePhotos
```

---

## 📊 Dataset Used

**Caltech-UCSD Birds-200-2011 (CUB-200-2011)**  
🔗 [https://www.vision.caltech.edu/datasets/cub_200_2011/](https://www.vision.caltech.edu/datasets/cub_200_2011/)

---

## 📁 Python File Details

- **`app.py`**  
  Runs the Flask application, handles user image uploads, loads the model, and returns the top 3 predictions to the frontend.

- **`utils.py`**  
  Supports `app.py` by:
  - Loading the trained model
  - Applying image transformations
  - Predicting the top 3 bird species
  - Providing sample images of predicted species

- **`Evaluate.py`**  
  Evaluates model accuracy using Top-1, Top-3, and Top-5 metrics on the test set.  
  To use this, you must download the dataset and place it in:

  ```
  ../BirdSpeciesIdentifier/Project
  ```

- **`Train.py`**  
  Used to train the bird classification model (`bird_model.pth`).  
  Also requires the dataset to be downloaded and added to:

  ```
  ../BirdSpeciesIdentifier/Project
  ```

---
