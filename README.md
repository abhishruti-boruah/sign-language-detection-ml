# sign-language-detection-ml
# Sign Language Detection using Machine Learning

This project was developed as my **final year diploma project**. It detects hand signs using computer vision and machine learning techniques.

The system uses **MediaPipe** to detect hand landmarks and a **Random Forest classifier** to classify sign language gestures. The model was trained using **100+ collected datasets**.

## Technologies Used

* Python
* OpenCV
* MediaPipe
* Scikit-learn

## Features

* Real-time hand tracking
* Hand landmark detection
* Machine learning based gesture classification
* Webcam-based sign detection

## Project Workflow

1. Collect hand landmark data using a webcam
2. Train a machine learning model using the collected dataset
3. Predict sign language gestures in real time

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Collect training data:

```bash
python collect_data.py
```

Train the model:

```bash
python train_model.py
```

Run the prediction program:

```bash
python predict.py
```

## Author

**Abhishruti Jyoti Pranjana Boruah**
B.Tech Computer Science and Engineering
Tezpur University, Assam
