# 📌 Sentiment Analysis using DistilBERT

This project is a complete end-to-end Sentiment Analysis pipeline built using Hugging Face Transformers (DistilBERT) and deployed with a Tkinter-based UI app.

The project covers the following stages:

Raw Data Preprocessing – Cleaning and preparing the IMDB dataset.

Model Training – Fine-tuning DistilBERT on the IMDB dataset for binary sentiment classification (positive/negative).

Model Evaluation – Evaluating with classification report and confusion matrix.

Deployment – Building a simple Tkinter desktop app that allows both single and batch predictions.

## 🚀 Features

Raw IMDB dataset preprocessing (custom implementation).

Train and fine-tune DistilBERT on CPU (with resume support).

Evaluate model performance with metrics.

Save and load trained model.

Tkinter-based desktop app for sentiment prediction.

Supports both single text prediction and batch predictions from text files.

## 🛠️ Installation

Install dependencies:

pip install -r requirements.txt

## 📂 Dataset

The project uses the IMDB Dataset (50,000 reviews, balanced positive/negative).

Steps performed in preprocessing:

Lowercasing text.

Removing HTML tags.

Removing special characters & punctuations.

Handling extra whitespaces.

Tokenization & preparation for DistilBERT.

### 👉 Note: All raw data preprocessing has been implemented manually in this project.

## 🎯 Model Training

Run the training script:

python train_binary_sentiment.py


The model will:

Save checkpoints in ./results/

Save final trained model in ./final_model_cpu/

## 📊 Evaluation

After training, the model generates:

Classification Report (Precision, Recall, F1-score, Accuracy).

Confusion Matrix for sentiment distribution.

## 💻 Tkinter App

Run the app with:

python sentiment_app.py


Features of the app:

Enter a review → Get sentiment (Positive/Negative) + confidence score.

Upload a .txt file with multiple reviews → Get batch predictions.

## 📦 Requirements

Main dependencies include:

torch

transformers

scikit-learn

pandas, numpy

tk

## 📝 Results

Example evaluation results:

Classification Report:
               precision    recall  f1-score   support

    negative       0.81      0.71      0.76      2500
    positive       0.74      0.83      0.79      2500

    accuracy                           0.77      5000
   macro avg       0.78      0.77      0.77      5000
weighted avg       0.78      0.77      0.77      5000


Confusion Matrix:
 [[1783  717]
 [ 413 2087]]

## 🤝 Contribution

Contributions are welcome! Feel free to fork this repo, raise issues, or submit pull requests.
