# Sentiment Classification using Supervised Learning

## Project Overview 
This project builds a machine learning model to classify text reviews or social media posts into three sentiment categories: **positive**, **negative**, and **neutral**. The model uses supervised learning techniques, specifically Logistic Regression paired with TF-IDF features, to perform accurate sentiment analysis on raw text data. 

## Features 
- Text preprocessing utilizing TF-IDF vectorization.
- Sentiment classification into three distinct classes.
- Comprehensive model training and evaluation, outputting a classification report and confusion matrix.
- Sentiment prediction pipeline for new, unseen text inputs.

## Dataset 
The default sample dataset contains labeled text reviews with their corresponding sentiments. To improve the model's robustness, you can easily swap this out with larger, real-world datasets such as:
- IMDb movie reviews
- Amazon product reviews
- Twitter sentiment datasets

## Installation 

**Prerequisites:**
- Python 3.x

**Dependencies:**
- `pandas`
- `scikit-learn`

Install the required dependencies using `pip`: 

```bash
pip install pandas scikit-learn
```
## Usage
- Load and preprocess the dataset.
- Train the Logistic Regression model using the training data split.
- Evaluate the model's performance on the test set.
- Pass new text samples through the classifier to predict their sentiment.

## Code Overview
The main script performs the following sequential steps:
- **Data Initialization:** Loading and splitting the dataset.
- **Feature Engineering:** Extracting text features using TF-IDF.
- **Training:** Fitting the Logistic Regression model.
- **Evaluation:** Calculating metrics and displaying the results.
- **Inference:** Predicting sentiments on new input texts.

## Future Improvements
- Use larger and more diverse datasets for better generalization.
- Experiment with more advanced algorithms and models (e.g., SVM, LSTM, BERT).
- Integrate advanced text preprocessing pipelines like stemming, lemmatization, and stopword removal.
- Incorporate model interpretability tools.
- Deploy as a simple web app or API for real-time usage.

**Name:** Deepak Kumar
**Reg no.-** 25BAI11461
