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
