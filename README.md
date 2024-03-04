# Text-Analysis-of-data-to-predict-sentiment


# Overview:

This code implements sentiment analysis using logistic regression on text data. It starts with preprocessing the data to remove noise, then splits it into train and test sets. Next, it converts comments into vectors using Doc2Vec, trains a logistic regression model from scratch, predicts labels for the test data, evaluates the model's performance, and plots the ROC curve. Finally, it repeats the process using logistic regression from the Scikit-learn library and TF-IDF vectors.

# Explanation:

Importing Libraries: The necessary libraries are imported, including pandas, nltk, sklearn, gensim, matplotlib, and seaborn.

Mounting Google Drive: The Google Drive is mounted to access files stored on Google Drive.

Data Preprocessing: The data is preprocessed by converting comments to lowercase, removing special characters, tokenizing, removing stopwords, and stemming.

Splitting Data: The data is split into training and testing sets using train_test_split.

Converting Comments to Vectors: Comments are converted into vectors using Doc2Vec.

Initializing Weights and Bias: Weights and bias for logistic regression are initialized either as zeros or from a normal distribution.

Logistic Regression from Scratch: Logistic regression is implemented from scratch, including sigmoid function, updating weights, and the logistic regression algorithm itself.

Predictions: The model predicts labels for the test data.

Evaluation: Accuracy, classification report, confusion matrix, F1 score, and ROC curve are calculated to evaluate the model's performance.

Logistic Regression Using Library: Logistic regression is performed using the Scikit-learn library.

TF-IDF Vectorization: Comments are converted into vectors using TF-IDF vectorization.

Evaluation: The performance of the logistic regression model using TF-IDF vectors is evaluated similarly to the previous models.

# Future Aspects:

Deep Learning Models: Exploring deep learning models like LSTM or Transformers for better sentiment analysis.

Advanced Preprocessing: Implementing advanced preprocessing techniques like lemmatization or using pre-trained word embeddings.

Ensemble Methods: Trying ensemble methods like random forests or gradient boosting for improved performance.

Hyperparameter Tuning: Tuning hyperparameters using techniques like grid search or random search to optimize model performance.
