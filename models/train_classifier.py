"""
Module to train a message data set for classification.

Package:   Disaster Recovery - Message Classification

Module:    process_data.py

This module takes messaging data from a database file, trains a 
classification model and saves the trained model in a pickle file.

Inputs:
  1: File path to the input database
  2: File path to the trained model
Returns:
  1: Nothing
  
Example: python train_classifier.py DisasterResponse.db
                                    MessageClassfier.pkl
"""
###########################################################
# Import required libraries
###########################################################
import sys
import pickle

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

###########################################################
# Function to load the data from the database
###########################################################
def load_data(database_filepath):
    """
    Load and return features, labels and category names.

    The function reads a database file and splits the data
    into message features, category labels, and category names.

    Arguments:
     1. Path to the database file

     Returns:
     1. X - List of features (=list of text messages)
     2. y - List of labels (=list of how the text messages are classified)
     3. category_names - Names of the message categories
    """
    # Create connection to SQLite database
    engine_connect_string = "sqlite:///" + database_filepath
#    engine = create_engine('sqlite:///data/DisasterResponse.db')
    engine = create_engine(engine_connect_string)

    # Load Message data incl. category labels into df
    df = pd.read_sql("SELECT * FROM DisasterResponseTable", engine)

    # Read category names into a list
    category_names = list(df.columns)[4:]

    # Read messages as feature set X
    X = df.message.values

    # Read categories as labels y
    y = df.iloc[:,4:].values
    
    # Return data
    return X, y, category_names

###########################################################
# Function to tokenize a text message
# Used in the CountVectorizer transformer
###########################################################
def tokenize(text):
    """
    Tokenize a text message.

    The function gets a text message and returns a list of single 
    tokens from that message.

    Arguments:
     1. Text to tokenize

     Returns:
     1. List of clean tokes
    """
    # tokenize text
    tokens = nltk.word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

###########################################################
# Function to build the pipeline model for training.
# 
# The model chosen here is a result of some initial 
# optimization activities incl. GridSearch. In Summary:
# 1. AdaBoost classifier instead of RandomForest or NaiveBayes
# 2. TFIDF use_idf => True
# 3. AdaBoost learning rate => 1.0
# 4. CountVectorizer - max_features => 5000
###########################################################
def build_model():
    """
    Create a classification pipeline / model

    The model chosen here is a result of some initial 
    optimization activities incl. GridSearch. In Summary:
    1. AdaBoost classifier instead of RandomForest or NaiveBayes
    2. TFIDF use_idf => True
    3. AdaBoost learning rate => 1.0
    4. CountVectorizer - max_features => 5000

    Arguments:
     1. None

     Returns:
     1. The pipeline model
    """
    # Create the pipeline model with the appropriate hyperparameters
    pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=5000, 
                                 tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultiOutputClassifier(
                    AdaBoostClassifier(learning_rate=1.0)))
    ])

    return pipeline

###########################################################
# Function to predict and score 
###########################################################
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the trained pipeline / model

    Arguments:
     1. model - the trained pipeline model
     2. X_test - test features
     3. Y_test - the true labels
     4. category_names - list of category names

     Returns:
     None
    """
    # Predict values based on trained pipeline model
    Y_pred = model.predict(X_test)


    # Score the results
    print("Precision, Recall and F1-Score for each category:")
    print("... followed by the confusion matrix for each category")
    for i, cat_name in enumerate(category_names):
        print("Evaluation for Category '{}':".format(cat_name))
        print("=======================================================")
        print(classification_report(Y_test[:, i:i+1], Y_pred[:, i:i+1]))
        print(confusion_matrix(Y_test[:, i:i+1], Y_pred[:, i:i+1]))
        print("\n\n")    


###########################################################
# Function to save the trained model
###########################################################
def save_model(model, model_filepath):
    """
    Save the trained model.

    Arguments:
     1. model - the trained pipeline model
     2. model_filepath - path where the model will be saved

     Returns:
     None
    """
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


###########################################################
# Main function
###########################################################
def main():
    """
    Initiates the major training steps up to saving the model

    Arguments:
    1: File path to the input database
    2: File path to the trained model

    Returns:
    None
    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()