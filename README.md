
# Disaster Response Classification

## Background
This project is part of the Udacity DataScience learning program and complements the Data Engineering block of the program.

In this project, the students are asked to apply their newly acquired skills in Date Engineering and Machine Learning pipelines to analyze disaster data from Figure Eight. The goal is to build a ML model for an API that classifies disaster messages.

In this repository, you'll find a data set containing real messages that were sent during disaster events. In context of this project, a machine learning pipeline has been developed to categorize these events in order to send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker could input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

Below are a few screenshots of the web app.

![Disaster Response Project](https://github.com/praemisit/Message_Classifier/blob/master/DisasterResponseProject_Screenshot.png)

## Installation

The code is mainly based on the standard libraries provided by Anaconda3. The only other library that needs to be installed is plotly. In anaconda you can install this library with "conda install plotly".

## Motivation

When a disaster event occurs, hundreds of messages are being send to the respective agencies. In order to provide the appropriate help as quickly as possible, it is critical to pick messages with the highest priority and to distribute those messages to the appropriate disaster relief agencies. 

This app shall help to send the right messages to the rights persons.

## File Description

The following files belong to this project:

1. "readme.md": File to provide some rough idea about the project and how to run the application.
2. "app" folder: Templates and a python script for running the web app.
3. "data" folder: Data sets and a python script to clean the raw data.
4. "models" folder: The trained ML model as pickel file and a python script which trains the model based on the cleaned data.

## How to interact with the project

The development of the project in completed, closed and not intended for further use in other projects. 

## Instructions 
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/DisasterResponseModel.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements

Special thanks and credits go to Figure Eight for providing the data sets. At the same time I want to express a deep "Thank You" to all the developers and contributors for such amazing libraries like Pandas, Numpy, scikit-learn, and many many more.

