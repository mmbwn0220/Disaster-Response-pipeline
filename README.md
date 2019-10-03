# Disaster-Response-pipeline
Disaster Response pipeline project

# Project Overview

This project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
In the Project data folder, you'll find a data set containing real messages files that were sent during disaster events: data/disaster_messages.csv and data/disaster_categories.csv . A ETL pipeline ( data/process_data.py)  will clean the data and will save to a database ( data/DisasterResponse.db). A machine learning pipeline( models/train_classifier.py) created to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

This project also include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
