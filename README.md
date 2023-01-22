# Disaster Response Pipeline Project
## Table of Contents
1. Introduction
2. File Descriptions
3. Installation
4. Instructions
5. Acknowledgements

## Introduction
This project is a project included in the Udacity Data Scientist Nanodegree.
The Project is a collaboration with Figure Eight. 

In the project, pre labeled messages that correspond to disasters are used to train
a model that labels future messages according to the labels provided in the training dataset. 

The project finishes with a web application, where the user can run the model on any message the want and
see the classification result.

## File description

The project consists of three main steps: data preparation, model training and optimisation
and the web app design. The files with the python code as well as the data files used during this project are organized in three folders

### Folder: data
process_data.py -> This file contains the python code to read in the data from the csv files, clean it and store it in a databank
categories.csv -> csv file with the categories connected with each message
messages.csv -> csv file containing the messages 
DisasterResponse.db -> sqlite database as saved by running process_data.py 

#### Folder: models
train_classifier.py -> Python code to read in the data from the database in the data folder and train a classifier model on it

#### Folder: app
subfolder templates -> templates for the web app
run.py -> python code to run the web ap with flask

### Installation
The code was written on Python 3.8.

The code can be found in this Github Repo https://github.com/anni-kaffeekanni/DisasterResponsePipeline

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Open http://localhost:3000 in a browser

### Acknowlegements

Acknowlegements to Udacity for providing the project support and to Figure Eight for providing the data set
