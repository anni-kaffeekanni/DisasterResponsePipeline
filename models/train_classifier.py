# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 11:56:21 2023

@author: AKG
"""

''' resources
    
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
    https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
'''

import sys
import re
import pandas as pd
from sqlalchemy import create_engine, inspect


import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    ''' This function loads the data from the thable 'messages_dataset' of the database
        and splits in two dataframes X and Y containing the messages (X) and the classification (Y)
        
        Args: 
            database_filepath: full filepath to the database
            
        Returns: 
            X: DataFrame with only one column called 'message' containing the messages.
            Y: Binary array with the labels of each message
            category_names: names of the categories in the order of the columns of Y
    '''
    
    engine = create_engine(f'sqlite:///{database_filepath}.db')
    
    df = pd.read_sql_table('messages_dataset', engine)

    X = df['message']
    Y = df[df.columns[4:]]
    category_names=Y.columns

    return(X, Y, category_names)

def tokenize(text):
    ''' This function tokenizes the text content of a message by normalizing, splitting into single words,
        removing stop words and converting to the stef form of the words
        
        Args: 
            text: Message content as string
        
        Returns: 
            stemmed_tokens: a list of the retrieved tokens
    '''
    
    text = re.sub(r'[^\w\s]', '',text.lower())
    
    tokens = word_tokenize(text)
    
    clean_tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    stemmed_tokens = [PorterStemmer().stem(token) for token in clean_tokens]
    
    return stemmed_tokens


def build_model():
    ''' This function sets up a pipeline to build the model, specifies the parameters for Gridsearch parameter optimization
        and returns the model
        
        Args: None
        Returns: cv: GridSearchCV model
    '''

    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),('tfidf',TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    parameters = {'clf__estimator__n_estimators': [10,30],
              'clf__estimator__min_samples_split': [2, 3, 4]
             }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    #cv = pipeline # bypass GridSearchCV model
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''This model predicts the categories for the messages in the test set and evaluates the prediction
       by printing the f1 score, precision and recall for each category
       
       Args: 
           model: the model to evaluate
           X_test: Test messages
           Y_test: Binary array representing the categories of the test messages
           category_name: the name of the category for each column of Y
    '''
    
    Y_pred = model.predict(X_test)
    
    for i in range(Y_pred.shape[1]): 
    
        print(category_names[i])
        
        print(classification_report(Y_test[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    ''' This function saves the model as a pickle
    
        Args: 
            model: trained classifier model
            model_filepath: sting with the filepath for saving the model
    '''
    
    pickle.dump(model, open(f'{model_filepath}.pickle', 'wb'))

def main():
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        
        valid_input_data=True
        
    elif len(sys.argv) == 1:
        
        data_folder = 'C:\\Users\\akg\\Desktop\\Udacity_project\\disaster_response_pipeline_project\\data\\'
        
        database_filepath = data_folder + 'DisasterResponse'
        model_filepath = 'classifier_model'
        
        valid_input_data=True
        
    else: 
        
        valid_input_data = False
        
    if valid_input_data:

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #originaly test_size = 0.2

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()