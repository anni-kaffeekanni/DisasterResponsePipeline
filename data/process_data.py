# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 17:20:59 2023

@author: AKG
"""

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect


def load_data(messages_filepath, categories_filepath):
    ''' This function reads in the messages and categories from a csv file
        and returns a combined dataframe where the data form both files are merged using the id column
        
        Args:
        message_filepath:   full filepath of csv file containing the messages
        categories_filepath: full filepath ov csv file containing the categories for each message
        
        Returns:
        df: combined data in one pandas DataFrame
    '''
    
    df_messages = pd.read_csv(messages_filepath)
    
    df_categories = pd.read_csv(categories_filepath, sep=',')
    
    # merge single df to one df
    df = df_messages.merge(df_categories, on='id')
    
    return df


def clean_data(df):
    ''' This finction cleans the dataset containing the messages and the respective categories. 
        The cleaning process includes the following steps:
        1. Drop duplicate rows
        2. Transfere info from Categories column into a form where there is one column for each category
            and a value of 0 or 1 indicating weather the message belongs in this category
         
        Args: 
        df: dataframe as returned by the "load_data()" function
        
        Returns: 
        df: cleaned dataframe
    '''
    
    # 1.drop duplicates
    df.drop_duplicates(inplace=True)
    
    # 2.get seperate columns for the catogeries
    
    categories = df['categories'].str.split(';', expand=True)
    
    categories.columns = categories.iloc[0,:].str.strip('-01')
    
    for column in categories:
        categories[column] = categories[column].str.strip('-'+column)
        categories[column] = categories[column].astype(int)

    print(categories.head())

    # 3. Filter out all rows in the categories df, that have no binary entry
    categories = pd.DataFrame(data=np.where(categories.values >=1,1,0), columns=categories.columns)

    print(categories.head())

    df = pd.concat([df.drop(columns=['categories']),categories], axis=1)
    
    return df


def save_data(df, database_filename):
    ''' This function saves the processed data into a database
        
        Args:
            df: DataFrame to save to database
            database_filename: name of the database
    '''
    
    engine = create_engine(f'sqlite:///{database_filename}.db')
    
    df.to_sql('messages_dataset', engine, index=False)
              
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        valid_input_data=True
        
    elif len(sys.argv) == 1: 
        
        messages_filepath = 'messages.csv'
        categories_filepath = 'categories.csv'
        
        database_filepath = 'DisasterResponse'
        
        valid_input_data = True
        
    else: 
        
        valid_input_data = False
    
    if valid_input_data:

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()