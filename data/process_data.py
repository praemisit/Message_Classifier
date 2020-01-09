"""
Module to extract an provide data for message classification.

Package:   Disaster Recovery - Message Classification

Module:    process_data.py

This module loads and cleans message data for further processing.

Inputs:
  1: File path of the message dataset
  2: File path of the category dataset
  3: File path of the database to save the cleaned data 
Returns:
  1: Nothing
  
Example: python process_data.py disaster_messages.csv 
                                disaster_categories.csv
                                DisasterResponse.db
"""

###########################################################
# Import required libraries
###########################################################
import sys
import pandas as pd
from sqlalchemy import create_engine


###########################################################
# Function to load the data from file(s)
###########################################################
def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories into a df.

    The function reads a data set with messages and a data
    set with categories. Once the data is loaded, the function
    merges and returns the two data sets.

    Arguments:
     1. Path to the messages file
     2. Path to the categories file

     Returns:
     1. Pandas Data Frame with the merged data
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Drop duplicated rows in each of the two data sets
    messages = messages.drop_duplicates()
    categories = categories.drop_duplicates()
    
    # Merge datasets
    df = messages.merge(categories, on=("id"), how="inner")

    # return dataset
    return df

###########################################################
# Function to clean-up the data for further processing
###########################################################
def clean_data(df):
    """
    Clean the data set to make it suitable for further analysis.

    The function split the category column into 36 individual columns
    and removes any duplicated rows.

    Arguments:
     1. Data frame which shall be converted / cleaned

     Returns:
     1. Pandas Data Frame with the cleaned data
    """
    ####
    # Create a data frame of the 36 individual category columns
    # by splitting the categories column at the separator ";"
    categories = df["categories"].str.split(pat=";", expand=True)

    ####
    # To determine the appropriate column names, we just take the 
    # first row and take up to second last character from each value.

    # First row:
    row = categories.iloc[0]

    # Get values up to the second last character of each value in the row
    category_colnames = [(lambda x: x[:-2])(x) for x in row]
    
    # Now, rename the columns
    categories.columns = category_colnames

    ####
    # Convert values in all columns of categories to just contain 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors="coerce")

    # Drop the original categories column from `df`
    df.drop(["categories"], axis=1, inplace=True)

    # Some values in the related column do have a 2 instead of a 1. 
    # We adjust this here
    categories["related"] = categories["related"].replace(2, 1)
    
    # Concatenate the original data frame with the new `categories` data frame
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # return dataset
    return df

###########################################################
# Function to save the data in an SQLite database
###########################################################
def save_data(df, database_filename):
    """
    Save the data in an SQLite database.

    Arguments:
     1. Data frame which shall be saved
     2. Database filename

     Returns:
     None
    """
    db_connect_string = "sqlite:///" + database_filename

    engine = create_engine(db_connect_string)
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists="replace")  


def main():
    """
    Main Script which reads input parameters and 
    runs through the data load, data clean and data save
    steps.

    Arguments:
    1: File path of the message dataset
    2: File path of the category dataset
    3: File path of the database to save the cleaned data 

     Returns:
     None
    """    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the file paths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the file path of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()