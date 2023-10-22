# Import required libraries
import pandas as pd
import numpy as np
import logging
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize a logger
logger = logging.getLogger(__name__)

# Define a data preprocessing function
def data_preprocessing(train, test, val, col1, col2, col3, col4, col5, col6, col7, col8, target):
    try:
        # Cleaning data entries and treating outliers
        
        # Remove rows from train and test where col5 > 95th percentile and target matches col4
        train = train[-((train[col5] > train[col5].quantile(0.95)) & (train[target] == train[col4]))]
        test = test[-((test[col5] > test[col5].quantile(0.95)) & (test[target] == test[col4]))]

        # Filter rows where col6 is less than or equal to 10 in train, test, and val
        train = train[train[col6] <= 10]
        test = test[test[col6] <= 10]
        val = val[val[col6] <= 10]
        
        # Impute missing values
        
        # Replace 0 values in col7 with the mode of 'age' column
        train.loc[train[col7] == 0, col7] = train.age.mode()[0]
        test.loc[test[col7] == 0, col7] = test.age.mode()[0]
        val.loc[val[col7] == 0, col7] = val.age.mode()[0]
    
        # Fill missing values in col4 with the median
        train[col4].fillna(train[col4].median(), inplace=True)
        test[col4].fillna(test[col4].median(), inplace=True)
        val[col4].fillna(val[col4].median(), inplace=True)
        
        # Fill missing values in col8 with 0
        train[col8].fillna(0, inplace=True)
        test[col8].fillna(0, inplace=True)
        val[col8].fillna(0, inplace=True)
        
        return train, test, val
    
    except Exception as e:
        # Log any errors that occur during data preprocessing
        logger.error(f"Error occurred in Data Preprocessing: {e}", exc_info=True)
        pass
