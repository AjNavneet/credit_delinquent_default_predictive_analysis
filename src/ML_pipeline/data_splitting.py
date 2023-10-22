# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize a logger
logger = logging.getLogger(__name__)

# Define a function for splitting the dataset
def training_testing_dataset(data):
    try:
        # Extract features and target variable
        df = data.drop(columns=['SeriousDlqin2yrs'], axis=1)
        y = data['SeriousDlqin2yrs']

        # Split the dataset into training and testing sets
        df_test, df_train, y_test, y_train = train_test_split(df, y, test_size=0.8, random_state=42, stratify=y)
        
        # Combine features and target variables for the training and testing sets
        train = pd.concat([df_train, y_train], axis=1)
        test = pd.concat([df_test, y_test], axis=1)
        
        return df_test, df_train, y_test, y_train, train, test
    
    except Exception as e:
        # Log any errors that occur during the data splitting process
        logger.error(f"Error occurred in splitting data into train and validation: {e}", exc_info=True)
        pass
