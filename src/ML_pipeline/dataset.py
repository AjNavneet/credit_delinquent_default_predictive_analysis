# Import required libraries
import pandas as pd
import numpy as np
import logging
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize a logger
logger = logging.getLogger(__name__)

# Define the file paths for the train and validation datasets
train_path = "../input/cs-training.csv"
validation_path = "../input/cs-test.csv"

# Define a function to read data files
def read_data():
    try:
        # Read the train and validation datasets from the specified file paths
        train = pd.read_csv(train_path)

        # Rename the 'Unnamed: 0' column to 'CustomerID' if it exists in the train dataset
        if 'Unnamed: 0' in train.columns.tolist():
            train = train.rename(columns={'Unnamed: 0': 'CustomerID'})

        val = pd.read_csv(validation_path)

        # Rename the 'Unnamed: 0' column to 'CustomerID' if it exists in the validation dataset
        if 'Unnamed: 0' in val.columns.tolist():
            val = val.rename(columns={'Unnamed: 0': 'CustomerID'})

        # Drop the 'SeriousDlqin2yrs' column from the validation dataset
        val.drop(columns=['SeriousDlqin2yrs'], axis=1, inplace=True)

        return train, val

    except Exception as e:
        # Log any errors that occur during data reading
        logger.error(f"Error occurred in reading data files: {e}", exc_info=True)
        pass
