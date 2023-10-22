from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling
import logging
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize a logger
logger = logging.getLogger(__name__)

# Define a function for upsampling the minority class
def upsampling_class(train, test, val, upsampling=None):
    try:
        if upsampling:
            # Separate features and target variable for training and testing data
            x_train = train.drop(columns=['CustomerID', 'SeriousDlqin2yrs'], axis=1)
            y_train = train['SeriousDlqin2yrs']

            test_x = test.drop(columns=['CustomerID', 'SeriousDlqin2yrs'], axis=1)
            test_y = test['SeriousDlqin2yrs']

            val_x = val.drop(columns=['CustomerID'], axis=1)

            # Create a SMOTE object for oversampling the minority class
            smote = SMOTE(sampling_strategy='minority', k_neighbors=2, random_state=42)
            os_data_X, os_data_y = smote.fit_resample(x_train, y_train)

            return os_data_X, os_data_y, test_x, test_y, val_x
        else:
            # If upsampling is not required, return the original datasets
            x_train = train.drop(columns=['CustomerID', 'SeriousDlqin2yrs'], axis=1)
            y_train = train['SeriousDlqin2yrs']

            test_x = test.drop(columns=['CustomerID', 'SeriousDlqin2yrs'], axis=1)
            test_y = test['SeriousDlqin2yrs']

            val_x = val.drop(columns=['CustomerID'], axis=1)

            return x_train, y_train, test_x, test_y, val_x

    except Exception as e:
        # Log any errors that occur during upsampling
        logger.error(f"Error occurred in Upsampling of the minority class: {e}", exc_info=True)
        pass
