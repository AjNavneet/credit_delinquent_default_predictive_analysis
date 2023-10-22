import logging
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize a logger
logger = logging.getLogger(__name__)

# Define a function for training a model
def train_model(classifier, x_train, y_train, x_test, y_test):
    try:
        # Fit the classifier model to the training data
        classifier.fit(x_train, y_train)
        
        return classifier  # Return the trained model
        
    except Exception as e:
        # Log any errors that occur during model training
        logger.error(f"Error occurred in Training the model: {e}", exc_info=True)
        pass
