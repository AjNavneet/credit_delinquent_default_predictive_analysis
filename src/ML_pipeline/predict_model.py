import logging
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize a logger
logger = logging.getLogger(__name__)

# Define a function to make predictions using a trained model
def predict_model(model, x_val):
    try:
        # Make predictions using the model
        predictions = model.predict(x_val)
        proba = model.predict_proba(x_val)
        probas = proba[:, 1]
        
        # Add the predicted labels and probability scores to the validation dataset
        x_val['predictions'] = predictions
        x_val['probability_score'] = probas
        
        return x_val
    except Exception as e:
        # Log any errors that occur during prediction
        logger.error(f"Error occurred in Predicting the validation dataset: {e}", exc_info=True)
        pass
