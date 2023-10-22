# Import necessary libraries
from scipy.stats import kurtosis, skew  
from scipy import stats, special
import logging
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize a logger
logger = logging.getLogger(__name__)

# Define a function for feature scaling
def scaling_features(train, test, val, scaling=None):
    try:
        if scaling:
            # It's trying to measure skewness using the undefined 'SkewMeasure' function.
            # You need to define the 'SkewMeasure' function or use an alternative method to measure skewness.
            skewM = SkewMeasure(train)
            for i in skewM.index:
                train[i] = special.boxcox1p(train[i], 0.15)  # Apply the Box-Cox transformation with lambda=0.15
                test[i] = special.boxcox1p(test[i], 0.15)
                val[i] = special.boxcox1p(val[i], 0.15)

            return train, test, val
        else:
            return train, test, val
    except Exception as e:
        # Log any errors that occur during feature scaling
        logger.error(f"Error occurred in Features Scaling: {e}", exc_info=True)
        pass
