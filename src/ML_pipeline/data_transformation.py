# Import required libraries
import pandas as pd
from scipy import stats, special
from scipy.stats import skew
import logging
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize a logger
logger = logging.getLogger(__name__)

# Define a function to measure skewness in numerical features
def SkewMeasure(df):
    nonObjectColList = df.dtypes[df.dtypes != 'object'].index
    skewM = df[nonObjectColList].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewM = pd.DataFrame({'skew': skewM})
    return skewM[abs(skewM) > 0.5].dropna()

# Define a function for scaling features
def scaling_features(train, test, val, scaling=None):
    try:
        if scaling:
            # Measure skewness of features and select those with skewness greater than 0.5
            skewM = SkewMeasure(train)
            
            # Apply Box-Cox transformation with lambda = 0.15 to selected features
            for i in skewM.index:
                train[i] = special.boxcox1p(train[i], 0.15)
                test[i] = special.boxcox1p(test[i], 0.15)
                val[i] = special.boxcox1p(val[i], 0.15)

            return train, test, val
        else:
            return train, test, val
    except Exception as e:
        # Log any errors that occur during feature scaling
        logger.error(f"Error occurred in Features Scaling: {e}", exc_info=True)
        pass
