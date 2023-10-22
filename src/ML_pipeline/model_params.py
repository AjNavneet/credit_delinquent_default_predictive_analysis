# Import the LightGBM classifier
from lightgbm import LGBMClassifier

# Define a function for model parameters
def model_params():
    # Create an instance of the LGBMClassifier with specified hyperparameters
    model = LGBMClassifier(
        colsample_bytree=0.65,
        max_depth=4,
        min_data_in_leaf=400,
        min_split_gain=0.25,
        num_leaves=70,
        random_state=42,
        reg_lambda=5,
        subsample=0.65,
        scale_pos_weight=10
    )
    
    return model
