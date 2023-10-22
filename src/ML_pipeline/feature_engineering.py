# Import required libraries
import numpy as np
import logging
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize a logger
logger = logging.getLogger(__name__)

# Define a function for feature engineering
def feature_engineering(train, test, val, col1, col2, col3, col4, col5, col6, col7, col8, col9):
    try:
        # Create and concatenate additional features
        
        # CombinedPastDue: Sum of col1, col2, and col3
        train['CombinedPastDue'] = train[col1] + train[col2] + train[col3]
        test['CombinedPastDue'] = test[col1] + test[col2] + test[col3]
        val['CombinedPastDue'] = val[col1] + val[col2] + val[col3]

        # CombinedCreditLoans: Sum of col4 and col5
        train['CombinedCreditLoans'] = train[col4] + train[col5]
        test['CombinedCreditLoans'] = test[col4] + test[col5]
        val['CombinedCreditLoans'] = val[col4] + val[col5]

        # MonthlyIncomePerPerson: col6 divided by (col7 + 1)
        train['MonthlyIncomePerPerson'] = train[col6] / (train[col7] + 1)
        test['MonthlyIncomePerPerson'] = test[col6] / (test[col7] + 1)
        val['MonthlyIncomePerPerson'] = val[col6] / (val[col7] + 1)

        # MonthlyDebt: col6 multiplied by col8
        train['MonthlyDebt'] = train[col6] * train[col8]
        test['MonthlyDebt'] = test[col6] * test[col8]
        val['MonthlyDebt'] = val[col6] * val[col8]

        # isRetired: 1 if col9 is greater than 65, else 0
        train['isRetired'] = np.where((train[col9] > 65), 1, 0)
        test['isRetired'] = np.where((test[col9] > 65), 1, 0)
        val['isRetired'] = np where((val[col9] > 65), 1, 0)

        # RevolvingLines: Difference between col4 and col5
        train['RevolvingLines'] = train[col4] - train[col5]
        train['hasRevolvingLines'] = np.where((train['RevolvingLines'] > 0), 1, 0)
        train.drop(columns=['RevolvingLines'], axis=1, inplace=True)
        test['RevolvingLines'] = test[col4] - test[col5]
        test['hasRevolvingLines'] = np.where((test['RevolvingLines'] > 0), 1, 0)
        test.drop(columns=['RevolvingLines'], axis=1, inplace=True)
        val['RevolvingLines'] = val[col4] - val[col5]
        val['hasRevolvingLines'] = np.where((val['RevolvingLines'] > 0), 1, 0)
        val.drop(columns=['RevolvingLines'], axis=1, inplace=True)

        # hasMultipleRealEstates: 1 if col5 is greater than or equal to 2, else 0
        train['hasMultipleRealEstates'] = np.where((train[col5] >= 2), 1, 0)
        test['hasMultipleRealEstates'] = np.where((test[col5] >= 2), 1, 0)
        val['hasMultipleRealEstates'] = np.where((val[col5] >= 2), 1, 0)

        # IsAlone: 1 if col7 is equal to 0, else 0
        train['IsAlone'] = np.where((train[col7] == 0), 1, 0)
        test['IsAlone'] = np.where((test[col7] == 0), 1, 0)
        val['IsAlone'] = np.where((val[col7] == 0), 1, 0)

        return train, test, val
    
    except Exception as e:
        # Log any errors that occur during feature engineering
        logger.error(f"Error occurred in Feature Engineering: {e}", exc_info=True)
        pass
