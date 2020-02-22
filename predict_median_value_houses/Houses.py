from common.utils.io_util import load_data
from common.utils.transformations_util import data_to_dataframe
from common.services.logger import get_logger
from stats import get_data_statistics,get_dataframe_statistics
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

log = get_logger()


class Boston_Houses:
    """
    Class to load ,preprocess and run the model
    """
    def __init__(self):
        boston = load_data()
        # get_data_statistics(boston)
        self.boston_df = data_to_dataframe(boston.data)
        self.boston_df.columns = boston.feature_names
        # append the target to the data
        self.boston_df['PRICE'] = boston.target
        # get_dataframe_statistics(boston_df)
        # from the stats, there are no categorical variables and hence the model can be trained directly

    def instantiate_model(self,model_type, num_of_features, learning_rate, tree_depth, l1_regularization, num_of_trees):
        """
        Instantitates the model according to different paramters
        :return:
        """
        self.xg_reg = xgb.XGBRegressor(objective=model_type, colsample_bytree=num_of_features,
                                     learning_rate=learning_rate,
                                  max_depth=tree_depth, alpha=l1_regularization, n_estimators=num_of_trees)

    def train_model(self):
        """
        Train the model
        :return:
        """

        # seperate the featurs and target
        features, target = self.boston_df.iloc[:, :-1], self.boston_df.iloc[:, -1]

        # Convert to Dmatrix
        self.boston_df_dmatrix = xgb.DMatrix(data=features, label=target)
        self.features_train, self.features_test, self.target_train, self.target_test = train_test_split(features,
                                                                                                        target,
                                                                                    test_size=0.2,random_state=123)

        self.xg_reg.fit(self.features_train, self.target_train)


    def predict(self):
        """
        Predict the test set
        :return:
        """
        self.pred_target = self.xg_reg.predict(self.features_test)

    def compute_performance(self):
        """
        Compute the Root mean sqaured error between the actual and predicted to analyze performance
        :return:
        """
        self.rmse = np.sqrt(mean_squared_error(self.target_test,self.pred_target))
        log.info(f'RMSE of the model is : {self.rmse}')

    def cross_validation(self):
        """
        Perform cross validation to prevent overfitting
        :return:
        """
        params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
                  'max_depth': 5, 'alpha': 10}

        self.cv_results = xgb.cv(dtrain=self.boston_df_dmatrix, params=params, nfold=3,
                            num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

        print((self.cv_results["test-rmse-mean"]).tail(1))