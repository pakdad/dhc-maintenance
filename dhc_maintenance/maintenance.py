# Predictive Maintenance
#### Import required libraries
# %%
# 1.0 Getting Set-Up
# %%
import copy
import os
# from attr import Attribute
# import plotly.graph_objs as go
# import plotly as plotly
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import types
import pandas as pd
import pickle
import seaborn as sns
from datetime import datetime
# from scipy import stats
from matplotlib.animation import FuncAnimation
# from IPython import display
# from cProfile import label
# import chart_studio.plotly as py
from pandas.api.types import is_numeric_dtype
# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
# %% 

class Maintenance:
    """Maintenance object for predicting pipe failure.""" 

    def __init__(self, inventory, seed = 12, model = None):
        ## Saving inputs
        self.data = inventory
        self.seed = seed
        self.model = model
        ## Other Variables that will (potentially) be defined later
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.val_X = None
        self.val_y = None
        self.split = None
        self.target = None
        self.valid = 0
        self.stdcols =  ['Length', 'Age', 'EFLC', 'Fatigue', 'Temperature Background', 'Temperature Min', 'Temperature Max',
 'Temperature Means', 'Temperature Median', 'PUR Age', 'ΔT 1', 'ΔT 2', 'ΔT 3', 'ΔT 4', 'ΔT 5', 'ΔT 6', 'ΔT 7',
 'ΔT 8', 'ΔT 9', 'ΔT 10', 'ΔT 11', 'ΔT 12', 'ΔT 13', 'ΔT 14', 'ΔT 15', 'ΔT 16', 'ΔT 17', 'ΔT 18', 'ΔT 19', 'ΔT 20',
 'ΔT 21', 'ΔT 22', 'ΔT 23', 'ΔT 24', 'ΔT 25', 'ΔT 26', 'ΔT 27', 'ΔT 28', 'ΔT 29', 'ΔT 30', 'ΔT 31', 'ΔT 32', 'ΔT 33', 'ΔT 34',
 'ΔT 35', 'ΔT 36', 'ΔT 37', 'ΔT 38', 'ΔT 39', 'ΔT 40', 'ΔT 41', 'ΔT 42', 'ΔT 43', 'ΔT 44', 'ΔT 45', 'ΔT 46', 'ΔT 47',
 'ΔT 48', 'ΔT 49', 'ΔT 50', 'ΔT 51', 'ΔT 52', 'ΔT 53', 'Max ΔT', 'Mean Cycle', 'DNcat_0', 'DNcat_1', 'DNcat_2', 'DNcat_3', 'DNcat_4', 'DNcat_9',
 'Supply', 'Retrun', 'not_Aged', 'is_Aged', 'Failure_Target']



    def split_data(self,split, target, valid=0):
        '''Splits data into train, test and validation set(if specified) based on target variable'''
        ## split: Proportion of data in test split
        ## valid: Proportion of data in validation set (always taken from training data)
        ## target: names of target variable column(s) --> if multiple provide as list
        ## Split into Train/test split, validation if wanted
        ## Want to implement rolling origin at some point
        ## NOT TIME SERIES: ADJUST
        self.target = target
        self.split = split
        self.valid = valid
        if split + valid >= 1:
            return "Need some data for training."

        ## Simple approach: Train, then validation, then test
        X = self.data.drop(target, axis = 1)
        y = self.data[target]
        dflen = self.data.shape[0]
        train_stop = int(dflen*(1-split))
        val_stop = int(train_stop + dflen*valid)
        self.train_X = X[0:train_stop]
        self.train_y = y[0:train_stop]
        self.val_X = X[train_stop:val_stop]
        self.val_y = y[train_stop:val_stop]
        self.test_X = X[val_stop:dflen]
        self.test_y = y[val_stop:dflen]
       

    def prepare(self, split=False, drop = 'std', cat_ok = False):
        '''Preprocesses data by dropping columns, replacing outliers and na values, '''
        ## split: Set to true if data is already split and should be updated there
        ## drop: list (or name) of columns to drop from training set
        ## cat_ok: if categorical is ok for model set to true, 
        ## if not, will check datatypes of enum cols when set to false
        ## Scale/Normalize, get rid of NaNs, get rid of outliers, make columns consistent, min/max
        ## Replacing outliers and scaling data
        self.data.replace({-999:0.00})
        self.data.replace({9999:0.00})

        ## Would prefer to impute with avg of the 2 observations around it
        self.data.fillna(0,inplace = True)

        ## Get rid of Categorical variables and scale 
            ## Not time series, scaling isn't essential and code is producing nans with small amt of
            ## samples, work on later if time.
        #scaler = StandardScaler()
        #num_df = self.train_X.select_dtypes(
        #                            include=['int64', 'float64','double', 'float64', 'float32']
        #                                )
        #num_dfS = scaler.fit_transform(num_df)
        #self.data.drop(list(num_df.columns), axis = 1, inplace = True)
        #num_df = pd.DataFrame(num_dfS, columns = list(num_df.columns), dtype = "float32")
        #self.data = pd.concat([num_df,self.data], axis = 1)

        ## Doing this dummying before SMOTE, hopefully okay
        try:
            dummy_list = ['DNCat', 'Flow', 'Aged']
            dummies = pd.get_dummies(self.data['DNCat'], prefix = 'DNCat')
            dummies = pd.concat([dummies, 
                                pd.get_dummies(
                                    self.data['Flow']).rename(columns={0:"Supply", 1:"Return"})],
                                axis = 1) 
            dummies = pd.concat([dummies, 
                                pd.get_dummies(
                                    self.data['Aged']).rename(columns={0:"not_Aged", 1:"is_Aged"})],
                                axis = 1)
            self.data = pd.concat([self.data, dummies], axis = 1)
            self.data.drop(["Aged","Flow", "DNCat"], inplace = True, axis = 1)
        except KeyError:
            print("First categorical step done previously.")

        ## Converting other types from enum str to numeric
        if not cat_ok:
            enum_cols = ['Type', 'Medium Pipe Mode', 'Laying System',
            'Connection', "Life_Status", "Failures", "Status_at_Failure",
            'Failure_Degree', 'Failure_Types', 'Condition']
            try:
                for col in enum_cols:
                    if not is_numeric_dtype(self.data[col]):
                        self.data[col] = self.data[col].apply(lambda x: getattr(x, 'name'))
                        self.data[col].as_type('float32')
            except Exception:
                return "Data format will break models. Most likely need to recreate pipe.dataframe() with argument format = \'value\'"


        ## Checking if data needs to be resplit and any other columns should be dropped
        if isinstance(drop,list):
            try:
                self.data.drop(drop,axis=1, inplace = True)
            except KeyError:
                print("A column couldn't be found when dropping, so nothing was dropped. \
                    check for typos or it doesn't exist.")
        elif drop == 'std':
            for colname, colval in self.data.iteritems():
                if colname not in self.stdcols:
                    self.data.drop([colname], axis= 1 ,inplace = True)
        if split:
            self.split_data(self.split, self.target, self.valid)
        return None



    def SMOTE(self):
        '''Artificially balancing the data by increasing the number of failure events'''
        ## Expand event class if not augmented, what categorical variables(future to change data)
        np.random.seed(self.seed)
        cat_df = self.train_X.select_dtypes(
                                    exclude=['int64', 'float64','double', 'float64', 'float32']
                                        )
        cat_cols = []
        for key in cat_df.columns:
            cat_cols.append(self.train_X.columns.get_loc(key))
        smx = SMOTENC(random_state=self.seed,  categorical_features=cat_cols)
        self.train_X, self.train_y = smx.fit_resample(self.train_X, self.train_y)
        self.train_X = pd.DataFrame(self.train_X)
        self.train_y = pd.DataFrame(self.train_y)
        if isinstance(self.target,list):
            self.train_y.columns = self.target
        else:
            self.train_y.columns = [self.target]
        return None


    def fit(self, model = None):
        '''Fits the model based on AUC metric'''
        ## Fit desired model with data
        np.random.seed(self.seed)
        ## Both None
        if not self.model and not model:
            return "need a model to fit!"
        ## Model is None
        elif model is None:
            model = self.model
        ## self.model is none
        else:
            self.model = model
        print("data look", self.data)
        self.model.fit(self.train_X, self.train_y, eval_metric='auc')


            ## --> also could use validation here or in seperate function to compare models
    def evaluate(self):
        '''Evaluates the model on training set'''
        ## See how model does on test set
            ## Set Threshold somewhere??
        np.random.seed(self.seed)
        if not self.model:
            return "need a model for evaluation!"
        try:
            self.predict(self.test_X)
        except:
            return "Model should be fit. run fit(self, model = self.model)"
        dtrain_predictions = self.model.predict(self.train_X)
        dtrain_predprob = self.model.predict_proba(self.train_X)[:,1]
        print("\nModel Report")
        print(f"Accuracy: {metrics.accuracy_score(self.train_y, dtrain_predictions)}")
        print(f"AUC Score (Balanced): {metrics.roc_auc_score(self.train_y, dtrain_predprob)}")
        d = {'predictions':dtrain_predictions,'probabilities':dtrain_predprob}
        return pd.DataFrame(d)


    ## Creates a feature importance plot
    def feature_importance(self):
        '''Creates a feature importance plot'''
        if not self.model:
            return "need a model for evaluation!"
        try:
            self.predict(self.test_X)
        except:
            return "Model should be fit. run fit(self, model = self.model)"
        feat_imp = pd.Series(self.model.get_booster().get_fscore()).sort_values(ascending=False) 
        feat_imp.plot(kind='bar', title='Feature Importance', color='k', figsize=(12,8)) 
        plt.ylabel('Feature Importance Score')
        plt.tight_layout()
        return feat_imp

    def predict(self, data):
        '''Predicts new values with fitted model'''
        ## Passes new data through model and predicts whether it will be predicted to failure
            ## Set Threshold somewhere??
        if not self.model:
            return "need a model for evaluation!"
        try:
            self.predict(self.test_X)
        except:
            return "Model should be fit. run fit(self, model = self.model)"
        return self.model.predict(data)
