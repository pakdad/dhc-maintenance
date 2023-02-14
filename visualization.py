# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import dhc_maintenance.simulation as simulation
import dhc_maintenance.toolkit as toolkit
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import os
import dhc_maintenance.pipe as pipe
from cProfile import label
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly as plotly
import numpy as np
import matplotlib.pylab as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import types
import seaborn as sns
from datetime import datetime
from scipy import stats
from IPython import display
import dhc_maintenance.maintenance as maintenance
# %%
df = pd.read_csv("data/inventory.csv")
# %%
df = df[['Supply', 'Return']].dropna()
# %%
df.plot()
# %%
