"""
Walking through PdM

Research Project Instandhaltung-FW
Project Number 03ET1625B
__________________
Pakdad Pourbozorgi Langroudi, M.Sc.
wissenschaftlicher Mitarbeiter / Research Associate

HafenCity Universität Hamburg (HCU)
Henning-Voscherau-Platz 1, 20457 Hamburg, Raum 5.008

pakdad.langroudi@hcu-hamburg.de
www.hcu-hamburg.de
+49 (0)40 42827-5332

Will Hoffmann
Research Assistant

Northwestern University
633 Clark St, Evanston, IL 60208, USA

willhoffmann2024@u.northwestern.edu
www.northwestern.edu
+1 563-543-2813
"""
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
## Backward simulation
network = simulation.Utility()

# %%
# check if the folder data is there, otherwise create it
network.weather.search_station_ID('hamburg')
# network.city_to_sim("hamburg")
# %%
network.weather.load_weather("01975")
# %%
df_artificial = network.weather.data.sort_index()
# %%
network_temp_artificial = df_artificial.loc['2018-01-01 00:00:00':'2021-12-31 23:00:00']
# %%
rng = np.random.default_rng(seed=5)
# %%
# y = mx + b
# b = 110
# m = (80-110)/15
# %%
def gliding_sup_temp(temp):
    # y = mx + b
    b = 110
    m = (80-110)/15
    if temp >= 15:
        return 80
    elif temp < 15 and temp > -10:
        y = m*temp + b
        return round(y,2)
    else:
        return 130
# %%
# y = mx + b
# b = 60
# m = (55-60)/15
# %%
def gliding_ret_temp(temp):
    # y = mx + b
    b = 60
    m = (55-60)/15
    if temp >= 15:
        return 55
    elif temp < 15 and temp > -10:
        y = m*temp + b
        return round(y,2)
    else:
        return 65
# %%
network_temp_artificial['Supply'] = network_temp_artificial['Out_Temp'].apply(gliding_sup_temp)
network_temp_artificial['Return'] = network_temp_artificial['Out_Temp'].apply(gliding_ret_temp)

# %%
network_temp_artificial = network_temp_artificial[['Supply', 'Return']]
network_temp_artificial
# %%
np.random.seed(5)
rand_sup = np.round(np.random.uniform(-5,5,(network_temp_artificial.shape[0])),2)
rand_ret = np.round(np.random.uniform(-2.5,2.5,(network_temp_artificial.shape[0])))
# %%
network_temp_artificial['Supply'] += rand_sup
network_temp_artificial['Return'] += rand_ret
# %%
network_temp_artificial
# %%
fig, ax = plt.subplots(figsize=(7,3), dpi=300)
ax.plot(network_temp_artificial['Supply'], color='r', label='Supply', linewidth=.1)
ax.plot(network_temp_artificial['Return'], color='steelblue', label='Return', linewidth=.1)
ax.grid(linestyle='--')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
fig.tight_layout()
fig.savefig('fig/Artificial.png', dpi='figure', format='png',)
# %%
fig, ax = plt.subplots(figsize=(7,3), dpi=300)
ax.plot(network_temp_artificial['Supply'].loc['2020-01-01 00:00:00':'2020-12-31 23:00:00'], color='r', label='Supply', linewidth=.5)
ax.plot(network_temp_artificial['Return'].loc['2020-01-01 00:00:00':'2020-12-31 23:00:00'], color='steelblue', label='Return', linewidth=.5)
ax.grid(linestyle='--')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
fig.tight_layout()
fig.savefig('fig/Artificial1.png', dpi='figure', format='png',)
# %%
with open('data/temperature_dummy.csv','w') as csv:
    network_temp_artificial.to_csv(csv)
# %%
tools = toolkit.Tools()
# %%
tools.statistical_analyser(network.temperature_timeserie)
# %%
years = tools.extract_years(network.temperature_timeserie)
# %%
network.weather.data
# %%
network.weather.data.index.is_unique
# %%
network.temperature_changes.index.is_unique
# %%
# the corr method will plot the correlation of our vrabiales
# this will make it easy for us for feature selection
network.corr()
# %%
network.corr("spearman")
# %%
# Import inventory
CITY = "Hannover"
CITY = CITY.title()
path = "data/" + CITY + "/inventory.pickle"
with open(path, 'rb') as inventory:
    inventory = pickle.load(inventory)

# %%
inventory.head()
# %%
# create our pipe objects in a dictionary 
object_inventory = {}
# in this analysis we are interested in Kunststoffmantelrohr
# therefore we filter them based on their type
city_abr = CITY[0:3].upper()
for i in inventory.loc[inventory.Type == 'KMR'].iterrows():
    # Useing KMR class to instantiate pipe objects
    object_inventory[city_abr + str(i[0])] = pipe.KMR(
    ID=int(i[1][1]),
    TYPE=pipe.PipeSystem(2),
    medium_count=pipe.MedPipeCount(1),
    dn=i[1][2],
    laying=i[1][3],
    length=i[1][4],
    flow=i[1][5],
    build_year=i[1][6],
    connection=i[1][7],
    life_status=i[1][8],
    failure_years=i[1][9],
    failure_degrees=i[1][10],
    failure_types=i[1][11],
    decommission_year=i[1][12],
    )
# %%
exeptions = ['HAN25437']

# %%
exeptions = []
for key, value in object_inventory.items():
    try:
        value.evaluate(network)
    except Exception as e:
        exeptions.append(key)
        #print(value.ID)
        print(e)
# %%
exeptions
# %%
if exeptions:
    for exeption in exeptions:
        del object_inventory[exeption]
# %%
for value in object_inventory.values():
    if value.decommission_year != 0:
        if value.build_year > value.decommission_year:
            print(value.ID)

# %%
length_object = []
for key, value in object_inventory.items():
    value.evaluate(network)
    length_object.append((key, len(value.dataframe().columns)))

# %%
maximum = max([x[1] for x in length_object])
indices = [length_object[i] for i, x in enumerate([x[1] for x in length_object]) if x == maximum]
key = indices[0][0]
#%%
# Human Redable
inventory_analysis = object_inventory[key].dataframe(network = network, augment = True, format = 'value')
for value in object_inventory.values():
    df = value.dataframe(network = network, augment = True, format = 'value')
    inventory_analysis = pd.concat((inventory_analysis, df), ignore_index = True, axis = 0)
# Drop first row
inventory_analysis.drop(index=inventory_analysis.index[0], 
        axis=0, 
        inplace=True)
# %%
inventory_analysis = inventory_analysis.fillna(0)
inventory_analysis.head()

# %% MAINTENANCE ____________________________________

# Import inventory_analysis
A = "Hannover"
# %%
#with open('data/Hannover/inventory_dummy.csv','w') as csv:
#    inventory_A.to_csv(csv)
# %%
#test = pd.read_csv('data/Hannover/inventory_dummy.csv', sep=',', decimal=',')

# %%
inventory_A = inventory_analysis
# %%
inventory_A.shape
# %%
inventory_A.insert(0, "Network", ["A" for i in range(len(inventory_A))])
# %%
inventory_orig = inventory_A.fillna(0)
# %%
inventory_A.loc[(inventory_A.Failures == 1) & (inventory_A.Aged == 1)]
# %% 
# %%
# 2.0 Data Exportation
inventory_orig.head()
#%%
inventory.head()
#%%
inventory_A.head()
# %%
# Make a binary failure event
inventory_A['Failure_Target'] = np.where((((inventory_A.Condition==2) | (inventory_A.Condition==4)) & (inventory_A.Aged==1)), 1, 0)
# %% No failures --> artificially creating them with this data
inventory_A.loc[0:500,"Failure_Target"] = 1
# %%
inventory_A.head()
#%%
inventory_A.drop(['Network'], axis = 1, inplace = True)
# %%
maint_obj = maintenance.Maintenance(inventory_A)

# %%
# 3.0 Data transformations and Feature Engineering
# define your feature window. This is the window by which we will aggregate our pipe values.
# The feature window is a parameter that depends on the context of the business problem.
# I am setting the value to 10 years
# Calculate the number of years from the first year a pipe appears to the current year.
# This field will be called “Aged” Also, create a variable called “too_soon.” When “too_soon” is equal to 0,
# we have less than 10 years (feature_window) of history for the pipe.
# this is integrated in the pdm file as "Aged"
maint_obj.prepare()

maint_obj.split_data(split = 0.2, valid = 0.1, target = 'Failure_Target')
# %%
# 4.3 SMOTE the Training Data
# Note that we are only balancing the training data set. You may be asking why.
# Remember that our goal is to build a model the represents reality, right?
# When we SMOTE the data, we change the failure rate to 50%.
# This is nowhere near what we see in the actual pipe data.
# Thus, it makes sense to build the model on the SMOTE data but evaluate it on the unaltered data.
# The unaltered data will be a better reflection of what to expect when you deploy the model to production.
# Define the Training features and Target.

# Synthetically Balance the training data sets with a SMOTE algorithm.
# After we apply the SMOTE algorithm, we will have a balanced data set.
# 50% Failures and 50% Non-Failures. Note that this takes a while to run.
maint_obj.SMOTE()

# %%
### 6.0 Build the model on the balanced training data set
# %%
# Define model specs.
# %%
# We are initializing our model with default model parameters.
# Note that we could probably improve the results by tweaking the parameters,
# but we will save that exercise for another day. 
xgb0 = XGBClassifier(objective='binary:logistic', use_label_encoder=False)
maint_obj.fit(model = xgb0)
# %%
### 7.0 Evaluate the Model
# Probably the most confusing element of PM problems is building a realistic assessment of the model.
# Because of timing and the small number of failures, understanding how the model will work once deployed
# in production is challenging. There are standard metrics for evaluating models like accuracy, AUC,
# and a confusion matrix.  In sections 7.1 and 7.2, I will show how, given the transformations we used
# to build our model and the complexity of the problem, these metrics do not give us a realistic view
# of model performance when deployed into production. These standard metrics are definitely useful but
# are not sufficient.
# In section 7.3, I lay out how I typically evaluate PM models.

# %%
#### 7.1 Evaluate the model using an AUC and accuacy metrics.
#First, we will evaluate the balanced training data with the default, a 50% cut-off.
# For information on how to find the best cut-off for these types of problems, please see the following.
# https://medium.com/swlh/determining-a-cut-off-or-threshold-when-working-with-a-binary-dependent-target-variable-7c2342cf2a7c

maint_obj.evaluate()

maint_obj.feature_importance()



# %%
