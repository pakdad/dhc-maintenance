# Walking through District Heating Pipe Predictive Maintenance
# %%
# import necessary libraries
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
network.load_temperature_changes("data/temperature_dummy.csv")
# %%
tools = toolkit.Tools()
# %%
tools.statistical_analyser(network.temperature_changes)
# %%
years = tools.extract_years(network.temperature_changes)
# %%
network.weather.data
# %%
network.weather.data.index.is_unique
# %%
network.temperature_changes.index.is_unique
# %%
network.concat()
# %%
# the corr method will plot the correlation of our vrabiales
# this will make it easy for us for feature selection
network.corr()
# %%
network.corr("spearman")
# %%
# based on our correlation plot we can drop the not intereted
# features and clean the datafram (nan values) for our training
col_list = ['Precipitaion_height', 'Precipitation_index', 'Precipitation_form']
network.drop_clean(col_list)
# %%
# This method will backward simulate up to the availabe
# historical temperature data
network.simulate()
# %%
# The optimize method is in testing phase now
# it is not necessary to run it but it is recommended
network.optimize()
# %%
print(network.supply_simulate.trainer.accuracy)
print(network.return_simulate.trainer.accuracy)
# %%
len(network.supply_training_set[1])
# %%
# Import inventory
path = "data/inventory_dummy.csv"
with open(path, 'r') as inventory:
    inventory = pd.read_csv(inventory,
                            index_col=0,
                            converters={"Failure Year": lambda x: [int(i.strip("[]").replace('"','')) for i in x.split(",")],
                                        "Failure Degree": lambda x: x.strip("[]").replace("'","").split(", "),
                                        "Failure Types": lambda x: x.strip("[]").replace("'","").split(", "),
                                        },
                            )
# %%
inventory.head()
# %%
# create our pipe objects in a dictionary 
object_inventory = {}
# in this analysis we are interested in Kunststoffmantelrohr
# therefore we filter them based on their type
# %%
for i in range(1000):
    # Useing KMR class to instantiate pipe objects
    object_inventory["HH-" + str(inventory.iloc[i]['ID'])] = pipe.KMR(
    ID=inventory.iloc[i]['ID'],
    TYPE=pipe.PipeSystem[inventory.iloc[i]['Type']],
    medium_count=pipe.MedPipeCount(1),
    dn=inventory.iloc[i]['DN'],
    laying=pipe.LayingSystem[inventory.iloc[i]['Laying System']],
    length=inventory.iloc[i]['Length'],
    flow=pipe.Flow[inventory.iloc[i]['Flow']],
    build_year=inventory.iloc[i]['Build Year'],
    connection=pipe.PipeConnection[inventory.iloc[i]['Connection']],
    # here we set 10% chance that the pipe is decomissioned
    life_status=pipe.Status[inventory.iloc[i]['Life Status']],
    failure_years=inventory.iloc[i]['Failure Year'],
    failure_degrees=[pipe.FailureLevels[inventory.iloc[i]['Failure Degree'][0]]],
    failure_types=[pipe.FailureType[inventory.iloc[i]['Failure Types'][0]]],
    decommission_year=inventory.iloc[i]['Decommission Year'],
    )

# %%
# This takes a while
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
    print(key)
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
A = "Hamburg"
# %%
#with open('data/Hamburg/inventory_dummy.csv','w') as csv:
#    inventory_A.to_csv(csv)
# %%
#test = pd.read_csv('data/Hamburg/inventory_dummy.csv', sep=',', decimal=',')

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
