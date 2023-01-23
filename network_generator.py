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
import numpy as np
import dhc_maintenance.simulation as simulation
import matplotlib.pylab as plt

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
# fig.savefig('fig/Artificial1.png', dpi='figure', format='png',)
# %%
with open('data/temperature_dummy_test.csv','w') as csv:
    network_temp_artificial.to_csv(csv)
# %%
