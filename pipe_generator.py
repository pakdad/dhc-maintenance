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
import pandas as pd
import random
import dhc_maintenance.pipe as pipe
import matplotlib.pylab as plt
import pickle

# %%
random.seed(2022)
# %%
def sample_maker(df, col, k=1):
    """Generate a weighted sample according to a dataframe"""
    bins = df[col].unique().tolist()
    bins.sort()
    population = bins.copy()
    bins.append(bins[-1]+1)
    bins_freq = np.histogram(df[col], bins)[0].tolist()
    bins_weights = [i/sum(bins_freq) for i in bins_freq]
    sample = random.choices(population, weights=bins_weights, k=k)
    return sample
# %%
# according to the data of network A
dn_pop = [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400, 500, 600]
DNbins_weights = [0.058407738095238096,
                  0.027777777777777776,
                  0.14620535714285715,
                  0.17373511904761904,
                  0.1314484126984127,
                  0.1300843253968254,
                  0.07465277777777778,
                  0.017857142857142856,
                  0.0951140873015873,
                  0.060515873015873016,
                  0.043402777777777776,
                  0.03732638888888889,
                  0.000992063492063492,
                  0.001984126984126984,
                  0.000248015873015873,
                  0.000248015873015873]
# %%
DN = random.choices(dn_pop, weights=DNbins_weights, k=1000)
# %%
length_bins=np.arange(25,250,25)
Lengthbins_weights = [0.7872393247269116,
 0.1284756703078451,
 0.047418073485600794,
 0.018123138033763656,
 0.00893743793445879,
 0.004965243296921549,
 0.0026067527308838135,
 0.001737835153922542,
 0.0004965243296921549]
# %%
discrete_length = random.choices(length_bins, weights=Lengthbins_weights, k=1000)
# %%
continuous_length = []
for i in discrete_length:
    continuous = random.choices(np.arange(i-24,i), k=1)[0]
    continuous_length.append(continuous)
# %%
setOfIDs = set()
while len(setOfIDs) < 1000:
    setOfIDs.add(random.randint(100000, 999999))
setOfIDs = list(setOfIDs)
# %%
f_h =["f" if random.choices(["f", "h"], [.03,.97])[0] == "f" else "h" for i in range(1000)]
build_year = [random.choice(np.arange(1972,2022)) for i in range(1000)]
failure_years = [[random.choice(np.arange(build_year[i],2022))] if f_h[i]=="f" else [0] for i in range(1000)]
life_status=[pipe.Status(random.choices([0,1], [.1,.9])[0]).name for i in range(1000)]
failure_degrees=[[pipe.FailureLevels(0).name] if failure_years[i] ==[0]
                 else [pipe.FailureLevels(random.choice([1,2])).name]
                 for i in range(1000)]
failure_type=[[pipe.FailureType(0).name] if failure_years[i] ==[0]
                 else [pipe.FailureType(random.choice([1,2,3,4,5,6])).name]
                 for i in range(1000)]
decommission_year = []
for count, val in enumerate(life_status):
    if failure_years[count] != [0] and val == "Decommissioned":
        decommission_year.append(failure_years[count][0])
    elif failure_years[count] == [0] and val != "Decommissioned":
        decommission_year.append(0)
    else:
        decommission_year.append(random.choice([build_year[count]+1,2023]))


laying_system = ['Burried' for i in range(1000)]
pipe_type = ['KMR' for i in range(1000)]
flow = [pipe.Flow(random.choice([0,1])).name for i in range(1000)]
connection = [pipe.PipeConnection(random.choice([1,2])).name for i in range(1000)]
# %%
inventory = {
    'Type' : pipe_type,
    'ID' : setOfIDs,
    'DN': DN,
    'Laying System' : laying_system,
    'Length' : continuous_length,
    'Flow' : flow,
    'Build Year' : build_year,
    'Connection' : connection,
    'Life Status' : life_status,
    'Failure Year' : failure_years,
    'Failure Degree' : failure_degrees,
    'Failure Types' : failure_type,
    'Decommission Year' : decommission_year, 
}
# %%
inventory = pd.DataFrame(inventory) 
inventory.head()  

# %%
with open("data/inventory_dummy.csv", "w") as file:
    inventory.to_csv(file)
# %%
