import numpy as np
import pandas as pd
from datetime import datetime
import rainflow as rf
from ftplib import FTP
import requests
from . import toolkit as tools
import os
import copy
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from enum import Enum
import copy
style.use('ggplot')

class PipeSystem(Enum):
    '''Pipe system types'''
    FreePipes = 1
    KMR = 2
    MMR = 3
    PMR = 4
    SMR = 5
    FLE = 6

class MedPipeCount(Enum):
    '''Number of pipes in system'''
    Single = 1
    Twin = 2
    Triplet = 3
    Quad = 4

class PipeConnection(Enum):
    '''Connection of pipe'''
    Line = 1
    House = 2
    Unknown = 9

class LayingSystem(Enum):
    '''Laying system of pipe'''
    OverHead = 1
    Burried = 2
    Canal = 3
    Building = 4
    Unknown = 9

class Flow(Enum):
    '''Specifies flow or supply'''
    Return = 0
    Supply = 1

class Status(Enum):
    '''Pipe status'''
    Decommissioned = 0
    InOperation = 1

class Failure(Enum):
    '''Is the pipe damaged?'''
    NotDamaged = 0
    Damaged = 1

class FailureLevels(Enum):
    '''Is the failure significant?'''
    NoFailure = 0
    Trivial = 1
    Substantial = 2

class FailurePart(Enum):
    '''Tells which part failed'''
    NoFailure = 0
    Compensator = 1
    Fitting = 2
    Weld = 3
    Shaft = 4
    Mount = 5
    Isolation = 6
    Unknown = 9

class FailureCause(Enum):
    '''Specifies cause of failure'''
    NoFailure = 0
    Ageing = 1
    Installation = 2
    ThirdParty = 3
    other = 4
    Unknown = 9

class FailureType(Enum):
    '''Specifies how pipe failed'''
    NoFailure = 0
    Corrosion = 1
    Fatigue = 2
    Isolation = 3
    Installation = 4
    Leakage = 5
    ThirdParty = 6
    Unknown = 9

class PipeCondition(Enum):
    '''Condition of pipe'''
    Healthy = 1
    good = 2
    Risky = 3
    Discarded = 4

class Aged(Enum):
    '''is the pipe aged??'''
    TooSoon = 0
    Aged = 1
    
class PipeDNCat(Enum):
    '''DN Category of pipe'''
    S = 0
    M = 1
    L = 2
    XL = 3
    XXL =4

class Pipe:
    """Create pipe object and evaluate the pipe attributes."""

    def __init__(self, ID, TYPE, dn, laying, length, flow, build_year, connection,
                 life_status, failure_years, failure_degrees, failure_types, decommission_year, 
                 augment =0):
        """Instantiate pipe object with mentioned attributes."""
        self.ID = ID
        self.type = TYPE
        self.dn = dn
        self.laying = laying
        self.length = length
        self.flow = flow
        self.build_year = build_year
        self.connection = connection
        self.life_status = life_status
        self.failure_years = failure_years
        self.failure_degrees = failure_degrees
        self.failure_types = failure_types
        self.decommission_year = decommission_year
        self.life()
        self.failure_year_prediction = 0
        self.DNCat()
        self.augment = augment
        self.start_date = None
        self.end_date = None
        self.temperatur_BG = None
        self.temperatur_min = None
        self.temperatur_max = None
        self.temperatur_mean = None
        self.temperatur_median = None
        self.counts = None
        self.mean = None
        self.eflc = None
        self.count = None
        self.delta_T = None
        self.delta_s = None
        self.fatigue_magnitude = None
        ## For evaluate
        self.status_at_failure = []
        self.failure_count = []
        self.failures = []
        self.cycles = []
        self.eflcs = []
        self.fatigues = []
        self.temperature_BGs = []
        self.pipe_df = None


    def __str__(self):
        return self.ID

    def record(self, failure_year, failure_degree, life_status):
        """Append failure events."""
        if self.failure_years[0] == 0:
            self.failure_years = []
        self.failure_years += [failure_year]
        self.failure_degrees += [failure_degree]
        self.life_status = life_status
        if life_status.value == 0:
            self.decommission_year = failure_year
            self.life()

    def life(self):
        """Calculate age of the pipe according to life status."""
        self.age_at_failures=[]
        for year in self.failure_years:
            if year == 0:
                self.age_at_failures += [0]
            else:
                self.age_at_failures += [year - self.build_year]
        if self.life_status.value == 1:
            self.age = datetime.now().year - self.build_year
            self.age_at_decommission = 0
        else:
            self.age = self.decommission_year - self.build_year
            self.age_at_decommission = self.decommission_year - self.build_year
        if self.age > 10:
            self.aged = Aged(1)
        else:
            self.aged = Aged(0)

    def DNCat(self):
        '''Turns dncat value into enum'''
        if self.dn <= 65:
            self.dncat= PipeDNCat(0)
        elif self.dn <= 150:
            self.dncat= PipeDNCat(1)
        elif self.dn <= 250:
            self.dncat= PipeDNCat(2)
        elif self.dn <= 400:
            self.dncat= PipeDNCat(3)
        else:
            self.dncat= PipeDNCat(4)

    def cycle_count(self, time_series, start_year='', end_year='', optimizer=None, binsize=1):
        """Count cycles based on rainflow algorithm."""
        # convertign build year and age to time stamp to
        # slice with index from the temperature change history data-frame
        if start_year:
            start_date = str(start_year)+"-01-01 00:00:00"
            start_date = datetime.strptime(start_date,"%Y-%m-%d %H:%M:%S")
        else:
            start_date = str(self.build_year)+"-01-01 00:00:00"
            start_date = datetime.strptime(start_date,"%Y-%m-%d %H:%M:%S")

        self.start_date = start_date

        if end_year:
            end_date = str(end_year)+"-12-31 23:00:00"
            end_date = datetime.strptime(end_date,"%Y-%m-%d %H:%M:%S")
        else:
            if self.life_status.name == "Decommissioned":
                end_date = str(self.decommission_year)+"-12-31 23:00:00"
                end_date = datetime.strptime(end_date,"%Y-%m-%d %H:%M:%S")
            else:
                end_date = time_series.index[-1]

        self.end_date  = end_date

        # count cycles with rainflow algorithm
        # counting bins set to 1 degree centigrad according to 
        # IEA DHC Fatigue Analysis of District Heating Systems 1999
        counts = rf.count_cycles(time_series.loc[self.start_date:self.end_date].to_numpy(), 
        binsize=binsize)
        self.temperatur_BG = round(
            time_series.loc[self.start_date:self.end_date].to_numpy().min())
        self.temperatur_min = round(
            time_series.loc[self.start_date:self.end_date].to_numpy().min())
        self.temperatur_max = round(
            time_series.loc[self.start_date:self.end_date].to_numpy().max())
        self.temperatur_mean = round(
            time_series.loc[self.start_date:self.end_date].to_numpy().mean())
        self.temperatur_median = round(np.median(
            time_series.loc[self.start_date:self.end_date].to_numpy()))


        if optimizer:
            sim_adjusted = []
            sim  = [i[1] for i in counts]
            for i, obj in enumerate(optimizer):
                if i < len(sim):
                    if obj != 0.0:
                        sim_adjusted += [sim[i]+sim[i]*obj*self.age/3]
                    else:
                        sim_adjusted += [sim[i]*self.age/3]
                else:
                    sim_adjusted += [obj*self.age/3]

            counts = [(i+1,j) for i,j in enumerate(sim_adjusted)]

        # extrapolate cycle if the simulation is not available in range
        if start_date < time_series.index[0]:
            # t1 is the difference from the build year
            # to the first available date of simulation
            # t2 is the time span where cycle counts performed in the simulation
            t1 = time_series.index[0] - self.start_date
            t1 = round(t1.days / 365.25)
            t2 = time_series.index[0] - self.end_date
            t2 = round(t2.days / 365.25)
            for count in counts:
                number = count[1] + (count[1] / t2) * t1
                count = (count[0], number)

        self.counts = counts
        return self.counts, self.temperatur_BG

    def mean_cycle(self, counts):
        """Calculate mean cycle value"""
        coefficient = []
        for i in counts:
            if i[1] != 0:
                coefficient += [i[0]*i[1]]
        self.mean = sum(coefficient)/len(coefficient)
        return self.mean

    def count_N_0(self, *time_series):
        """Calculate equitvalent full temperature cycles."""
        if self.counts == 0:
            if not time_series:
                print("Please pass a time series to method!")
            else:
                self.cycle_count(time_series[0])
        else:
            pass
        n_i = []
        for i in self.counts:
            n_i += [i[1]*i[0]**4]
        self.eflc = sum(n_i)/(110**4)
        return self.eflc
    
    def elastic_modulus(self, temperature):
        """Calculate E modulus at elevated temperature for P2365GH
        according to DIN EN 13941-1:2019-12 5.2.3.3."""
        E = (21.4 - (temperature/175))*10**4 # (N/mm^2) (MPa)
        return E
    
    def thermal_expansion_coefficient(self, temperature):
        """Calculate thermal expansion coefficient at elevated temperature 
        for P2365GH according to DIN EN 13941-1:2019-12 5.2.3.3."""
        alpha = (11.4 + (temperature/129))*10**-6 # (1/K)
        return alpha
    
    def fatigue(self, *time_series):
        """Thermal-Fatigue analysis of exerted stress by temperature changes."""
        if self.counts == 0:
            if not time_series:
                return print("Please pass a time series to method!")
            else:
                self.cycle_count(time_series[0])
                
        else:
            pass
        count = [ count[1] for count in self.counts ]
        delta_T = [ count[0] for count in self.counts ]
        delta_s = []
        # Calculate stress at elevated temperature for P2365GH
        # according to DIN EN 13941-1:2019-12 4.4.2
        for i in delta_T:
            T = i + self.temperatur_BG - 10
            delta_s += [(self.thermal_expansion_coefficient(T) *
                           self.elastic_modulus(T)*i)] # (MPa)
        self.count = count
        self.delta_T = delta_T
        self.delta_s = delta_s
        
        ni_Ni = []
        for ni,si in zip(self.count, self.delta_s):
            Ni = (5000/si)**4
            ni_Ni += [ni/Ni]

        self.fatigue_magnitude = sum(ni_Ni)
        return self.delta_s, self.fatigue_magnitude  
            
    def evaluate(self, network, optimizer=None, binsize=1):
        """Evaluate damage history and the statuse quo of the pipe."""
        if len(self.fatigues) == 0 and len(self.failure_count) != 0:
            return "Function already ran"
        
        for i, j in enumerate(self.failure_years):
            if j == 0 and self.life_status.value == 1:
                j = network.simulation['Supply'].index[-1].year
                i = -1
                status = Status(1)
                failed = Failure(0)
            elif j == 0 and self.life_status.value == 0:
                break
            else:
                if j == self.decommission_year:
                    status = Status(0)
                else:
                    status = Status(1)
                failed = Failure(1)
            if self.flow.name == "Supply":
                # in case of optimized calculation here it is needed to pass supply_coor attribute
                # if not you can leave the supply_corr empty
                # 1 stands for the binsize. if nothing given it will be concidered as 1
                # for optimization we can pass network.supply_corr
                counts, temperatur_BG = self.cycle_count(network.simulation['Supply'], end_year=j, optimizer=optimizer, binsize=binsize)
                eflc = self.count_N_0()
                delta_s, fatigue_magnitude = self.fatigue()
            else:
                # in case of optimized calculation here it is needed to pass return_coor attribute
                # if not you can leave the return_corr empty
                # 1 stands for the binsize. if nothing given it will be concidered as 1
                # for optimization we can pass network.return_corr
                counts, temperatur_BG = self.cycle_count(network.simulation['Return'], end_year=j, optimizer=optimizer, binsize=binsize)
                eflc = self.count_N_0()
                delta_s, fatigue_magnitude = self.fatigue()

            self.status_at_failure += [status]
            self.failure_count += [i+1]
            self.failures += [failed]
            self.cycles += [counts]
            self.eflcs += [eflc]
            self.fatigues += [fatigue_magnitude]
            self.temperature_BGs += [temperatur_BG]

        if self.life_status.value == 0:
            if self.decommission_year == self.failure_years[-1]:
                pass
            else:
                year = self.decommission_year
                if self.flow.name == "Supply":
                    # in case of optimized calculation here it is needed to pass supply_coor attribute
                    # if not you can leave the supply_corr empty
                    # 1 stands for the binsize. if nothing given it will be concidered as 1
                    # for optimization we can pass network.supply_corr
                    counts, temperatur_BG = self.cycle_count(network.simulation['Supply'], end_year=year, optimizer=optimizer, binsize=binsize)
                    eflc = self.count_N_0()
                    delta_s, fatigue_magnitude = self.fatigue()
                else:
                    # in case of optimized calculation here it is needed to pass return_coor attribute
                    # if not you can leave the return_corr empty
                    # 1 stands for the binsize. if nothing given it will be concidered as 1
                    # for optimization we can pass network.return_corr
                    counts, temperatur_BG = self.cycle_count(network.simulation['Return'], end_year=year, optimizer=optimizer, binsize=binsize)
                    eflc = self.count_N_0()
                    delta_s, fatigue_magnitude = self.fatigue()

                self.status_at_failure += (Status(0))
                self.failure_count += [end(i)]
                self.failures += [Failure(0)]
                self.cycles += [counts]
                self.eflcs += [eflc]
                self.fatigues += [fatigue_magnitude]
                self.temperature_BGs += [temperatur_BG]

    
    def dataframe(self, format='name', augment=False, n=3, network = None):
        """Create a row of dataframe for data analysis."""
        ## Making the function work for both 
        if not network and augment:
            return "need network object in arg \"network\" in order to augment. And need network\n to evaluate and get correct failure values"
        try:
            self.medium_count
        except:
            self.activation_rates = [0] * len(self.failure_years)
            ## Need to put in default value
            self.medium_count = MedPipeCount(1)
        for i,failure in enumerate(self.failure_years):
            df = {
                'ID':[self.ID],
                'Type': [getattr(self.type, format)],
                ## Medium Pipe
                'Medium Pipe Mode': [getattr(self.medium_count, format)],
                'DN':[self.dn],
                'DNCat':[getattr(self.dncat, format)],
                'Laying System':[getattr(self.laying, format)],
                'Length':[self.length],
                'Flow':[getattr(self.flow, format)],
                'Build_Year':[self.build_year],
                'Connection':[getattr(self.connection, format)],
                'Life_Status':[getattr(self.life_status, format)],
                'Failures':[getattr(self.failures[i], format)],
                'Status_at_Failure':[getattr(self.status_at_failure[i], format)],
                'Failure_Year':[failure],
                'Failure_Count':[self.failure_count[i]],
                'Failure_Degree':[getattr(self.failure_degrees[i], format)],
                'Failure_Types':[getattr(self.failure_types[i], format)],
                'Decommission_Year':[self.decommission_year],
                'Age':[self.age],
                'Aged':[getattr(self.aged, format)],
                'Age_at_Failure':[self.age_at_failures[i]],
                'Age_at_Decommission':[self.age_at_decommission],
                'EFLC':[self.eflcs[i]],
                'Fatigue':[self.fatigues[i]],
                'Temperature_Background':[self.temperature_BGs[i]],
                'Temperature_Min':[self.temperatur_min],
                'Temperature_Max':[self.temperatur_max],
                'Temperature_Means':[self.temperatur_mean],
                'Temperature_Median':[self.temperatur_median],
                ## Pur age
                'PUR Age':[self.activation_rates[i]],
                'Augmented':[self.augment]
                }
            


            if self.failures[i].value == 0 and self.status_at_failure[i].value == 1:
                df["Condition"] = getattr(PipeCondition(1), format)
            elif self.failures[i].value == 1 and self.status_at_failure[i].value == 1:
                df["Condition"] = getattr(PipeCondition(2), format)
            elif self.failures[i].value == 0 and self.status_at_failure[i].value == 0:
                df["Condition"] = getattr(PipeCondition(3), format)
            elif self.failures[i].value == 1 and self.status_at_failure[i].value == 0:
                df["Condition"] = getattr(PipeCondition(4), format)

            for j in self.cycles[i]:
                df["ΔT " + str(j[0])]=j[1]

            if self.aged.value == 1:
                for j in range(len(self.cycles[i])):
                    k = -j-1
                    if self.cycles[i][k][1] != 0:
                        index = len(self.cycles[i])+k
                        break
                df["Max_ΔT"] = self.cycles[i][index][0]
                df["Mean_Cycle"] = self.mean_cycle(self.cycles[i])
            else:
                ## meant to be 0 or 1???? 
                df["Max_ΔT"] = 1
                df["Mean_Cycle"] = 1

            df = pd.DataFrame(df)
            ## Cleaning up if not KMR
                ## maybe just check first/middle/last instead of looping??
            if self.medium_count == MedPipeCount(1) and all(i == 0 for i in self.activation_rates):
                pipe_data.drop(['PUR Age','Medium Pipe Mode'], axis = 1, inplace=True)
            if i == 0:
                pipe_data = df
            else:
                pipe_data = pd.concat([pipe_data,df], ignore_index = True, axis = 0) # ignore_index = True)
        
        self.pipe_df = pipe_data
        if augment:
            if self.medium_count == MedPipeCount(1) and all(i == 0 for i in self.activation_rates):
                return "Must be of type KMR to augment"
            else:
                self.augment_pipe(network=network, n=n, format = format)
        ## Equivalent of R case_when --> evaluates conditions then sets to respective
        ## choice if true
        conditions = [
                 self.pipe_df['Condition'].eq(getattr(PipeCondition(2), format)),
                 self.pipe_df['Condition'].eq(getattr(PipeCondition(4), format)) & self.pipe_df['Aged'].eq(getattr(Aged(1), format))
                    ]

        choices = [1,1]

        self.pipe_df['Failure_Target'] = np.select(conditions, choices, default=0)
        return self.pipe_df
 
    
class KMR(Pipe):
    """Represent aspects of pipe, specific to pre-insulated pipe."""
    
    def __init__(self, ID, TYPE, medium_count, dn, laying, length, flow, build_year, connection,
                 life_status, failure_years, failure_degrees, failure_types, decommission_year, augment = 0):
        super().__init__(ID, TYPE, dn, laying, length, flow, build_year, connection,
                 life_status, failure_years, failure_degrees, failure_types, decommission_year, augment)
        self.medium_count = medium_count
        self.life()
        self.failure_year_prediction = 0
        self.DNCat()
           
    def arrhenius(self, *time_series):
        """Returns reaction rate constant of the foam subjected to medium pipe's heat."""
        # y=mx+b for arrhenius calculations based on DIN EN 253:2020-3 page 32
        m = (np.log10(10950) - np.log10(112.5)) / (120 - 190)
        b = np.log10(10950) - m * 120
        if self.counts == 0:
            if not time_series:
                return print("Please pass a time series to method!")
            else:
                self.cycle_count(time_series[0])
        else:
            pass
        temperature_hours = []
        for i in self.counts:
            # our data is hourely basis, therefore each half cycle indicates an hour.
            # to interpolate the time we need daily basis data therefore we devide it
            # to 12 hours as below.
            temperature_hours += [(i[0]+self.temperatur_BG, (i[1] / 12))]
        sigma = []
        for i in temperature_hours:
            sigma += [i[1] / (10**(m * i[0] + b))]
        self.activation_rate = sum(sigma)
        return self.activation_rate

    def evaluate(self, network, optimizer=None, binsize=1):
        """Evaluate damage history and the statuse quo of the pipe."""
        self.status_at_failure = []
        self.failure_count = []
        self.failures = []
        self.cycles = []
        self.eflcs = []
        self.fatigues = []
        self.temperature_BGs = []
        self.activation_rates = []

        
        for i, j in enumerate(self.failure_years):
            if j == 0 and self.life_status.value == 1:
                j = network.simulation['Supply'].index[-1].year
                i = -1
                status = Status(1)
                failed = Failure(0)
            elif j == 0 and self.life_status.value == 0:
                break
            else:
                if j == self.decommission_year:
                    status = Status(0)
                else:
                    status = Status(1)
                failed = Failure(1)
            if self.flow.name == "Supply":
                # in case of optimized calculation here it is needed to pass supply_coor attribute
                # if not you can leave the supply_corr empty
                # 1 stands for the binsize. if nothing given it will be concidered as 1
                # for optimization we can pass network.supply_corr
                counts, temperatur_BG = self.cycle_count(network.simulation['Supply'], end_year=j, optimizer=optimizer, binsize=binsize)
                eflc = self.count_N_0()
                delta_s, fatigue_magnitude = self.fatigue()
                activation_rate = self.arrhenius()
            else:
                # in case of optimized calculation here it is needed to pass return_coor attribute
                # if not you can leave the return_corr empty
                # 1 stands for the binsize. if nothing given it will be concidered as 1
                # for optimization we can pass network.return_corr
                counts, temperatur_BG = self.cycle_count(network.simulation['Return'], end_year=j, optimizer=optimizer, binsize=binsize)
                eflc = self.count_N_0()
                delta_s, fatigue_magnitude = self.fatigue()
                activation_rate = self.arrhenius()

            self.status_at_failure += [status]
            self.failure_count += [i+1]
            self.failures += [failed]
            self.cycles += [counts]
            self.eflcs += [eflc]
            self.fatigues += [fatigue_magnitude]
            self.temperature_BGs += [temperatur_BG]
            self.activation_rates += [activation_rate]


        if self.life_status.value == 0:
            if self.decommission_year == self.failure_years[-1]:
                pass
            else:
                year = self.decommission_year
                if self.flow.name == "Supply":
                    # in case of optimized calculation here it is needed to pass supply_coor attribute
                    # if not you can leave the supply_corr empty
                    # 1 stands for the binsize. if nothing given it will be concidered as 1
                    # for optimization we can pass network.supply_corr
                    counts, temperatur_BG = self.cycle_count(network.simulation['Supply'], end_year=year, optimizer=optimizer, binsize=binsize)
                    eflc = self.count_N_0()
                    delta_s, fatigue_magnitude = self.fatigue()
                    activation_rate = self.arrhenius()
                else:
                    # in case of optimized calculation here it is needed to pass return_coor attribute
                    # if not you can leave the return_corr empty
                    # 1 stands for the binsize. if nothing given it will be concidered as 1
                    # for optimization we can pass network.return_corr
                    counts, temperatur_BG = self.cycle_count(network.simulation['Return'], end_year=year, optimizer=optimizer, binsize=binsize)
                    eflc = self.count_N_0()
                    delta_s, fatigue_magnitude = self.fatigue()
                    activation_rate = self.arrhenius()

                self.status_at_failure += [Status(0)]
                self.failure_count += [i]
                self.failures += [Failure(0)]
                self.cycles += [counts]
                self.eflcs += [eflc]
                self.fatigues += [fatigue_magnitude]
                self.temperature_BGs += [temperatur_BG]
                self.activation_rates += [activation_rate]

    ## n: number of new pipe objects to add
    def augment_pipe(self, network=None ,n=3, format = format):
        '''Augment pipe object by n '''
        if not network:
            return "need network object in arg \"network\" in order to generate data for dataframe"
        try:
            self.pipe_df
        except:
            return "Need dataframe object to augment. Run pipe.dataframe() first"
        ### Assuming full data(i.e. row for all years as in drawing)

        ## Get years all years that should be classified as fails and keep track of each new pipe
        all_fails = []
        for i in range(1,n + 1):
            fails = []
            for event in self.failure_years:
                ## Failed within first n years? Handled.
                if(event - i > self.build_year):
                    fails += [event - i]
            ## Off chance it failed in build year? Handled.
            if len(fails) > 0:
                all_fails += [fails]
            fails = []

        ## Augmenting pipe objects
            ## Create a new pipe for each new data point
            ## Run dataframe(augment = False) with all
            ## Add all together
            ## Add final df to initial df
        for i,new_fails in enumerate(all_fails):
            # Using KMR class to instantiate pipe objects
            #new_pipe = copy.deepcopy(self)
            #new_pipe.failures = new_fails
            # Useing KMR class to instantiate pipe objects
            new_pipe = KMR(
            ID = self.ID,
            TYPE=PipeSystem(2),
            medium_count=MedPipeCount(1),
            dn= self.dn,
            laying=self.laying,
            length=self.length,
            flow=self.flow,
            build_year=self.build_year,
            connection=self.connection,
            life_status=self.life_status,
            failure_years=new_fails,
            failure_degrees=self.failure_degrees,
            failure_types=self.failure_types,
            decommission_year=self.decommission_year,
            augment = 1
            )
            ## Adding on to pipe system we have
            new_pipe.evaluate(network)
            new_pipe.life()
            self.pipe_df = pd.concat([self.pipe_df, new_pipe.dataframe(augment = False, format = format)], ignore_index=True)
    
class KAN(Pipe):
    """Represent aspects of pipe, specific to canal pipe."""
    
    def __init__(self, ID, TYPE, medium_count, dn, laying, length, flow, build_year, connection,
                 life_status, failure_years, failure_degrees, failure_types, decommission_year, augment = 0):
        super().__init__(ID, TYPE, dn, laying, length, flow, build_year, connection,
                 life_status, failure_years, failure_degrees, failure_types, decommission_year, augment)
        self.medium_count = medium_count
        self.life()
        self.failure_year_prediction = 0
        self.DNCat()

    ## n: number of new pipe objects to add --> Maybe add this to intermediate class now
    def augment_pipe(self, network=None ,n=3, format = format):
        '''Augment pipe object by n '''
        if not network:
            return "need network object in arg \"network\" in order to generate data for dataframe"
        try:
            self.pipe_df
        except:
            return "Need dataframe object to augment. Run pipe.dataframe() first"
        ### Assuming full data(i.e. row for all years as in drawing)

        ## Get years all years that should be classified as fails and keep track of each new pipe
        all_fails = []
        for i in range(1,n + 1):
            fails = []
            for event in self.failure_years:
                ## Failed within first n years? Handled.
                if(event - i > self.build_year):
                    fails += [event - i]
            ## Off chance it failed in build year? Handled.
            if len(fails) > 0:
                all_fails += [fails]
            fails = []

        ## Augmenting pipe objects
            ## Create a new pipe for each new data point
            ## Run dataframe(augment = False) with all
            ## Add all together
            ## Add final df to initial df
        for i,new_fails in enumerate(all_fails):
            # Using KMR class to instantiate pipe objects
            #new_pipe = copy.deepcopy(self)
            #new_pipe.failures = new_fails
            # Useing KMR class to instantiate pipe objects
            new_pipe = KMR(
            ID = self.ID,
            TYPE=PipeSystem(2),
            medium_count=MedPipeCount(1),
            dn= self.dn,
            laying=self.laying,
            length=self.length,
            flow=self.flow,
            build_year=self.build_year,
            connection=self.connection,
            life_status=self.life_status,
            failure_years=new_fails,
            failure_degrees=self.failure_degrees,
            failure_types=self.failure_types,
            decommission_year=self.decommission_year,
            augment = 1
            )
            ## Adding on to pipe system we have
            new_pipe.evaluate(network)
            new_pipe.life()
            self.pipe_df = pd.concat([self.pipe_df, new_pipe.dataframe(augment = False, format = format)], ignore_index=True)
    
    