# %%
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
style.use('ggplot')


# %%
class Weather:
    """Weather object to retrieve data from DWD ftp server."""
    
    def __init__(self):
        """Instantiate weather object with mentioned attributes."""
        self.data = 0
        # DWD station text file url
        dwd = "https://opendata.dwd.de/climate_environment/CDC/help/TU_Stundenwerte_Beschreibung_Stationen.txt"
        data = requests.get(dwd)
        # Build data frame of station list
        text = data.content.decode('latin-1')
        self.text = text
        s_rows = text.split('\n')
        s_rows_cols = [each.split() for each in s_rows]
        for item in s_rows_cols:
            if len(item)>8:
                x = ""
                for i in range(len(item[6:-1])):
                    x = x + item[6+i]
                item[6] = x
                while len(item)>8:
                    del item[-2]
        header_row = ['Stations_id',
                      'von_datum',
                      'bis_datum',
                      'Stationshoehe',
                      'geoBreite',
                      'geoLaenge',
                      'Stationsname',
                      'Bundesland',
                     ]
        dwd_station = pd.DataFrame(s_rows_cols[2:], columns = header_row)
        dwd_station.index = dwd_station['Stations_id']
        dwd_station.drop(['Stations_id'], axis=1, inplace=True)
        self.dwd = dwd_station
        self.cities_to_ids = {"hannover": "02014", "saarbrücken": "04336", "köln": "02667"}
        
        
    def save_station_list(self):
        '''giving a name and saving it in any required format'''
        #opening the file in write mode
        with tools.safe_open_w("data/TU_Stundenwerte_Beschreibung_Stationen.txt", 'w') as f:
            #writes the URL contents from the server
            f.write(self.text)

    def search_station_ID(self, search):
        '''Finds station ID from search'''
        # return self.cities_to_ids[search.lower()]
        return self.dwd.loc[self.dwd["Stationsname"].str.contains(search, na=False, case=False)]
        
    def load_weather(self, ID):
        '''Loads weather data from specific ID'''
        ftp = FTP('opendata.dwd.de')
        ftp.login()
        DE_Hourly_path = "/climate_environment/CDC/observations_germany/climate/hourly"
        pathes = {'air_temperature': '',
          'cloudiness': '',
          'precipitation': '',
          'sun': '',
          'wind': '',
         }

        for folder in pathes:
            ftp.cwd(f'{DE_Hourly_path}/{folder}/historical')
            filenames = ftp.nlst()
            for filename in filenames:
                if ID in filename:
                    pathes[folder] = f"{DE_Hourly_path}/{folder}/historical/{filename}"
        
        DATA = {}
        for name, path in pathes.items():
            if path:
                with open("data/"+os.path.basename(path), 'wb') as f:
                        # Define the callback as a closure so it can access the opened 
                        # file in local scope
                        def callback(data):
                            f.write(data)
                        ftp.retrbinary('RETR ' + path, callback)
                DATA[name] = tools.csvtopd("data/"+os.path.basename(path), delimiter=";", index_col="MESS_DATUM")
        
        if 'air_temperature' in DATA:        
            DATA['air_temperature'].drop(columns=['STATIONS_ID', 'QN_9', 'eor'],
                                         inplace=True)
            DATA['air_temperature'].columns = ['Out_Temp', 'Hum']
            
        if 'cloudiness' in DATA:
            DATA['cloudiness'].drop(columns=['STATIONS_ID', 'QN_8', 'V_N_I', 'eor'],
                                    inplace=True)
            DATA['cloudiness'].columns = ['Cloud_Cover']
        
        if 'precipitation' in DATA:
            DATA['precipitation'].drop(columns=['STATIONS_ID', 'QN_8', 'eor'],
                                       inplace=True)
            DATA['precipitation'].columns = ['Precipitaion_height',
                                             'Precipitation_index',
                                             'Precipitation_form',
                                            ]
        if 'sun' in DATA:
            DATA['sun'].drop(columns=['STATIONS_ID', 'QN_7', 'eor'], inplace=True)
            DATA['sun'].columns = ['Sunshine_Duration']
        
        if 'wind' in DATA:
            DATA['wind'].drop(columns=['STATIONS_ID', 'QN_3', '   D', 'eor'], inplace=True)
            DATA['wind'].columns = ['Mean_Wind_Spdeed']

        # Check inconsistency in indices
        check_dupticates = dict()
        for key, value in DATA.items():
            check_dupticates[key] = tools.date_check(value.index.array)

        # Remove duplicated indices and keep the first one
        for key in DATA.keys():
            if check_dupticates[key]:
                DATA[key] = DATA[key].loc[~DATA[key].index.duplicated(keep='first')]
        df = pd.concat(DATA, axis=1, sort=False)
        df_columns = [k[1] for k in df.columns]
        df.columns = df_columns
                
        # Extract Month, Day, and Hour from the index
        df.index = pd.to_datetime(df.index, format='%Y%m%d%H')

        # Create holidays and off days from the week days
        dti = df.index
        daysofweek = dti.to_series().dt.dayofweek.tolist()

        off_days = [1 if day in (5, 6) else 0 for day in daysofweek]

        # Add Month, Hour, and Holiday columns to the dataframe
            ## Switched from pd.DatetimeIndex
        df['Month'] = df.index.month
        df['Hour'] = df.index.hour
        df['Off_Days'] = np.array(off_days)
        
        self.data = df

class Utility():
    """Utility temperature changes and backward simulation."""
    
    def __init__(self):
        self.weather = Weather()
        self.data = []
        self.training_set = []
        self.temperature_changes = None
        self.supply_training_set = None
        self.return_training_set = None
        self.X_history = None
        self.supply_simulate = None
        self.return_simulate = None
        self.simulation  = None
        self.counts_supply_real = None
        self.counts_return_real = None
        self.counts_supply_simulation = None
        self.counts_return_simulation = None
        self.supply_corr = None
        self.return_corr = None
        ## Stores most recent weather call
        self.weather = Weather()
        self.temperature_timeserie = None
        
    def load_temperature_changes(self, path):
        '''Extracts temperature changes from path'''
        temperature_changes = pd.read_csv(path,
                                          delimiter=",",
                                          encoding='utf8',
                                          header=[0],
                                          decimal='.',
                                          index_col=0,
                                         )
        temperature_changes.index = pd.to_datetime(temperature_changes.index, format='%Y%m%d%H')
        self.temperature_changes = temperature_changes
        
    def concat(self):
        '''Concatenates temperature changes to weather data'''
        if isinstance(self.weather.data, pd.DataFrame) and isinstance(self.temperature_changes, pd.DataFrame):
            self.data = pd.concat([self.temperature_changes, self.weather.data], axis=1, sort=False)
            # Slice weather_dataframe based on date to form
            # the dataframe with temperature change data
            START = list(self.temperature_changes.index)[0]
            END = list(self.temperature_changes.index)[-1]
            self.training_set = self.data[START:END]
        else:
            print("Please make sure that both weather and system temperature data is loaded!")
            
    def corr(self, *mode):
        '''Plots Pearson Correlation matrix'''
        if not mode:
            mode = 'pearson'
        else:
            mode = mode[0]
        if isinstance(self.training_set, pd.DataFrame):
            fig, ax = plt.subplots(figsize=(20,20))
            sns.heatmap(self.training_set.corr(method=mode), annot=True)
            ax.set_title('Variable Correlation')
            plt.show()
        ## /// Handle when not an Instance??
            
    def drop_clean(self, col_list=None):
        '''Drops specified columns and cleans data'''
        if col_list:
            self.data = self.data.drop(col_list, axis=1)
            self.data = self.data[self.data[self.data.Out_Temp > -999].index[0]:]
            self.training_set = self.training_set.drop(col_list, axis=1)
        
        # Clean dataframe
        # Check if the N/A values were continuous in some hours
        # it could be potentially a system shutdown, if it is just
        # one event it could be a log failure and will be replaced
        # by the mean value of the its neighbours
        null_index = self.training_set['Supply'].loc[self.training_set['Supply'].isnull()].index
        if null_index.to_list():
            # replace null values with mean of the neighbour values
            for null in null_index:
                index = self.training_set.index.get_loc(null)
                self.training_set.loc[null, 'Supply'] = (self.training_set.iloc[index-1, 0] + self.training_set.iloc[index+1, 0]) / 2
                self.training_set.loc[null, 'Return'] = (self.training_set.iloc[index-1, 1] + self.training_set.iloc[index+1, 1]) / 2
        # check if there are still null values in data-set
        null_index = self.training_set['Supply'].loc[self.training_set['Supply'].isnull()].index
        if null_index.to_list():
            print("There is null value in dataframe.\n Please clean again.")
        self.training_set.replace(to_replace=np.nan, value=-999.0, inplace=True)
        self.training_set.drop(self.training_set[self.training_set['Supply'] < 60].index, inplace = True)
        self.training_set.drop(self.training_set[self.training_set['Return'] < 20].index, inplace = True)
        # Prepare features and labels for supply and return
        X_supply = np.array(self.training_set.drop(['Supply', 'Return'], axis=1))
        y_supply = np.array(self.training_set['Supply'])
        self.supply_training_set = (X_supply, y_supply)

        X_return = np.array(self.training_set.drop(['Supply', 'Return'], axis=1))
        y_return = np.array(self.training_set['Return'])
        self.return_training_set = (X_return, y_return)
        
        self.data.replace(to_replace=np.nan, value=-999.0, inplace=True)
        self.X_history = np.array(self.data.drop(['Supply', 'Return'], axis=1))
    
    ## Will: Change to get method to just extract necessary things for simulate function??
    def simulate(self):
        """Backward simulate system temperature changes."""
        self.supply_simulate = Simulate(self.supply_training_set, self.X_history)
        self.return_simulate = Simulate(self.return_training_set, self.X_history)
        self.simulation = pd.DataFrame(data=np.stack((self.supply_simulate.predict,
                                                      self.return_simulate.predict),
                                                     axis=1),
                                       index=self.data.index,
                                       columns=["Supply", "Return"])
        print("Simulation done: Data in self.simulation, .return_simulate, .supply_simulate")
        
    def optimize(self, binsize=1):
        """Optimize cycle count of backward simulation"""
        # setting start and end date
        start_date = self.training_set.index[0]
        end_date = self.training_set.index[-1]
        # count cycles with rainflow algorithm
        # counting bins set to 1 degree centigrad according to 
        # IEA DHC Fatigue Analysis of District Heating Systems 1999
        counts_supply_real = rf.count_cycles(self.training_set.loc[start_date:end_date, "Supply"].to_numpy(), binsize=binsize)
        counts_return_real = rf.count_cycles(self.training_set.loc[start_date:end_date, "Return"].to_numpy(), binsize=binsize)
        counts_supply_simulation = rf.count_cycles(self.simulation.loc[start_date:end_date, "Supply"].to_numpy(), binsize=binsize)
        counts_return_simulation = rf.count_cycles(self.simulation.loc[start_date:end_date, "Return"].to_numpy(), binsize=binsize)
        
        counts_supply_real = [i[1] for i in counts_supply_real]
        counts_return_real = [i[1] for i in counts_return_real]
        counts_supply_simulation = [i[1] for i in counts_supply_simulation]
        counts_return_simulation = [i[1] for i in counts_return_simulation]

        def correlate(real, sim):
            """Ratio of the real counts to simulation counts."""
            corr = []
            for i,obj in enumerate(real):
                if i < len(sim):
                    if obj != 0.0:
                        corr.append(real[i]/obj)
                    else:
                        corr.append(real[i])
                else:
                    corr.append(real[i])
            return corr
        
        self.counts_supply_real = counts_supply_real
        self.counts_return_real = counts_return_real
        self.counts_supply_simulation = counts_supply_simulation
        self.counts_return_simulation = counts_return_simulation
        self.supply_corr = correlate(counts_supply_real, counts_supply_simulation)
        self.return_corr = correlate(counts_return_real, counts_return_simulation)
        print("optimization done")
    
    def city_to_sim(self, city, col_list = ['Precipitaion_height', 'Precipitation_index', 'Precipitation_form']):
        '''Takes a city name and produces simulation data for network'''
        
        # with wather.search_station_ID method we can search for
        # relavant station and find the station ID

        weather_id = self.weather.cities_to_ids[city.lower()]
        # "02014" for hanover --> 
        # weather.load_weather method will fetch the data
        # from the ftp server of the DWD and load it to our tool
        self.weather.load_weather(weather_id)


        path = 'data/' + city.title() + '/supply_return.csv'
        self.load_temperature_changes(path)


        temperature_timeserie = self.temperature_changes[self.temperature_changes['Supply']>60]
        temperature_timeserie = temperature_timeserie[temperature_timeserie['Return']>50]
        self.temperature_timeserie = temperature_timeserie.round(0)
        ## making sure data works??
        self.concat()

        self.drop_clean(col_list)
        self.simulate() 

        # it is not necessary to run it but it is recommended
        self.optimize()

        

class Simulate:
    """Backward simulate system temperature changes."""
    def __init__(self, training_set, X_history):
        self.trainer = Trainer(*training_set)
        self.predict = self.trainer.regressor.predict(self.trainer.scaler.transform(X_history))

class Trainer:
    """Scale, Learn, and return the trained regressor."""
    def __init__(self, X, y):
        self.regressor = RandomForestRegressor(random_state=32)
        self.X = X
        self.y = y
        # Scale the features
        self.scaler = preprocessing.StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled,
                                                            self.y,
                                                            test_size=0.2)
        # Fit the regressor
        self.regressor.fit(X_train, y_train)
        self.accuracy = self.regressor.score(X_test, y_test)


