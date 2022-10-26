'''
Instandhaltung-FW toolkit
'''
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as mb
import zipfile as zp
import numpy as np
import pandas as pd
import rainflow as rf
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



class Tools:
    """Some tools for data analysis that will be used for all the networks."""
    def __init__(self):
        self.about = print("This is a collection of tools that\nis frequently used for data analysis.")
        '''Calculates mode and mean for both supply and return and returns 5 number summaries for both supply and return'''
        self.mode_supply = None
        self.mean_supply = None
        self.mode_return = None
        self.mean_return = None
    
    # Calculates and returns the mode of a Pandas Series
    # return only the first mode always, so that the return value is a scalar
    def mode(self, x):
        '''Finds the Mode of the data'''
        return x.mode()[0]

    def mean(self, x):
        '''Calculates the mean to the nearest whole number'''
        return round(x.mean(), 0)

    def percentile_25(self, x):
        '''25th percentile of data'''
        return x.quantile(0.25)

    def percentile_75(self, x):
        '''75th percentile of data'''
        return x.quantile(0.75)

    def statistical_analyser(self, df):
        '''Calculates mode and mean for both supply and return and returns 5 number summaries for both supply and return'''
        self.mode_supply = df.Supply.mode()[0]
        self.mean_supply = df.Supply.mean()
        self.mode_return = df.Return.mode()[0]
        self.mean_return = df.Return.mean()
        statistical_measures = df.agg(
        {
            "Supply": ["min", "max", self.mean , "median", self.mode, self.percentile_25, self.percentile_75],
            "Return": ["min", "max", self.mean, "median", self.mode, self.percentile_25, self.percentile_75],
        }
        )
        return statistical_measures

    def extract_years(self, df):
        '''Selects the unique years of the data'''
        return df.groupby(df.index.year).first().index

    def season_selector(self, df, year='', season=''):
        '''Selects entries taking place in specified season'''
        if year == '':
            if season == 'warm':
                df = df.loc[(df.index.month > 3) & (df.index.month < 10)]
            elif season == 'cold':
                df = df.loc[(df.index.month > 9) | (df.index.month < 4)]
            else:
                return df
        else:
            if season == 'warm':
                df = df.loc[df.index[df.index.get_loc(str(year)+"-04-01 00:00:00", method='nearest')]:df.index[df.index.get_loc(str(year)+"-09-30 23:00:00", method='nearest')]]
            elif season == 'cold':
                df = df.loc[df.index[df.index.get_loc(str(year)+"-10-01 00:00:00", method='nearest')]:df.index[df.index.get_loc(str(year+1)+"-03-31 23:00:00", method='nearest')]]
            else:
                df = df.loc[df.index[df.index.get_loc(str(year)+"-04-01 00:00:00", method='nearest')]:df.index[df.index.get_loc(str(year+1)+"-03-31 23:00:00", method='nearest')]]
        return df
    def cm2inch(self, *tupl):
        '''Converts Centimeters to Inches'''
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)


def message(info):
    """Message with an ok botton."""
    root = tk.Tk()
    root.eval('tk::PlaceWindow %s center' % root.winfo_toplevel())
    root.withdraw()
    mb.showinfo('OK', info)
    root.destroy()


def read_dir(directory='/'):
    """Fetch folder path from windows."""
    root = tk.Tk()
    root.withdraw()
    root.directory = filedialog.askdirectory()
    root.destroy()
    if root.directory:
        read_last_directory = root.directory
    else:
        read_last_directory = '/'
    return root.directory, read_last_directory


def read_path(directory='/'):
    """Fetch file path from windows."""
    root = tk.Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilename(parent=root,
                                               initialdir=directory,
                                               title='Choose a file')
    root.destroy()
    if root.filename:
        read_last_directory = root.filename[::-1].split('/', 1)[1][::-1] + '/'
    else:
        read_last_directory = '/'
    return root.filename, read_last_directory


def read_pathes(directory='/'):
    """Fetch file pathes from windows."""
    root = tk.Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilenames(parent=root,
                                                initialdir=directory,
                                                title='Choose a file')
    root.destroy()
    if root.filename:
        read_last_directory = root.filename[0].rsplit('/', 1)[0] + '/'
    else:
        read_last_directory = '/'
    return root.tk.splitlist(root.filename), read_last_directory


def save_path(directory='/'):
    """Path to saving file directory."""
    root = tk.Tk()
    root.withdraw()
    root.filename = filedialog.asksaveasfilename(
        parent=root,
        initialdir=directory,
        title='Save a file',
        filetypes=(("CSV files", "*.csv"), ("Pickle files", "*.pickle")),
    )
    root.destroy()
    if root.filename:
        save_last_directory = root.filename[::-1].split('/', 1)[1][::-1] + '/'
    else:
        save_last_directory = '/'
    return root.filename, save_last_directory


def file_name_ext(pathes):
    """Extract file names from the pathes"""
    file_names = []
    if isinstance(pathes, str):
        name = pathes.split('/', )[-1]
        name = name.split('.', )[0]
        file_names.append(name)
    else:
        for path in pathes:
            name = path.rsplit("/", 1)[-1]
            file_names.append(name.rsplit(".", 1)[0])
    return file_names


def select_path(pathes):
    """Ask user which file to open."""
    if len(pathes) > 1:
        file_names = file_name_ext(pathes)
        info = "\nWhich file would you like to open? (please enter the index.)"

        count = 1
        for file in file_names:
            info += f"\n{count}" + ": " + file
            count += 1
            selected_file = input(info)
            selected_file = int(selected_file) - 1
    else:
        selected_file = 0
    return pathes[selected_file]


def zipindex(zfile, keyword):
    """Find index of the desirable file in zip file."""
    for i in range(len(zfile.infolist())):
        if str(zfile.infolist()[i]).find(keyword) >= 0:
            zindex = i
    return zindex


def date_check(date_array):
    """Date consistancy check."""
    inconsistency_index_list = []
    for i in range(0, date_array.shape[0]):
        if i == date_array.shape[0] - 1:
            pass
        else:
            if int(str(date_array[i + 1])) - int(str(date_array[i])) == 0:
                inconsistency_index_list.append(i)
    return inconsistency_index_list


def csvtopd(path='', **kwargs):
    """Open and build Pandas dataframe."""
    if not path:
        path = read_path()
        if not path:
            pass
        else:
            file = select_path(path)
    else:
        file = path
    if not path:
        pass
    else:
        if file.endswith(".zip"):
            zipfile = zp.ZipFile(file)
            zip_index = zipindex(zp.ZipFile(file), "produkt")
            dataframe = pd.read_csv(
                zipfile.open(zipfile.infolist()[zip_index]), **kwargs)
        else:
            dataframe = pd.read_csv(file, **kwargs)
    return dataframe


def insert_dir(info, directory=''):
    """Message and ask for directory."""
    message(info)
    path, last_directory = read_dir(directory)
    return path, last_directory


def insert_file(info, directory=''):
    """Message and ask for directory of the loading file."""
    message(info)
    path, last_directory = read_path(directory)
    files = file_name_ext(path)
    return files, path, last_directory



def mean_cycle(count_cycles):
    '''means of current and previous cycles'''
    mean = []
    X = 0
    for i in count_cycles:
        m = (i[0] + X) / 2
        mean.append((m, i[1]))
        X = i[0]
    return mean


def N(mean):
    '''Summing mean object calculations'''
    result = []
    for i in mean:
        count = 1 / (71**4) * i[1] * i[0]**4
        result.append(count)
    return sum(result)


def frequency_count(flag, temp):
    '''Finding frequency count'''
    count = []
    binlabels = []
    if flag == 'V':
        binInterval = [i for i in range(80, 126, 5)]
        for i, obj in enumerate(binInterval):
            if i != 0:
                binlabels.append((obj + binInterval[i - 1]) / 2)
        p = pd.cut(temp['Prediction_Supply'],
                   bins=binInterval,
                   labels=binlabels)
    else:
        binInterval = [i for i in range(50, 71, 5)]
        for i,obj in enumerate(binInterval):
            if i != 0:
                binlabels.append((obj + binInterval[i - 1]) / 2)
        p = pd.cut(temp['Prediction_Return'],
                   bins=binInterval,
                   labels=binlabels)

    for i in binlabels:
        count.append((i, (p.values == i).sum() / 24))
    return count


# y=mx+b for arrhenius calculations based on DIN EN 253:2020-3 page 32
m = (np.log10(10950) - np.log10(112.5)) / (120 - 190)
b = np.log10(10950) - m * 120


def arrhenius(flag, temp):
    '''Calculate Arhenius'''
    count = frequency_count(flag, temp)
    sigma = []
    for i in count:
        sigma.append(i[1] / (10**(m * i[0] + b)))
    return sum(sigma)

import os, os.path
import errno

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    '''Make Directory from path'''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path, write):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, write)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller  """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def trainer(X, y, regressor):
    """Scale, Learn, and return the trained regressor."""
    # Scale the features
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y,
                                                        test_size=0.2)
    # Fit the regressor
    regressor.fit(X_train, y_train)
    accuracy = regressor.score(X_test, y_test)
    return regressor, accuracy, X_scaled