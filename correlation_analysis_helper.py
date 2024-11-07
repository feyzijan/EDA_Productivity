import neurokit2 as nk
import numpy as np
import pandas as pd
import datetime
import json
import pickle
import os
from logging import getLogger
from typing import Any
from warnings import warn
import sys
import seaborn as sns
import pylab 
import joblib
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from neurokit2.eda import eda_peaks
from pywt import wavedec
from scipy.stats import linregress
import scipy.stats as stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.decomposition import PCA
from scipy.stats import shapiro, kstest, anderson
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow and Keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.colors import ListedColormap

# Scikit-learn for evaluation
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import scipy

from data_prep_helper import *


def load_each_subject_individually():
    X_list = []
    y_list = []

    for p in p_list_a3:
        p = p.split("_")[0]
        folder_path = f"ModelDatasets/{p}/a3"
        X = pd.read_csv(f"{folder_path}/x.csv")    
        y = pd.read_csv(f"{folder_path}/y.csv")
        X_list.append(X)
        y_list.append(y)

    for p in p_list_a4:
        p = p.split("_")[0]
        folder_path = f"ModelDatasets/{p}/a4"
        X = pd.read_csv(f"{folder_path}/x.csv")    
        y = pd.read_csv(f"{folder_path}/y.csv")

        X_list.append(X)
        y_list.append(y)

    return X_list, y_list


'''
Check normality of features
'''
def check_normality(X):

    # Initialize a dictionary to store results
    normality_tests = {}

    for feature in X.columns:
        if pd.api.types.is_numeric_dtype(X[feature]):
            # Shapiro-Wilk Test
            stat_sw, p_value_sw = shapiro(X[feature])

            # Kolmogorov-Smirnov Test against normal distribution
            stat_ks, p_value_ks = kstest(X[feature], 'norm', args=(X[feature].mean(), X[feature].std()))

            # Anderson-Darling Test
            result_ad = anderson(X[feature], dist='norm')
            stat_ad = result_ad.statistic
            crit_values = result_ad.critical_values

            # P values should be >0.5
            normality_tests[feature] = {
                'Shapiro-Wilk p-value': round(p_value_sw,3),
                'KS Test p-value': round(p_value_ks,3),
                'Anderson-Darling statistic': stat_ad,
                'AD Critical values': crit_values
            }

    # Convert results to a DataFrame for easier viewing
    normality_tests_df = pd.DataFrame(normality_tests).T
    return normality_tests_df
    # print(normality_tests_df)


