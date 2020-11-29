import xlsxwriter
import pandas as pd # used to load and manipulate data
import numpy as np # data manipulation
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pandas import DataFrame
from sklearn.utils import resample
from sklearn.model_selection import train_test_split #split data into training and testing sets
from sklearn.preprocessing import scale #scale and center data
from sklearn.svm import SVC #SVM for classification
from sklearn.model_selection import GridSearchCV #Cross validation
from sklearn.metrics import confusion_matrix #plots confusion matrix
from sklearn.metrics import plot_confusion_matrix #draws a confusion matrix
from sklearn.decomposition import PCA #to perform PCA to plot the data

path = 'C:/Users/isaac/OneDrive/Documents/Projects/ML/SVM/default_of_credit_card_clients.xls'
df = pd.read_excel(path, header = 1, sep='\t')


print(df)

# pd.set_option("display.max_rows", None, "display.max_columns", None)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 0)

# df = pd.read_csv('default_of_credit_card_clients.csv', header=1, sep='\t')
# df.columns = ["default"]
df.rename(columns = {"default payment next month" : "DEFAULT"}, inplace = True)
print(df.head())
df.drop('ID', axis=1, inplace=True)
print(df.head())
# print(df.head().to_string()]) Useful for printing out all of data

df['SEX'].unique() #Make sure SEX only contains 1 and 2
df['EDUCATION'].unique() #Should only contain 1, 2, 3, 4
df['MARRIAGE'].unique() #Should only contain 1, 2, 3

print(len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)]))
