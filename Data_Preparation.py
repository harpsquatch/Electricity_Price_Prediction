# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:18:51 2022

@author: Harpreet Singh
"""

import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


#We have two .csv files, 4 year data of electrcity generation and weather in Spain, lets load them 
raw_df_energy = pd.read_csv("C:\\Users\\Harpreet Singh\\Documents\\Machine Learning\\Electricity Price Predictor\\Data\\energy_dataset.csv")

raw_df_weather = pd.read_csv("C:\\Users\\Harpreet Singh\\Documents\\Machine Learning\\Electricity Price Predictor\\Data\\weather_features.csv")


#Dataset features: 
    # Energy Data - Hourly data - Energy generation from various energy resources - Day ahead price and actual price - For 5 cities in Spain
    
    # Weather Data - Hourly data - Weather information about the geneartion of energy in Spain - For the same 5 cities in Spain
    

#Other facts: These fives cities are - Madrid, Bibao, Barcelona, Valencia and Seville. These cities comprises 1/3rd population of spain 


# We can witness some NaN values in the energy data. Some of the columns are completely useless too. 

#raw_df_energy.replace(0, np.nan, inplace=True) Cannot do this because after converting into NaN, you cannot know if they were actually 0 or no 

# So lets take all those columns in which there are 0s AND NaNs and delete them. Not we are not deleting those columns in which there are other values other than 0 and NaN 
f = raw_df_energy.replace([0,'n'], np.nan).apply(lambda x: any(~x.isnull()))
#f contains T/F   #replace 0,n, with nan         #apply local inverse isnull function to all columns and get bool values, True means there are other values other than 0 and NaN and we will npt delete them  


##raw_df_energy.info() 
raw_df_energy= raw_df_energy.loc[:,f]

#Time data is not correctly parsed and currently being recognised as an object 


raw_df_energy['time'] = pd.to_datetime(raw_df_energy['time'], utc =True, infer_datetime_format = True)


df_energy = raw_df_energy.set_index('time')



print("There are {} duplicates".format(df_energy.duplicated(keep = 'first').sum()))

# There are 0 Duplicates 

print("There are {} missing values or NaNs in the Energy dataset after deleting the NaN columns".format(df_energy.isnull().values.sum()))

#So we have 23991 missing values and NaNs, we just cannot drop the rows with missing values, need to deal with it.How about fill some value. 

#First lets see how many NaNs are actually present in the dataset

df_energy.isnull().sum(axis = 0)

#Also we wont be needing forecast related columns, so just delete them 

df_energy = df_energy.drop(['total load forecast','forecast solar day ahead','forecast wind onshore day ahead'], axis = 1)


# This is to plot the total plot 

fig, ax = plt.subplots(figsize=(30, 12))
ax.set_xlabel('Time', fontsize=16)
ax.plot(df_energy['total load actual'][0:24*7*2])
ax.legend(fontsize=16)
ax.set_title('Actual Total Load (First 2 weeks - Original)', fontsize=24)
ax.grid(True)

# It can be seen null values that there are null values for a few hours. This also indicates that an interpolation would 
#fill the NaNs well. Lets use linear interpolation with a forward direction. This is simplest cuz only a little part of the data is missing  


#It can also witnessed using the varaible explorer that the NaNs mostly coincide with each other 

df_energy.interpolate(method = 'linear', limit_direction = 'forward', inplace = True, axis = 0)

print('Non-zero values in each column:\n', df_energy.astype(bool).sum(axis = 0),sep = '\n')

df_energy.isnull().sum(axis = 0) #With this you can see we dont have any null. 


#Energy data has been cleaned succesfully and is ready for further use as input into our model.


#Lets prepare weather data as well. 



















