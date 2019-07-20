# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:05:51 2019

@author: wyxx6
"""
# import numpy and pandas
import numpy as np
import pandas as pd
# import visualization libraries and set %matplotlib inline for jupyter
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# read in the csv file as a dataframe called df
df = pd.read_csv('911.csv')

# check the info() of the df
df.info()
# check the head of df
df.head()

# Q: What are the top 5 zipcodes for 911 calls?
df['zip'].value_counts().head(5)
# Q： What are the top 5 townships for 911 calls?
df['twp'].value_counts().head(5)
# Q： How many unique title codes are there?
len(df['title'].unique())
df['title'].nunique()

# Creating New Features (index location: iloc)
x = df['title'].iloc[0]
x.split(':')[0]
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

# What is the most common Reason for a 911 call based on this new column?
df['Reason'].value_counts()

# use seaborn to create countplot of 911 calls by Reason
sns.countplot(x = 'Reason', data = df)

# What is the the data type of the objects in the timeStamp column?
df.info()
df['timeStamp'].iloc[0]
type(df['timeStamp'].iloc[0])

# Use pd.to_datatime to convert the column from strings to DateTime objects
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

time = df['timeStamp'].iloc[0]
time.hour

# use apply() to create 3 new columns called Hour, Month, and Day of Week.
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

# the Day of Week is an integer of 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week
dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)

# Use seaborn to create a countplot of the Day of Week column with the hue baased on the Reason column
sns.countplot(x = 'Day of Week', data = df, hue = 'Reason', palette = 'viridis')
# relocate the legend
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)

# Do the same for month
sns.countplot(x = 'Month', data = df, hue = 'Reason', palette = 'viridis')
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)

# Months of 9, 10 , 11 are missing
# create a groupby object called byMonth.
byMonth = df.groupby('Month').count()
byMonth.head()

# create a simple plot off of the dataframe indicating the count of calls per month
byMonth['twp'].plot()

# if you can use seaborn implot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column
sns.lmplot(x = 'Month', y = 'twp', data = byMonth.reset_index())

# create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method
df['Date'] = df['timeStamp'].apply(lambda t: t.date())

# groupby the date column with the count() aggregate and create a plot of counts of 911 calls
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()

# recreate this plot but create 3 separate plots with each plot representing a Reason for 911
df[df['Reason'] == 'Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()

df[df['Reason'] == 'Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()

df[df['Reason'] == 'EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()

# creating heatmaps with seaborn and our data
# restructure the data frame so that the columns become the Hours and the Index becomes the Day of the Week. 
dayHour = df.groupby(by = ['Day of Week', 'Hour']).count()['Reason'].unstack()
dayHour.head()

# Now Create a HeatMap using this new DataFram
plt.figure(figsize = (12, 6))
sns.heatmap(dayHour, cmap = 'viridis')

# Create a clustermap using this DataFrame
sns.clustermap(dayHour, cmap = 'viridis')

# repoeat theses same plots and operations. For a DataFrame that shows the Month as the column. 
dayMonth = df.groupby(['Day of Week', 'Month']).count()['Reason'].unstack()
dayMonth.head()

plt.figure(figsize = (12, 6))
sns.heatmap(dayMonth, cmap = 'viridis')
sns.clustermap(dayMonth, cmap = 'viridis')









