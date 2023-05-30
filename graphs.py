import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read the csv file
df = pd.read_csv('filename2.csv')

# create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# get the first 30 rows
df1 = df.iloc[:30]
# get the next 30 rows
df2 = df.iloc[30:60]
# get the last 30 rows
df3 = df.iloc[60:90]

# set width of bar
barWidth = 0.25

# define the categories
categories = ['Calls to fitness', 'Generation', 'Best Fitness', 'Word Percentage', 'local minimas', 'Best Solution Variability']

# iterate over the categories and create subplots
for i, category in enumerate(categories):
    if category == 'Best Solution Variability':
        # calculate the range of best solution for each type of run
        range_values1 = df1['Best Fitness'].max() - df1['Best Fitness'].min()
        range_values2 = df2['Best Fitness'].max() - df2['Best Fitness'].min()
        range_values3 = df3['Best Fitness'].max() - df3['Best Fitness'].min()

        # create bar plot for best solution variability
        ax = axes[i // 3, i % 3]
        ax.bar(r1, [range_values1, range_values2, range_values3], color=['#7f6d5f', '#557f2d', '#2d7f5e'],
               width=barWidth, edgecolor='white')
        ax.set_xticks(r1)
        ax.set_xticklabels(['Normal', 'Lamarck', 'Darwin'])
        ax.set_title(category)
    else:
        # get the average value for each type of run
        avg_values1 = df1[category].mean()
        avg_values2 = df2[category].mean()
        avg_values3 = df3[category].mean()

        # set position of bars on X axis
        r1 = np.arange(3)
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]

        # create bar plot for each category
        ax = axes[i // 3, i % 3]
        ax.bar(r1, [avg_values1, avg_values2, avg_values3], color=['#7f6d5f', '#557f2d', '#2d7f5e'],
               width=barWidth, edgecolor='white')
        ax.set_xticks(r1)
        ax.set_xticklabels(['Normal', 'Lamarck', 'Darwin'])
        ax.set_title(category)

# Adjust spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()
