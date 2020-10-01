# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Step 4: Data Wrangling
#
# The raw output data from coregistration step cannot be passed into machine learning model directly. They don't have labels and have too many features. 
#
# The label is the thing we will try to predict with the machine learning models. In our case, the label is a True or False assertion. Each input data row is either a mineral deposit or not a mineral deposit. Given a data row, the machine learning model will tell us if the location is a mineral deposit or not. This is the ultimate goal we are trying to achieve in this machine learning workflow.
#
# The feature is a column of the input data. There are many features in the coregistration output data. It is not wise to use too many features in the machine learning analysis because of the [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). More features mean more dimensions. If you have 20 features/columns, the machine learning model has to do the analysis in a 20-dimensional space. Just imagine if you were thrown into a 20-dimensional space, I guess it would not be a happy experience to find a way out. Yes, some machine learning models are smart enough and can reduce the number of dimensions. But at the current point of human history, humans are still a little bit smarter than computers. So, let's help computers by reducing the number of features.
#
# We will create a csv file, in which the last colomn is the label(0 or 1) and the other columns are features. 
#
# The feature selection is highly related to the specific research. We need to identify the features which are most important to the formation of mineral deposit. For example, some researchers might think the distance along the trench is important. Others might think the sea floor age is of great significance. Come out with your own hypothesis, wrangle the data accordingly and then send the data into a machine learning model to be evaluated. Repeat this process until we find the most important features. This process is similar to a psychic finding a perfect crystal ball to start a fortune telling business. The only difference is that the psychic is doing magic, but we are doing science.
#
# The following code cell will select features from coregistration output and create a csv file for the machine learning analysis in Step 5.

# +
import numpy as np
import pandas as pd
import Utils
#
# LOOK HERE! 
# This cell must run first to setup the working environment
#

#load the config file
Utils.load_config('config.json')
Utils.print_parameters()

# +
#load data 
coreg_input_data = pd.read_csv(Utils.get_coreg_input_dir() + 'positive_deposits.csv')
coreg_output_data = pd.read_csv(Utils.get_coreg_output_dir() + 'positive_deposits.csv')
print('The shape of coregistration input data is: ', coreg_input_data.shape)
print('The shape of coregistration output data is: ', coreg_output_data.shape)

if coreg_input_data.shape[0] == coreg_output_data.shape[0]:
    print('Good! The input and output data has the same length ', coreg_output_data.shape[0])

print()
print('the coregistration input data')
display(coreg_input_data)
print('the coregistration output data')
display(coreg_output_data)
print('the columns in coregistration output data are: ')
Utils.print_columns()


# -

# ##### After having a look at the data, let's start selecting features and add labels.

# +
import os

coreg_out_dir = Utils.get_coreg_output_dir()
positive_data = pd.read_csv(coreg_out_dir + '/positive_deposits.csv')
negative_data = pd.read_csv(coreg_out_dir + '/negative_deposits.csv')
candidates_data = pd.read_csv(coreg_out_dir + '/deposit_candidates.csv')

print(positive_data.columns)

feature_names = Utils.get_parameter('feature_names')

positive_features = positive_data[feature_names].dropna()
negative_features = negative_data[feature_names].dropna()
candidates_features = candidates_data[feature_names].dropna()

positive_features['label']=True
negative_features['label']=False

#save the data
ml_input_dir = Utils.get_ml_input_dir()
if not os.path.isdir(ml_input_dir):
    os.mkdir(ml_input_dir)
positive_features.to_csv(ml_input_dir + 'positive.csv', index=False)
negative_features.to_csv(ml_input_dir + 'negative.csv', index=False)
candidates_features.to_csv(ml_input_dir + 'candidates.csv', index=False)

positive_data.iloc[positive_features.index].to_csv(ml_input_dir + 'positive_all_columns.csv', index=False)
negative_data.iloc[negative_features.index].to_csv(ml_input_dir + 'negative_all_columns.csv', index=False)
candidates_data.iloc[candidates_features.index].to_csv(ml_input_dir + '/andidates_all_columns.csv', index=False)

import glob
files = glob.glob(ml_input_dir + '*')
print('\ngenerated files:')
for f in files:
    print(f)

# -

# #### The dataset has 5 features(dimensions). It is difficult to visualize the data in high dimensional space. So, let's plot it in two dimensions. 
#
# Sometimes we can see a little bit the pattern of mineral deposit distribution in the 2D plots. Sometimes we cannot. For example, it seems there are many mineral deposits when the seafloor age is between 20 and 110 million years in the plot below. Something to think about, in terms of what this may mean.
#
# The machine learning models can help us to find out the distribution pattern in a higher dimensional space, which might not emerge in a lower dimensional space. Machine learning analysis cannot tell us the exact distribution pattern, but it can give us an estimation with a certain accuracy. Two-dimensional xy-plots are useful for plotting one feature against a second feature of our choice, so explore whether we can detect any patterns in the data, in terms of known mineral deposits versus "non-deposit" locations.

# +
# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12,6),dpi=150)
ax = plt.axes()

coreg_out_dir = Utils.get_coreg_output_dir()
positive = pd.read_csv(coreg_out_dir + 'positive_deposits.csv')
negative = pd.read_csv(coreg_out_dir + 'negative_deposits.csv')

display(positive)

p1=ax.scatter(positive['conv_ortho'], 
           positive['conv_paral'],
           20, marker='.',c='red')
p2=ax.scatter(negative['conv_ortho'], 
           negative['conv_paral'], 
           20, marker='.',c='blue')

ax.set_xlim(-5, 20)
ax.set_ylim(-10, 10)
plt.xlabel('Convergence Orthogonal Velocity (cm/yr)')
plt.ylabel('Convergence Parallel Velocity (cm/yr)')
plt.title('Deposits (in red) and non-deposits (in blue)')
ax.legend([p2,p1],["Non-Deposit","Deposit"],
            loc=4, borderaxespad=0.3,numpoints=1)
plt.grid()
plt.show()


# +
# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12,6),dpi=150)
ax = plt.axes()

coreg_out_dir = Utils.get_coreg_output_dir()
positive = pd.read_csv(coreg_out_dir + 'positive_deposits.csv')
negative = pd.read_csv(coreg_out_dir + 'negative_deposits.csv')

p1=ax.scatter(positive['age'], 
           positive['dist_from_start']* 6371. * np.pi / 180,
           50, marker='.',c='red')
p2=ax.scatter(negative['age'], 
           negative['dist_from_start']* 6371. * np.pi / 180, 
           50, marker='.',c='blue')
ax.set_xticks(np.arange(0,231,10))
#ax.set_yticks(np.arange(0,8000,500))
ax.set_xlim(-5, 110)
ax.set_ylim(-100, 6500)
plt.xlabel('Age(Ma)')
plt.ylabel('Distance Along Trench (km)')
plt.title('Deposits (in red) and non-deposits (in blue)')
ax.legend([p2,p1],["Non-Deposit","Deposit"],
            loc=4, borderaxespad=0.3,numpoints=1)
plt.grid()
plt.show()

# -

# ##### Let's try to plot it in a 3-dimensional space

# +
# %matplotlib inline

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
from scipy import misc, ndimage

coreg_out_dir = Utils.get_coreg_output_dir()
positive = pd.read_csv(coreg_out_dir + 'positive_deposits.csv')
negative = pd.read_csv(coreg_out_dir + 'negative_deposits.csv')
data = pd.read_csv(coreg_out_dir + 'deposit_candidates.csv')

positive.dropna(inplace=True)
negative.dropna(inplace=True)
print(positive.shape, negative.shape)

data.dropna(inplace=True)
ages = data['age'].values.tolist()
dist_along_trench=(data['dist_from_start']* 6371. * np.pi / 180).values.tolist()
dist_nearest_edge=(data['dist_nearest_edge']* 6371. * np.pi / 180).values.tolist()


fig = plt.figure(figsize=(12,6),dpi=150)
ax = plt.axes()
plt.xlim([0,230])
plt.ylim([0,6500])

grid_x, grid_y = np.mgrid[0:231, 0:6501]
grid_data = griddata(list(zip(ages, dist_along_trench)), dist_nearest_edge, 
                     (grid_x, grid_y), method='linear', fill_value=0)
grid_data = ndimage.gaussian_filter(grid_data, sigma=10)
cb=plt.imshow(grid_data.T, extent=(0,230,0, 6500), origin='lower', aspect='auto',cmap=plt.cm.inferno)
#cb=ax.scatter(ages, dist_along_trench, c=dist_nearest_edge,cmap=plt.cm.jet)
p2=ax.scatter(negative['age'], negative['dist_from_start']* 6371. * np.pi / 180, 50, marker='.',c='blue')
p1=ax.scatter(positive['age'], positive['dist_from_start']* 6371. * np.pi / 180, 50, marker='.',c='red')
ax.set_xticks(np.arange(0,240,10))
ax.set_yticks(np.arange(0,6500,500))
plt.xlabel('Age(Ma)')
plt.ylabel('Distance Along Trench (km)')
plt.title('Deposits (in red) and non-deposits (in blue)')
plt.grid()
ax.legend([p2,p1],["Non-Deposit","Deposit"],
            loc=1, borderaxespad=0.3,numpoints=1)
fig.colorbar(cb, shrink=0.5, label='Distance to Nearest Edge (km)')

plt.show()
# -

# From above plot we can see that the mineral deposits seem to be concentrated in the "hot" area between ages 0 to 70 million years ago. However, we cannot predict mineral deposits by using the above plot alone because the accuracy would not be sufficient. Now, let's move to the next step and evaluate some machine learning models. After training the models, we can try to do some predictions on a test dataset, and then try to make predictions for unexplored locations.

# #### Note:
# The construction of a multi-dimensional space (what we have done in this notebook) is important because the deposit distribution might be clearer to see in one space than the other, but we don't know which space is the best one to choose to start with. Choose your features carefully based on your background reading and your hypotheses and iterate towards. And be aware of the [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).

# #### This is the end of step 4 and now open the step 5 notebook -- 5_Machine_Learning.ipynb


