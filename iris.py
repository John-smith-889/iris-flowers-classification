
"""
Goal of analysis : 

    Based on attached data build a model which will classify flower 
    species from the last order. Prepare a report where you describe 
    your way of approaching the problem and the steps you took to solve it. 
    Don’t forget to assess the quality of the model you have prepared.

"""
%reset
# Set path
import os
os.getcwd()
os.chdir(r"C:\path")

# import libraries
import pandas as pd
import numpy as np

# Import data
iris_dataset = pd.read_csv("iris.csv", 
                           sep = '|', # also alias "delimiter"
                           # None - names are generated from 0 up
                           index_col = None, # set particular column as index
                           decimal = '.') 

#==================#
# Data preparation #
#==================#

# Review dataset
print(iris_dataset)
iris_dataset.describe(include='all')

# Check information about data frame
iris_dataset.info() 
# Only 149 observation in Sepal.Width - it may be nan 
# recognized problem with Petal.Width column - it should be float, but is object

#=================
# Check nan values

any(iris_dataset.isna())
# there are some, confirmed by checking it in the Spyder GUI mode
# there is also one negative value of length in Sepal.Length column

# Check specific location of the value of nan via indexing
import scipy.sparse as sp
x,y = sp.coo_matrix(iris_dataset.isnull()).nonzero()
print(list(zip(x,y)))
# [(81, 1)]

#==============================================
# Investigate object type of Petal.Width column

# Check column types again
iris_dataset.dtypes
# Check values of "Petal.Width" variable in gui
# row 133 has improper "2,2" value (spoted while modelling)
# look programmatically
iris_dataset.loc[133,['Petal.Width']]
# make correction to 2,2 value (probably should be 2.2)
iris_dataset.loc[133,['Petal.Width']] = 2.2

# Convert column type to float
iris_dataset["Petal.Width"] = iris_dataset['Petal.Width'].astype(float)

# Check column types again
iris_dataset.dtypes
# converted correctly

# During checking values in gui, spotted one negative value, which
# shouldnt be here 

#===========================
# Investigate negative value

# Check min and max values of columns
iris_dataset.describe(include='all')
# minimum value of Sepal.Length column is -4.8

# Get the index of negative value in given column
iris_dataset.index[iris_dataset['Sepal.Length'] < 0].tolist()
# 25

# Check the negative value via indexing
iris_dataset.loc[25,['Sepal.Length']]
# -4.8

# insert nan value in place of negative value
iris_dataset.loc[25,['Sepal.Length']] = np.NaN

#============================
# Insert column means in nans 

# Insert means in previous nan, and previous negative value
iris_dataset.fillna(iris_dataset.mean(), inplace=True)

# Check inserted value
iris_dataset.loc[25,['Sepal.Length']]
# 5.85034
iris_dataset.loc[82,['Sepal.Width']]
# 3.06174
