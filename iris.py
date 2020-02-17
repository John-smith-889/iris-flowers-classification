
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
