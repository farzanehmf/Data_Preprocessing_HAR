#!/usr/bin/env python
# coding: utf-8

# Data Preprocessing

# **Creation date:** 7/6/2022
# **Author:** Farzaneh
# **Modification history:** No modification



# Importing Libraries
import numpy as np #mathematical operations
import pandas as pd #data manipulation and analysis
import dp_functions as func #importing my data preprocessing functions
from dp_parameters import Options #importing my parameters
from matplotlib import pyplot as plt #data visualization library for 2D and 3D plots
import seaborn as sns #plotting statistical graphics
import os # operating system interfaces


# Importing Datasets
opts=Options()
dataset= pd.read_csv(opts.filepath)


#Exploring the dataset
func.inf_(dataset,opts)


#Checking summary Statistics of the data
func.stat(dataset)


#Handling Duplicates
dataset=func.duplicate(dataset)


#Handling zero entry columns
#Remove zero columns before missing value imputation
dataset , zero_col_nam=func.zer_col(dataset)



#splitting features by type
data_no_label=func.features(dataset,opts) #All features dataframe
columns_no_label=list(data_no_label.columns) # features name 
colnames_numerics= list(data_no_label.select_dtypes(include=np.number).columns.tolist()) # Numeric features name
colnames_binary_only=list(func.binary_feat(data_no_label)) # Binary features name
colnames_float_only= list(set(colnames_numerics )- set(colnames_binary_only)) # Float features name



# labels converter
# Category or Binary Labels
if opts.label_type == 'binary':
    label=dataset.loc[:,opts.binary_label_variables]
else:
    label=dataset.loc[:,opts.category_label_variabels]
    label, enc_classes=func.cat_encoding(label)
    print('Encoded labels:',label, sep='\n')
    print('Encoded Classes:',enc_classes, sep='\n')


#Missing values detection and imputation
#Assumption on paper=When a label was not reported (e.g., NaN) it is a 'negative' example. 
#So missing values of labels will be filled by zero.

# missing values detection and imputation of labels
dataset.loc[:,opts.binary_label_variables]=func.miss_val(dataset.loc[:,opts.binary_label_variables],opts.miss_imputer_lab,opts)


#Missing value detection and imputation of features
dataset.loc[:,columns_no_label]=func.miss_val(dataset.loc[:,columns_no_label],opts.miss_imputer_feat,opts)


#Remove zero columns after missing value imputation
dataset,zero_col_nam=func.zer_col(dataset)


# Updating features name after missing value imputation and removing new zero columns
data_no_label=data_no_label.drop(zero_col_nam.values, inplace=False ,axis=1)
columns_no_label=list(data_no_label.columns) # Updated features name 
colnames_numerics= list(data_no_label.select_dtypes(include=np.number).columns.tolist()) # Updated numeric features name
colnames_binary_only=list(func.binary_feat(data_no_label)) # Updated binary features name
colnames_float_only= list(set(colnames_numerics )- set(colnames_binary_only)) # Updated float features name


#Outliers detection and treatment

# Detectection and treatment of outliers
no_outlier=func.outliers_IQR(dataset.loc[:,colnames_float_only],opts.quantile)
final_dataset=dataset.iloc[(no_outlier.index)]


# Scaling the Numerical Variables
#final_dataset.loc[:,colnames_float_only]=func.scaling_method(final_dataset.loc[:,colnames_float_only],opts)


# Evaluate the cleaned data
func.evaluat(final_dataset)


#Save to CSV file


# Save cleaned data
#final_dataset.to_csv(opts.resultpath, index = False)


