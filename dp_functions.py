'''
Creation date: 7/6/2022

Author: Farzaneh

Modification history: No modification
'''


#Import libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os 


'''
#Reading data user ids for extrasensory data
def uuid_list(drc):
    uuid_list = []
    for file in os.listdir(drc):
        if file.split('.')[-1] == 'gz':
            uuid_list.append(file.split('.')[0])
    print('We have %d users data in total.' %(len(uuid_list)) )
    return uuid_list
'''




def features(df,opts):
    '''
    Extract features of dataset (excluding labels, timestamps, unique_id_variables and extra_features)
    '''
    features=df.loc[:, ~df.columns.isin(opts.binary_label_variables + opts.category_label_variables+ opts.unique_id_variable+ opts.timestamp_variables +opts.extra_features)]
    return features



def binary_feat(df):
    '''
    Extract binary features of dataset
    '''
    binary_col=[col for col in df if np.isin(df[col].unique(), [0, 1]).all()]
    return binary_col



def inf_(df,opts):
    '''
    General information about the dataset
    '''
    print('─' * 40,'Dimensions of the dataset:',df.shape,sep="\n")
    print('─' * 40,'Number of instance: %d' %features(df,opts).shape[0],' ' * 40,sep="\n")
    print('─' * 40,'Features: %d' %(features(df,opts).shape[1]),' ' * 40,sep="\n")
    print('─' * 40,'Feature matrix : (%d , %d)' % (features(df,opts).shape[0], features(df,opts).shape[1]),' ' * 40,sep="\n")
    print('─' * 40,'Binary labels: %d' % (df[opts.binary_label_variables].shape[1]),' ' * 40,sep="\n")
    print('─' * 40,'Binary label matrix : (%d , %d)' % (df[opts.binary_label_variables].shape[0], df[opts.binary_label_variables].shape[1]),' ' * 40,sep="\n")
    print('─' * 40,'Categorical label matrix : (%d , %d)' % (df[opts.category_label_variables].shape[0], df[opts.category_label_variables].shape[1]),' ' * 40,sep="\n")
    print('─' * 40,'Datatypes of the columns:',' ' * 40, df.dtypes.value_counts(),sep="\n")
    print('─' * 40,'General info of the data:',' ' * 40,sep="\n")
    print(df.info())
    print('─' * 40)
    #print('-'*40, 'How Are The Labels Distributed?',' ' * 40,sep='\n')
    #plt.title('No of Datapoints per Activity', fontsize=15)
    #sns.countplot(df[opts.binary_label_variables])
    #plt.xticks(rotation=90)
    #plt.show()
    print('Showing the first 5 rows of the dataset:','─' * 40,sep="\n")
    return df.head()    



def stat(df):
    '''
    Summary Statistics of the data
    '''
    print('─' * 40,'Summary Statistics for Numerical data:', '─' * 40, df.describe(),'─' * 40,sep="\n")
    #print('Summary Statistics for Categorical data:','─' * 40,df.describe(exclude=[np.number]),sep="\n")
    return



def duplicate(df):
    '''
    Detect and remove duplicates
    '''
    duplicate=df.duplicated().sum()
    if duplicate>0:
        print('*'*20,'There are ',duplicate,' duplicates in data','*'*20)
        print('*'*20, 'Duplicates have been removed','*'*20)
        df=df.drop_duplicates()
        print('*'*20, 'New dataframe (no duplicates) dimension:',df.shape,'*'*20)
    else:
        print('*'*20,'There are not any duplicates in data','*'*20)
    return df
            

    
    

def missing_values(df):
    '''
    Handle missing values + Count missing values per column and total
    '''
    tmv=df.isnull().sum().sum()
    print('Total # of missing values:',tmv)
    print('─'*40,'# Of missing values for each column:','─'*40,df.isnull().sum(),sep='\n')
    print(' '*2)
    if tmv>0:
        print('*'*20,'We need to impute missing values','*'*20)
        print('*'*20,'Select an imputation method from dp_parameters.py','*'*20 )
    else:
        print('*'*20,'There are not any missing values in data','*'*20)
    return tmv



def missing_value_imputers(df,imputer,opts):
    '''
    Missing value impution method selection
    '''
    if imputer == 'Filling zero':
        return fill_zero(df)
    elif imputer == 'Mean imputation':
        return fill_mean(df)
    elif imputer == 'Median imputation':
        return fill_median
    elif imputer == 'Linear interpolation':
        return fill_interpol_lnr(df)


    
def miss_val(df,imputer,opts):
    '''
    Impute missing values if number of them >0
    '''
    if missing_values(df)>0:
        df=missing_value_imputers(df, imputer, opts)
    return df
        


# Methods to impute missing values

def fill_zero(df):
    '''
    Fill missing values by zero
    '''
    df2=df.fillna(0)
    print('*'*20,'The missing values imputation method:     Filling in by Zero ','*'*20)
    return df2   

def fill_mean(df):
    '''
    Fill missing values by mean
    '''
    df2 = pd.DataFrame()
    for column in df.columns:
        df2[column]= df[column].fillna(df[column].mean())
    print('*'*20,'The missing values imputation method:     Filling in by Mean ','*'*20)
    return df2

def fill_median(df):
    '''
    Fill missing values by median
    '''
    df2 = pd.DataFrame()
    for column in df.columns:
        df2[column]= df[column].fillna(df[column].median())
    print('*'*20,'The missing values imputation method:      Filling in by Median ','*'*20)
    return df2

def fill_interpol_lnr(df):
    '''
    Fill missing values by Linear_Interpolation 
    '''
    df2=df.interpolate(method='linear')
    print('*'*20,'The missing values imputation method:       Linear interpolation ','*'*20)
    return df2




def zer_col(df):
    '''
    Detect and remove columns with all zero entries
    '''
    zero_col_nam=df.columns[(df == 0).all()]
    x=df.drop(columns=zero_col_nam)
    print('*'*20,'The number of columns with all zero entries : '  ,  len(zero_col_nam) ,'*'*20)
    print(zero_col_nam.values)
    print('*'*20, 'New dataframe (no zero columns) dimension:',x.shape,'*'*20)
    return x,zero_col_nam




def outliers_IQR(df,q):
    '''
    Detect and remove outliers by IQR method
    '''
    # finding the Quantiles:
    Q1 = df.quantile(q)
    Q3 = df.quantile(1-q)
    # IQR : Inter-Quartile Range
    IQR = Q3 - Q1
    # Lower bound:
    LC = Q1 - (1.5*IQR)
    # Upper bound:
    UC = Q3 + (1.5*IQR)
    # Find count of Outliers
    outliers=df[(df<LC) | (df>UC)].reset_index(drop=True)
    _sum=outliers.count().sum()
    df2 = df[(df>LC) & (df<UC)]
    df3=df2.dropna()
    if _sum>0:
        print(' '*20,'*'*20,'There are', _sum,'outliers in data','*'*20)
        print('max outlier value: ',(outliers.max()),sep='\n')
        print('min outlier value: ',(outliers.min()),sep='\n')
        print(' '*20,'*'*20,'Outliers have been removed ','*'*20)
        #print('*'*20, 'New dataframe (no outliers) dimension:',df3.shape,'*'*20)
        return df3
    else:
        print(' '*20,'*'*20,'There are not any outliers in data','*'*20)
    return df2




def scaling_method(df,opts):
    '''
    Scale features
    '''
    if opts.scaling_method=='Standardization':
        return pd.DataFrame(stnd_scale(df),columns = df.columns)
    elif opts.scaling_method=='Normalization':
        return pd.DataFrame(norm_scale(df),columns = df.columns)
    
    
    
def stnd_scale(df):
    '''
    Scale features method : Standardization 
    '''
    SS = StandardScaler()
    scaled_dataset = SS.fit_transform(df)
    print(' '*20,'*'*20,'The numerical features have been scaled (Standardization )','*'*20)
    print(scaled_dataset  )
    return scaled_dataset              
              

    
def norm_scale(df):
    '''
    Scale features method : Normalization 
    '''
    MS = MinMaxScaler()
    MinMaxScaled = MS.fit_transform(df)
    print(' '*20,'*'*20,'The numerical features have been scaled (Normalization )','*'*20)
    print(MinMaxScaled  )
    return MinMaxScaled              
                
    
    
def cat_encoding(column):
    '''
    Encoding the categorical labels: Label encoding method
    '''
    LE = LabelEncoder()
    enc_activities = LE.fit(column)
    #enc_activities.classes_
    enc_label=enc_activities.transform(column)
    return pd.DataFrame(enc_label)      

def indicator(column):
    '''
    Indicator variables
    '''
    labels=pd.get_dummies(column)
    return labels



def label_converter(column,opts):
    '''
    Scale features
    '''
    if opts.label_converter=='Cat_encoding':
        return cat_encoding(column)
    elif opts.label_converter=='Indicator':
        return indicator(df)
    
    
    
    
    
def evaluat(df):
    '''
    Evaluate the cleaned dataset
    '''
    print('*'*20,'There are ',df.duplicated().sum(),' duplicates in cleaned data','*'*20)
    print('*'*20,'There are ',df.isnull().sum().sum(),' missing values in cleaned data','*'*20)
    print('*'*20,'There are not any outliers in cleaned data','*'*20)
    print('*'*20,'The numerical features have been scaled','*'*20)
              