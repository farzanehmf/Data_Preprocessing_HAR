#Data Preprocessing module
- Creation date: 7/6/2022
- Author: Farzaneh
- Modification history: 7/13/2022
---
#### Input: raw time-series data
#### Output: cleaned time-series data

To run:
1. Set all the options and settings in ```dp_parameters.py```
2. Run this command
```
Data_Preprocessing.ipynb
```

---
**The following data preprocessing steps need to be done before implementing the machine learning/ deep learning models on the dataset.**
- Checking summary statistics of the data
- Handling Duplicates
- Handling zero entry columns
- Encoding the categorical labels
- Missing values detection and imputation
- Outliers detection and treatment
- Scaling the Numerical Variables
```diff
In progress:
- Class balance
- Log transformation
```
# Description:

**Assumptions**
1. The user has knowledge about the data variables (labels, features, timestamp, user id,...)
2. When a label was not reported (e.g., NaN) it is a 'negative' example.


#### Checking summary statistics of the data
Summary statistics summarize information about the data. It tells you something about the values in your data set.\
We will check mean, std, min, 25%, 50%, 75%, and max of each column.

#### Handling Duplicates
Duplicates mean the exact same observations repeating themselves, we need to find and remove duplicates.

#### Handling zero entries columns
It is possible to have one or more columns with zero entries in real datasets, we will detect and remove them.

#### Encoding the categorical labels
ML/DL models are based on mathematical models that do not support anything that is not a number. Therefore, we need to express categorical data as numbers.\
- Label Encoding:
  - Mapping categories to numbers → e.g. Walking => 0, Sitting => 1, Standing => 2

#### Missing values detection and imputation
Missing data happens frequently in real datasets.
Options for handling missing data are: 
- Replacing numeric missing values with the mean of the column (when data is normally distributed)
- Replacing numeric missing values with the median of the column (when data is not normally distributed)
- Replacing numeric missing values with zero (when number of missing values are small)
- Replacing numeric missing values with Linear Interpolation :  It means to estimate a missing value by connecting dots in a straight line in increasing order.
```diff
In progress:
- Polynomial Interpolation
- Interpolation through Padding
```
#### Outliers detection and treatment
An Outlier is an observation that lies far from the rest of the observations. That means an outlier is larger or smaller than the remaining values in the set.

- Detecting outliers using the **Inter Quantile Range (IQR)**\
**Note:** data points that lie 1.5 times of IQR above Q3 and below Q1 are outliers.
**Steps:**
   - Calculate the 1st and 3rd quartiles(Q1, Q3)
   - Compute IQR=Q3-Q1
   - Compute lower bound = (Q1–1.5*IQR) and upper bound= (Q3+1.5*IQR)
   - Check for those who fall below the lower bound and above the upper bound and mark them as outliers
```diff
In progress:
- Percentile Method
- Standard Deviation Method
```
#### Scaling the Numerical Variables
Different numerical variables happen in different scales (e.g. Age <0 to 100> and Salary <0 to millions>). Many ML models are based on euclidean distance or weighted linear combination of variables and we don't want one variable to dominate the others just because they are on a different scale.\
Scaling Strategies:
- Normalization Scaling (Min Max Scalar): use when the variable is normally distributed.
- Standardization Scaling (Z-Score): use when data is not normally distributed. When in doubt use standardization scaling.

