<!-- title: Time Series Analysis and Forecasting with Python -->
<!-- omit in toc -->
# 🎯 Time Series Analysis and Forecasting with Python

<!-- omit in toc -->
## Description

We will learn how to use Python for forecasting time series data to predict new future data points.

<!-- https://drive.google.com/drive/folders/1KvmX33sRW5CtPq9fuDIH5oKhbI4FkE8t?usp=sharing -->
<!-- https://drive.google.com/file/d/1jHGUMno2qO9x4ZSZ6Us2KC2cJav1VxId/view -->

<!-- omit in toc -->
## Course Curriculum

<details>
<summary>Course Curriculum</summary>

1. Course Goals
   1. Understand how to use Python to work with Time Series data
   2. Use Pandas and Statsmodels to visualize Time Series data
   3. Be able to use a wide variety of forecasting techniques on Time Series data
2. Set Up and Installation
3. NumPy Basics: Quick section on NumPy basics and how to manipulate data with it
4. Pandas Basics: Pandas is used for data analysis and data exploration. We'll learn how to use this library since its fundamental to handling our data sources
5. Pandas Visualization: Pandas also comes with a lot of built in visualization tools which we will explore to quickly view time series data on a chart
6. Time Series with Pandas: After learning the Pandas Fundamentals we will focus on specialized tools within pandas specifically designed to work with time stamped data
7. Time Series Analysis with Statsmodels: Statsmodels is a statistical library for Python that contains an entire library of time series statistical analysis tools. This section will be an introduction to use Statsmodels for basic time series analysis
8. General Forecasting Models: In this section we'll dive deep into various forecasting models based on ARIMA (AutoRegressive Integrated Moving Averages)
9.  Deep Learning and Prophet: The final sections will show the latest state of the art methods for forecasting, including Recurrent Neural Networks and Facebook's Prophet library

</details>

<!-- omit in toc -->
## Table of Contents

<details>
<summary>Table of Contents</summary>

<!-- toc here -->
- [1. NumPy](#1-numpy)
  - [1.1. Broadcasting](#11-broadcasting)
  - [1.2. Random Numbers](#12-random-numbers)
  - [1.3. Indexing and Selection](#13-indexing-and-selection)
  - [1.4. NumPy Operations](#14-numpy-operations)
- [2. Pandas](#2-pandas)
  - [2.1. Main Features](#21-main-features)
  - [2.2. Series](#22-series)
  - [2.3. DataFrames](#23-dataframes)
  - [2.4. Indexing and Selecting Data](#24-indexing-and-selecting-data)
    - [2.4.1. Boolean Indexing](#241-boolean-indexing)
    - [2.4.2. Useful Methods for Indexing](#242-useful-methods-for-indexing)
  - [2.5. Missing Data with Pandas](#25-missing-data-with-pandas)
  - [2.6. Group By Operations](#26-group-by-operations)
  - [2.7. Common Operations](#27-common-operations)
  - [2.8. Data IO](#28-data-io)
- [3. Data Visualization with Pandas](#3-data-visualization-with-pandas)
  - [3.1. Customizing Plots with Pandas](#31-customizing-plots-with-pandas)
- [4. Time Series with Pandas](#4-time-series-with-pandas)
  - [4.1. Datetime index](#41-datetime-index)
  - [4.2. Time Resampling](#42-time-resampling)
  - [4.3. Time Shifting](#43-time-shifting)
  - [4.4. Windowing Operations](#44-windowing-operations)
  - [4.5. Visualizing Time Series Data](#45-visualizing-time-series-data)
  - [4.6. Notes from Time Series with Pandas Exercise](#46-notes-from-time-series-with-pandas-exercise)
- [5. Time Series Analysis with Statsmodels](#5-time-series-analysis-with-statsmodels)
  - [5.1. Libraries of this section:](#51-libraries-of-this-section)
  - [5.2. ETS Models and Decomposition with ETS](#52-ets-models-and-decomposition-with-ets)
  - [5.3. EWMA Models](#53-ewma-models)
  - [5.4. Holt-Winters Method](#54-holt-winters-method)
- [6. General Forecasting Models](#6-general-forecasting-models)
  - [6.1. Libraries for this section:](#61-libraries-for-this-section)
  - [6.2. Intro to Forecasting Models](#62-intro-to-forecasting-models)
    - [6.2.1. Test Train Split](#621-test-train-split)
    - [6.2.2. Evaluating Predictions](#622-evaluating-predictions)
  - [6.3. Auto-Corr Function (ACF) and Partial Auto-Corr Function (PACF)](#63-auto-corr-function-acf-and-partial-auto-corr-function-pacf)
  - [6.4. ARIMA](#64-arima)
- [7. Misc](#7-misc)
  - [7.1. How to Decompose Time Series Data into Trend and Seasonality](#71-how-to-decompose-time-series-data-into-trend-and-seasonality)

</details>

# 1. NumPy
NumPy is a numerical processing library that can efficiently handle large data sets stored as arrays (see [Glossary](https://numpy.org/doc/stable/glossary.html#term-little-endian)). Later we will learn about Pandas, which is built directly off of the NumPy library. It provides (see [Repo](https://github.com/numpy/numpy)):

- A powerful N-dimensional array object
- Sophisticated (broadcasting) functions
- Useful linear algebra, Fourier transform, and random number capabilities

Example: 

```python
import numpy as np

# Create a 2-D array, set every second element in
# some rows and find max per row:
x = np.arange(15, dtype=np.int64).reshape(3, 5)
x[1:, ::2] = -99
x
# array([[  0,   1,   2,   3,   4],
#        [-99,   6, -99,   8, -99],
#        [-99,  11, -99,  13, -99]])

x.max(axis=1)
# array([ 4,  8, 13])
```

Section Goals

- Creating NumPy Arrays
- Indexing and Selection on Arrays
- General Operations

We transform a nest list into a NumPy 2-dim array (matrix). Example: 

```python
l = [[1,2,3],[4,5,6]]
m = np.array(l)
```

## 1.1. Broadcasting

Note that NumPy broadcasts operations on arrays (see [Documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)). Generally, the smaller array is “broadcast” across the larger array so that they have compatible shapes. 

NumPy operations are usually done on pairs of arrays on an element-by-element basis. Conceptually we can prepend ones, until we have compatible shapes. See [mCoding Video](https://www.youtube.com/watch?v=oG1t3qlzq14). 

It works with plus, minus, times, exponentiation, min/max, and many more element-wise
operations.

## 1.2. Random Numbers

Note: from [Documentation](https://numpy.org/devdocs/reference/random/index.html). 

```python 
# Do this (new version)
from numpy.random import default_rng
rng = default_rng()
vals = rng.standard_normal(10)
more_vals = rng.standard_normal(10)

# instead of this (legacy version)
from numpy import random
vals = random.standard_normal(10)
more_vals = random.standard_normal(10)
```

Create an array of the given shape and populate it with random samples from a uniform distribution: 

```python
# Create a 2-D array and populate it with
# random numbers from a uniform distro
np.random.rand(2,2)
```

This uses a legacy container `RandomState`. The new version looks like this: 

```python
rng = np.random.default_rng()
rng.standard_normal(10)
```

Call `default_rng` to get a new instance of a `Generator`, then call its methods to obtain samples from different distributions. By default, `Generator` uses bits provided by `PCG64` which has better statistical properties than the legacy `MT19937` used in `RandomState`. See [Documentation](https://numpy.org/doc/stable/reference/random/generator.html). 

The seed is an optional parameter. Note that the seed only works in the cell it is passes as a parameter. 

```python
import numpy as np

rng = np.random.default_rng(12345)
# print(rng)
# Generator(PCG64)

rfloat = rng.random()
rfloat
# 0.22733602246716966

type(rfloat)
# <class 'float'>
```

## 1.3. Indexing and Selection

Slices from an array point to the original array. Broadcasting and reassigning will affect the original array. Example:  

```python
arr = rng.integers(1,11,10)
arr
# array([ 5,  4, 10,  4,  1,  5,  8,  2,  5,  2])

# slice of the first 5 elements of array 
s = arr[0:5:]

s[::] = 99
s
# array([99, 99, 99, 99, 99])

arr
# array([99, 99, 99, 99, 99,  5,  8,  2,  5,  2])
```

If this is not the intended behavior, we can copy the array using the `copy` method. 

We can also slice n-dim arrays. Example: 

```python
arr = rng.integers(1,11,10)
arr = arr.reshape(2,5)
arr
# array([[ 7,  5,  4,  3,  6],
#        [ 7, 10,  5,  2,  9]])

arr[1::,1::]
# array([[10,  5,  2,  9]])
```

Conditional selection: we can also broadcast comparisons. Example: 

```python
arr = rng.integers(1,11,10)
arr
# array([7, 8, 1, 4, 8, 9, 5, 9, 9, 4])
bool_arr = arr > 4
bool_arr
# array([ True,  True, False, False,  True,  True,  True,  True,  True, False])
```

We can pass the Boolean array as a filter for the original array. 

```python
arr[bool_arr]
# array([7, 8, 8, 9, 5, 9, 9])

# or more compact
# read as array where array is greater than 4
arr[arr>4]
```

## 1.4. NumPy Operations

We can broadcast operations or choose an axis or axes along which the operations are computer. Example: 

```python
arr = np.arange(9).reshape(3,3)
arr
# array([[0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8]])

# Determine the median along the rows
m_arr = np.median(arr, axis=1).astype('i')
m_arr
# array([1, 4, 7], dtype=int32)
```

Note that we cast the elements of the resulting array to an integer (see [astype method](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)). The data type codes can be found in the [documentation](https://numpy.org/doc/stable/reference/generated/numpy.typename.html). Alternatively we can use the [NumPy data types](https://numpy.org/devdocs/user/basics.types.html). 

# 2. Pandas

This section we'll learn about:

- Series and DataFrames
- Missing Data
- GroupBy
- Operations
- Data I/0 (Input and Output)

Note that the Documentation has a section on [Time series / date functionality](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-series-date-functionality). From its [Repo](https://github.com/pandas-dev/pandas): pandas is a Python package that provides fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python. Additionally, it has the broader goal of becoming the most powerful and flexible open source data analysis / manipulation tool available in any language.

## 2.1. Main Features

- Easy handling of [**missing data**][missing-data] (represented as
  `NaN`, `NA`, or `NaT`) in floating point as well as non-floating point data
- Size mutability: columns can be [**inserted and
  deleted**][insertion-deletion] from DataFrame and higher dimensional
  objects
- Automatic and explicit [**data alignment**][alignment]: objects can
  be explicitly aligned to a set of labels, or the user can simply
  ignore the labels and let `Series`, `DataFrame`, etc. automatically
  align the data for you in computations
- Powerful, flexible [**group by**][groupby] functionality to perform
  split-apply-combine operations on data sets, for both aggregating
  and transforming data
- Make it [**easy to convert**][conversion] ragged,
  differently-indexed data in other Python and NumPy data structures
  into DataFrame objects
- Intelligent label-based [**slicing**][slicing], [**fancy
  indexing**][fancy-indexing], and [**subsetting**][subsetting] of
  large data sets
- Intuitive [**merging**][merging] and [**joining**][joining] data
  sets
- Flexible [**reshaping**][reshape] and [**pivoting**][pivot-table] of
  data sets
- [**Hierarchical**][mi] labeling of axes (possible to have multiple
  labels per tick)
- Robust IO tools for loading data from [**flat files**][flat-files]
  (CSV and delimited), [**Excel files**][excel], [**databases**][db],
  and saving/loading data from the ultrafast [**HDF5 format**][hdfstore]
- [**Time series**][timeseries]-specific functionality: date range
  generation and frequency conversion, moving window statistics,
  date shifting and lagging


 [missing-data]: https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
 [insertion-deletion]: https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#column-selection-addition-deletion
 [alignment]: https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html?highlight=alignment#intro-to-data-structures
 [groupby]: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#group-by-split-apply-combine
 [conversion]: https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe
 [slicing]: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#slicing-ranges
 [fancy-indexing]: https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced
 [subsetting]: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing
 [merging]: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging
 [joining]: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#joining-on-index
 [reshape]: https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html
 [pivot-table]: https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html
 [mi]: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#hierarchical-indexing-multiindex
 [flat-files]: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#csv-text-files
 [excel]: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#excel-files
 [db]: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#sql-queries
 [hdfstore]: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#hdf5-pytables
 [timeseries]: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-series-date-functionality

## 2.2. Series

A series is the basic building block of Pandas. It holds an **array** of information organized by an **(named) Index**.

NumPy array has data with an index: 

| index | data |
| ----- | ---- |
| 0     | 123  |
| 1     | 456  |
| 2     | 789  |

Series is built on top of a NumPy array and can have a named index. 

| index | **named_index** | data |
| ----- | --------------- | ---- |
| 0     | **A**           | 123  |
| 1     | **B**           | 456  |
| 2     | **C**           | 789  |

Note that a pandas Series takes in **data** and an **index**. 

```python
my_series = pd.Series(data=[1,2,3], index=list('ABC'))
```
We can broadcast operations directly on Series based on the index (position), for example: 

```python
s1 = pd.Series(data=[1,2,3], index=list('ABC'))
s2 = pd.Series(data=[4,5,6,7,8], index=list('ABCDE'))
s1 + s2
# A    5.0
# B    7.0
# C    NaN
# D    NaN
# E    NaN
# dtype: float64

s1.iloc[0]
# 1
s1.loc['A']
# 1
```

## 2.3. DataFrames

A DataFrame is simply multiple series that share the same index. Two-dimensional, size-mutable, potentially heterogeneous tabular data. Data structure also contains labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. The primary pandas data structure (see [Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)). Example: 

```python
# Create a 2-dim NumPy array
arr = np.arange(1,10).reshape(3,3)
arr
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# Create DataFrame (2-dim data structure) 
# The rows in the 2-dim array become the rows of the DataFrame
# The index here is integer position based
df = pd.DataFrame(data = arr)
df
#  	0 	1 	2
# 0 	1 	2 	3
# 1 	4 	5 	6
# 2 	7 	8 	9

# The index can also be label based
# We can also named the columns (Series in the DataFrame)
pd.DataFrame(data = d, columns = list('ABC'), index=list('XYZ'))
#     A 	B 	C
# X 	1 	2 	3
# Y 	4 	5 	6
# Z 	7 	8 	9
```

The axis labeling information in pandas objects serves many purposes:

- Identifies data (i.e. provides metadata) using known indicators, important for analysis, visualization, and interactive console display
- Enables automatic and explicit data alignment
- Allows intuitive getting and setting of subsets of the data set

Another example: 

```python
rng = np.random.default_rng(42)
m = rng.standard_normal((5,4))
m

pd.DataFrame(data=m, index=list('ABCDE'), columns=list('WXYZ'))
#        W 	            X 	      Y 	         Z
# A 	0.304717 	-1.039984 	0.750451 	0.940565
# B 	-1.951035 	-1.302180 	0.127840 	-0.316243
# C 	-0.016801 	-0.853044 	0.879398 	0.777792
# D 	0.066031 	1.127241 	0.467509 	-0.859292
# E 	0.368751 	-0.958883 	0.878450 	-0.049926
```

## 2.4. Indexing and Selecting Data

We can index and slice data (label based or integer position based) in the data frame (see [Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)). 

| Object Type | Indexers                           |
| ----------- | ---------------------------------- |
| Series      | s.loc[indexer]                     |
| DataFrame   | df.loc[row_indexer,column_indexer] |

If we're interested in getting the columns, we use the basic indexing with `[]`. 

| Object Type | Selection      | Return Value Type               |
| ----------- | -------------- | ------------------------------- |
| Series      | series[label]  | scalar value                    |
| DataFrame   | frame[colname] | Series corresponding to colname |

```python
# label based slicing
df.loc[::,::]
# integer position based slicing
df.iloc[::,::]

# If we slice the DataFrame, we see that it consists of J Series, J = len(columns)
type(df.loc[::,'W'])
# pandas.core.series.Series

# We can also pass a list or array of labels
df.loc[::,['W','Z']]
```

### 2.4.1. Boolean Indexing

Another common operation is the use of boolean vectors to filter the data. The operators are: `|` for or, `&` for and, and `~` for not. These must be grouped by using parentheses. Example: 


```python
# Change value in cell [A,X] to one
df.loc['A','X'] = 1
# Create filter, return cells where value is greater than zero
cond = df > 0
# Filter DataFrame and drop NA cells
df[crit].dropna(axis='index')
# 	W 	X 	Y 	Z
# A 	0.304717 	1.0 	0.750451 	0.940565
```

From [Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing): you may select rows from a DataFrame using a boolean vector the same length as the DataFrame’s index (for example, something derived from one of the columns of the DataFrame):

```python
crit = df['W'] > 0
df[crit]
# or
df.loc[crit,::]
# 	      W 	         X           Y      	Z
# A 	0.304717 	-1.039984 	0.750451 	0.940565
# D 	0.066031 	1.127241 	0.467509 	-0.859292
# E 	0.368751 	-0.958883 	0.878450 	-0.049926
```

### 2.4.2. Useful Methods for Indexing

We can reset and set the index of the DataFrame. 

```python
rng = np.random.default_rng(42)
data = rng.standard_normal((5,4))
data
# array([[ 0.30471708, -1.03998411,  0.7504512 ,  0.94056472],
#        [-1.95103519, -1.30217951,  0.1278404 , -0.31624259],
#        [-0.01680116, -0.85304393,  0.87939797,  0.77779194],
#        [ 0.0660307 ,  1.12724121,  0.46750934, -0.85929246],
#        [ 0.36875078, -0.9588826 ,  0.8784503 , -0.04992591]])
idx = list('ABCDE')
cols = list('WXYZ')
df = pd.DataFrame(data=data, index=idx, columns=cols)
df
#           W         X         Y         Z
# A  0.304717 -1.039984  0.750451  0.940565
# B -1.951035 -1.302180  0.127840 -0.316243
# C -0.016801 -0.853044  0.879398  0.777792
# D  0.066031  1.127241  0.467509 -0.859292
# E  0.368751 -0.958883  0.878450 -0.049926

# Reset index to default integer position based index
df.reset_index(inplace=True)
# Rename old index
df.rename({'index':'old_index'},axis=1,inplace=True)
# Create data for the new index
new_index = 'AA BB CC DD EE'.split()
# Assign data to new column
df['new_index'] = new_index
# Set new index
df.set_index('new_index',inplace=True)
df
#           old_index         W         X         Y         Z
# new_index                                                  
# AA                A  0.304717 -1.039984  0.750451  0.940565
# BB                B -1.951035 -1.302180  0.127840 -0.316243
# CC                C -0.016801 -0.853044  0.879398  0.777792
# DD                D  0.066031  1.127241  0.467509 -0.859292
# EE                E  0.368751 -0.958883  0.878450 -0.049926
```

## 2.5. Missing Data with Pandas

We have three options: 

- Keep the missing data (NaN), if the forecasting method can handle it
- Drop the missing data (the entire row including the timestamp
- Fill in the missing data with some value (best estimated guess)

```python
series_dict = dict(A=[1,2,np.nan],B=[5,np.nan,np.nan],C=[1,2,3])
df = pd.DataFrame(series_dict)
df
#      A    B  C
# 0  1.0  5.0  1
# 1  2.0  NaN  2
# 2  NaN  NaN  3
df.dropna()
#      A    B  C
# 0  1.0  5.0  1
df.dropna(axis=1)
#    C
# 0  1
# 1  2
# 2  3

# Drop if the row has two or more NAs
df.dropna(thresh=2)
#      A    B  C
# 0  1.0  5.0  1
# 1  2.0  NaN  2

df.fillna('fill_value')
#             A           B  C
# 0         1.0         5.0  1
# 1         2.0  fill_value  2
# 2  fill_value  fill_value  3

df.fillna(method='pad')
#      A    B  C
# 0  1.0  5.0  1
# 1  2.0  5.0  2
# 2  2.0  5.0  3
```

## 2.6. Group By Operations

Often you may want to perform an analysis based off the value of a specific column, meaning you want to group together other columns based off another. In order to do this, we to perform 3 steps: 

1. Split / Group 
2. Apply
3. Combine

Pandas will automatically make the grouped by column the index of the new resulting DataFrame. Example: 


```python
data = dict(Company='GOOG GOOG MSFT MSFT FB FB'.split(),
            Person='Sam Charlie Amy Vanessa Carl Sarah'.split(),
            Sales=[200, 120, 340, 124, 243, 350])
data
# {'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
#  'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
#  'Sales': [200, 120, 340, 124, 243, 350]}
df = pd.DataFrame(data)
df
#   Company   Person  Sales
# 0    GOOG      Sam    200
# 1    GOOG  Charlie    120
# 2    MSFT      Amy    340
# 3    MSFT  Vanessa    124
# 4      FB     Carl    243
# 5      FB    Sarah    350

# Split the DataFrame
df.groupby('Company')
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fe9c704f7d0>

# Apply aggregate method or function call and combine
df.groupby('Company').describe()
#         Sales                                                        
#         count   mean         std    min     25%    50%     75%    max
# Company                                                              
# FB        2.0  296.5   75.660426  243.0  269.75  296.5  323.25  350.0
# GOOG      2.0  160.0   56.568542  120.0  140.00  160.0  180.00  200.0
# MSFT      2.0  232.0  152.735065  124.0  178.00  232.0  286.00  340.0

# We can also sort by a multi-index column
df.groupby('Company').describe().sort_values([('Sales','mean')])
#         Sales                                                        
#         count   mean         std    min     25%    50%     75%    max
# Company                                                              
# GOOG      2.0  160.0   56.568542  120.0  140.00  160.0  180.00  200.0
# MSFT      2.0  232.0  152.735065  124.0  178.00  232.0  286.00  340.0
# FB        2.0  296.5   75.660426  243.0  269.75  296.5  323.25  350.0

# Custom describe method with range
def describe2(df):
    df_res = df.describe()
    df_res.loc['range'] = df_res.loc['max'] - df_res.loc['min']
    return df_res
```

## 2.7. Common Operations

Methods: 

- unique
- value_counts(normalize=False)
- apply 
- drop
- info
- describe
- sort_values

Attributes: 

- index 
- columns

## 2.8. Data IO

See [Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html). Example: 

```python
l = pd.read_html(
    'https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/'
)
len(l)
# 1 
df = l[0]
df.head(2)
#                 Bank NameBank           CityCity StateSt  CertCert  \
# 0           Almena State Bank             Almena      KS     15426   
# 1  First City Bank of Florida  Fort Walton Beach      FL     16748   

#      Acquiring InstitutionAI Closing DateClosing  FundFund  
# 0                Equity Bank    October 23, 2020     10538  
# 1  United Fidelity Bank, fsb    October 16, 2020     10537  
```

# 3. Data Visualization with Pandas

See [Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#other-plots). 

**Histogram** of **dataframe** or **column**: 

- x-axis = value of num_observation along column
- y-axis = its frequency in dataframe
- color = columns of dataframe (note that a KDE might be a better choice to compare distributions)

```python
df1['A'].plot.hist()
# We can customize the histogram, for example: 
df1['A'].plot.hist(bins=20, edgecolor='k', grid=True).autoscale(enable=True,
                                                                axis='both',
                                                                tight=True)
```

**Bar Plot** of **dataframe**: 

- x-axis = index (treated as a cat_observation)
- y-axis = value of num_observation along column
- color = columns of dataframe

We can also stack the bars with the stacked param. 

```python
df2.plot.bar(stacked=True,grid=True)
```

**Line Plot** of **dataframe** or **column**: 

- x-axis = index (treated as a num_observation)
- y-axis = value of num_observation along column
- color = columns of dataframe

Information is displayed as a series of data points called 'markers' connected by straight line segments. 

```python
df2.plot.line()
# Customize
df2.plot.line(y='a', figsize=(10, 4), grid=True, lw=3).autoscale(enable=True,
                                                                 axis='both',
                                                                 tight=True)
```

**Area Plot** of **dataframe** or **column**: 

- x-axis = index
- y-axis = stacked num_observations
- color = columns of dataframe

Cumme stacked line plots. 

```python
# Cumme line plots; index is the x-axis of the plot
df2.plot.area(figsize=(10, 4), grid=True, alpha=0.5).autoscale(axis='both',
                                                               tight=True)
# Overlapping area plots of singular columns; not very useful  
df2.plot.area(figsize=(10, 4), grid=True, alpha=0.5,
              stacked=False).autoscale(enable=True, axis='both', tight=True)
```

**Scatter Plot** of **two num_columns of a dataframe** (as param): We visualize the following data for each marker / record (along 0-axis): 

- x-axis = num_observation_1
- y-axis = num_observation_2
- color = num_observation_3 
- size = num_observation_4

```python
df1.plot.scatter(x='A',
                 y='B',
                 s=df1['C'] * 50,
                 c='D',
                 alpha=.5,
                 cmap='coolwarm')
```

**Box Plot** of **dataframe** of **column**: Graphical representation of a Five-number summary with some additional information: 

- x-axis = num_column
- y-axis = statistics of num_observations along the column

If we group by a cat_column then we get: 

- box plot for each column 
- x-axis = value of cat_observation in group-by-column
- y-axis = statistics of num_observation along the column of the group

Note: The [Five-number summary](https://en.wikipedia.org/wiki/Five-number_summary) consists of: 

- (1) sample minimum 
- (2) lower / first quartile 
- (3) median / second quartile 
- (4) upper / third quartile 
- (5) sample maximum
- IQR = Q_3 - Q_1
- the whiskers
- outliers

```python
df2.head(3)
#           a         b         c         d  e
# 0  0.039762  0.218517  0.103423  0.957904  x
# 1  0.937288  0.041567  0.899125  0.977680  y
# 2  0.780504  0.008948  0.557808  0.797510  x

print(df2['e'].unique())
# ['x' 'y' 'z']

# Basic boxplot
df2.boxplot()

# We can group by values of a categorical column
df2.boxplot(by='e')
```

Example from the documentation: 

```python
# Instantiate RNG with seed
rng = np.random.default_rng(42)
# Create dataframe with two num_columns
df = pd.DataFrame(rng.standard_normal((10,2)),columns=['Col1', 'Col2'])
# Add a third cat_column
df['Col3'] = pd.Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'])
df
#        Col1      Col2 Col3
# 0  0.304717 -1.039984    A
# 1  0.750451  0.940565    A
# 2 -1.951035 -1.302180    A
# 3  0.127840 -0.316243    A
# 4 -0.016801 -0.853044    A
# 5  0.879398  0.777792    B
# 6  0.066031  1.127241    B
# 7  0.467509 -0.859292    B
# 8  0.368751 -0.958883  NaN
# 9  0.878450 -0.049926  NaN

boxplot = df.boxplot(by='Col3', figsize=(10, 4))
```
The result is two subplots (one for each num_column) with two box plots each (one for cat_observation value in the cat_column) with statistics from each group (Col1, CatA & Col1, CatB + Col2, CatA & Col2, CatB)

**KDE** of **dataframe** of **column**:

- x-axis = value of num_observation along column
- y-axis = approx. of PDF for num_column
- color = columns of dataframe

```python
df2.plot.kde()
```

**Hexbin** of  

Motivation: in a scatter plot with too many points, we cannot see how many are stacked on top of each other. Hexbin plots can be a useful alternative to scatter plots if your data are too dense to plot each point individually.

- x-axis = num_observation_1
- y-axis = num_observation_2

```python
# Instantiate RNG with seed
rng = np.random.default_rng(42)
df = pd.DataFrame(data=rng.standard_normal((1_000, 2)), columns=list('ab'))
df.head(3)
#           a         b
# 0  0.304717 -1.039984
# 1  0.750451  0.940565
# 2 -1.951035 -1.302180

df.plot.hexbin(x='a',y='b',gridsize=25,cmap='Oranges')
```

## 3.1. Customizing Plots with Pandas

```python
ax = df2[list('ac')].plot(figsize=(10, 3), ls=':', lw=3, title='My titel')
# c = 'blue'
ax.set(xlabel='My x label', ylabel='My y label')
ax.legend(bbox_to_anchor=(1, 1))
# loc = 0
```

# 4. Time Series with Pandas

In this section we will discuss:

- Date Time Index Basics
- Time Resampling
- Time shifting
- Rolling and Expanding
- Time Series Visualization
- Time Series Project Exercise

## 4.1. Datetime index

We start by creating datetime objects: 

```python
from datetime import datetime
datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])
```

NumPy handles dates more efficiently than Python's datetime format. See [Documentation](https://numpy.org/doc/stable/reference/arrays.datetime.html). 

```python
# This is treated as a normal array
np.array('2020-03-15 2020-03-16 2020-03-17'.split())
# array(['2020-03-15', '2020-03-16', '2020-03-17'], dtype='<U10')

# This is an array of datetime64 objects with day [D] precision
np.array('2020-03-15 2020-03-16 2020-03-17'.split(), dtype='datetime64')
#array(['2020-03-15', '2020-03-16', '2020-03-17'], dtype='datetime64[D]')

# We can also create a range of datetime objects
np.arange('2018-06-01', '2018-06-23', 7, dtype='datetime64[D]')
# array(['2018-06-01', '2018-06-08', '2018-06-15', '2018-06-22'],
#       dtype='datetime64[D]')
```

Pandas also has methods to work with datetime data. See [Documentation](https://pandas.pydata.org/docs/reference/api/pandas.date_range.html). Pandas has a specialized DateTimeIndex to work with datetime objects. 

```python
pd.to_datetime(['1/2/2018', '1/3/2018'])
# DatetimeIndex(['2018-01-02', '2018-01-03'], dtype='datetime64[ns]', freq=None)

pd.to_datetime(['2/1/2018', '3/1/2018'])
# DatetimeIndex(['2018-02-01', '2018-03-01'], dtype='datetime64[ns]', freq=None)

# We can specify the format to bypass Pandas inferring it it 
pd.to_datetime(['2/1/2018', '3/1/2018'], format='%d/%m/%Y')
# DatetimeIndex(['2018-01-02', '2018-01-03'], dtype='datetime64[ns]', freq=None)

pd.date_range('2020-01-01', periods=7, freq='D')
# DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
#                '2020-01-05', '2020-01-06', '2020-01-07'],
#               dtype='datetime64[ns]', freq='D')
```

Note that we use `arange` to specify the step size and type, and `date_range` to specify the number of elements (periods) and frequency (type). 

```python
data = rng.standard_normal((3, 2))
cols = list('AB')
idx = pd.date_range('2020-01-01', periods=3, freq='D')

df = pd.DataFrame(data, index=idx, columns=cols)
df
#                    A         B
# 2020-01-01  0.066031  1.127241
# 2020-01-02  0.467509 -0.859292
# 2020-01-03  0.368751 -0.958883

df.index
# DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[ns]', freq='D')
df.index.min()
# Timestamp('2020-01-03 00:00:00', freq='D')
df.index.argmin()
# 0
```

## 4.2. Time Resampling

Resampling is a convenience method for frequency conversion and resampling of time series.

Similar to a group operation but based on time frequency. See [Time Series / Date Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html) and [thread](https://stackoverflow.com/questions/17001389/pandas-resample-documentation). Also see [Offset Alias](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases), [Anchored Offsets](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets), and [Resampling](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling). Refer to the [Resample Page](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html). Some aliases can be found in the course material. 

| Alias    | Description                                      |
| -------- | ------------------------------------------------ |
| B        | business day frequency                           |
| C        | custom business day frequency                    |
| D        | calendar day frequency                           |
| W        | weekly frequency                                 |
| M        | month end frequency                              |
| SM       | semi-month end frequency (15th and end of month) |
| BM       | business month end frequency                     |
| CBM      | custom business month end frequency              |
| MS       | month start frequency                            |
| SMS      | semi-month start frequency (1st and 15th)        |
| BMS      | business month start frequency                   |
| CBMS     | custom business month start frequency            |
| Q        | quarter end frequency                            |
| BQ       | business quarter end frequency                   |
| QS       | quarter start frequency                          |
| BQS      | business quarter start frequency                 |
| A, Y     | year end frequency                               |
| BA, BY   | business year end frequency                      |
| AS, YS   | year start frequency                             |
| BAS, BYS | business year start frequency                    |
| BH       | business hour frequency                          |
| H        | hourly frequency                                 |
| T, min   | minutely frequency                               |
| S        | secondly frequency                               |
| L, ms    | milliseconds                                     |
| U, us    | microseconds                                     |
| N        | nanoseconds                                      |

```python
# Instantiate the dataframe with a named index
df = pd.read_csv(file, index_col='Date', parse_dates=True)
# If we do not pass the parse_dates parameter, use the following.
# df.index = pd.to_datetime(df.index)

# daily -> yearly mean
df.resample(rule='A').mean()
#                 Close        Volume
# Date                               
# 2015-12-31  50.078100  8.649190e+06
# 2016-12-31  53.891732  9.300633e+06
# 2017-12-31  55.457310  9.296078e+06
# 2018-12-31  56.870005  1.122883e+07

# daily close price -> mean yearly close price
df['Close'].resample('A').mean()
# Date
# 2015-12-31    50.078100
# 2016-12-31    53.891732
# 2017-12-31    55.457310
# 2018-12-31    56.870005
# Freq: A-DEC, Name: Close, dtype: float64
df['Close'].resample('A').mean().plot.bar()
```

We use `matplotlib.pyplot` when we want to customize the plot. See [Documentation](https://matplotlib.org/stable/api/pyplot_summary.html); [Bar Plot Page](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html). Pyplot also offers a [Tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py). Also see this [thread](https://stackoverflow.com/questions/14946371/editing-the-date-formatting-of-x-axis-tick-labels-in-matplotlib). 

```python
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

# Create x- and y-arrays
idx = df.resample(rule='A').mean().index.year
y = df['Close'].resample('A').mean().values

# Create plot
ax = plt.bar(idx,y,tick_label=idx)
```

## 4.3. Time Shifting

We can shift the data along the 0-axis. See [Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html).  

```python
# Empty data point in the first row and lose last data point
df.shift(1)

# Shift data points to end of the month
df.shift(periods=1, freq='M')
```

## 4.4. Windowing Operations

See [Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html). A common process with time series is to create data based on a [rolling mean](https://en.wikipedia.org/wiki/Moving_average) -> we divide data into windows of time -> apply aggregate function for each window. 

The `rolling` method behaves like `rows between n preceding and current row` in postgres. 

```python
# Plot of the closing price and its 30 moving average
df.Close.plot(figsize=(12, 5))
df.Close.rolling(window=30).mean().plot()
```

To add a legend to the plot, we can add the MA(30) column to the dataframe. 

```python
df['MA(30)'] = df.Close.rolling(window=30).mean()
df.iloc[29:34:]
#               Close   Volume     MA(30)
# Date                                   
# 2015-02-13  42.8942  6109522  39.835057
# 2015-02-17  43.1050  6386900  40.005020
# 2015-02-18  43.5593  6541986  40.214393
# 2015-02-19  43.6390  6109176  40.436533
# 2015-02-20  43.7982  6462662  40.633647
df['Close MA(30)'.split()].plot(figsize=(12, 5))
```

The `expanding` method behaves like `rows between unbounded preceding and current row` in postgres. 

```python
df.Close.expanding().mean().head()
# Date
# 2015-01-02    38.006100
# 2015-01-05    37.642100
# 2015-01-06    37.419667
# 2015-01-07    37.535950
# 2015-01-08    37.727980
# Name: Close, dtype: float64
```

## 4.5. Visualizing Time Series Data

See [Anatomy of a figure](https://matplotlib.org/stable/gallery/showcase/anatomy.html) for more details. See notebook for reference. See [plt dates Documentation](https://matplotlib.org/stable/api/dates_api.html#date-tickers). 

See also [strftime() and strptime() Behavior](https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior). 

```python
# Columns have wildly different scales -> Plot the curves separately
df.plot()

# Creaate plot and set title
ax = df.Close.plot(figsize=(12, 6), title='Close Price vs Date', grid=True)
# Scale the graph
ax.autoscale(axis='both', tight=True)
# Set labels
ax.set(xlabel='Date', ylabel='Close Price')

# Filter the data and plot
# Alternatively, pass the xlim and ylim param in the plot method
ax = df.Close.loc['2017-01-01':'2018-12-31'].plot(figsize=(12, 4), grid=True)
ax.autoscale(axis='both', tight=True)
```

We can also adjust the tick label: 

```python
# import dates module from plt
from matplotlib import dates
# Create subplot
ax = df.Close.loc['2017-01-01':'2017-03-01'].plot(figsize=(12, 4), grid=True)
# Scale subplot
ax.autoscale(axis='x',tight=True)

# Remove x-label
ax.set(xlabel='')

# Set location of tick labels
ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
# Set format of tick labels
ax.xaxis.set_major_formatter(dates.DateFormatter('%B %d (%a)'))
```

We can also adjust the minor tick label and the grid for each axis: 

```python
# Create subplot
ax = df.Close.loc['2017-01-01':'2017-03-01'].plot(figsize=(12, 4))
# Scale subplot
ax.autoscale(axis='x',tight=True)

# Remove x-label
ax.set(xlabel='')

# Set location of tick labels
ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
# Set format of tick labels
ax.xaxis.set_major_formatter(dates.DateFormatter('%d'))

# Set location of minor tick labels
ax.xaxis.set_minor_locator(dates.MonthLocator())
# Set format of minor tick labels
ax.xaxis.set_minor_formatter(dates.DateFormatter('\n\n%b'));

# Show grid
ax.xaxis.grid(True)
```


## 4.6. Notes from Time Series with Pandas Exercise

Libaries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
# from matplotlib.dates import DateFormatter
# from matplotlib import dates
# import matplotlib.dates as mdates
```

[Timedeltas Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html). 

# 5. Time Series Analysis with Statsmodels

## 5.1. Libraries of this section: 

```python
# General
import numpy as np
import pandas as pd
# from matplotlib import dates
import matplotlib.pyplot as plt
from datetime import datetime
import pylab
import seaborn as sns
# import statsmodels.api as sm
sns.set_style("whitegrid")
pylab.rc("figure", figsize=(16, 8))
pylab.rc("font", size=14)
```

```python
# Forecasting Models
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
```


See [Forecasting: Principles and Practice](https://otexts.com/fpp2/) by Rob J Hyndman and George Athanasopoulos. 

[Statsmodels Documentation](https://www.statsmodels.org/stable/index.html). [Statsmodels TSA Documentatio](https://www.statsmodels.org/stable/tsa.html)n. See also: [How to Decompose Time Series Data into Trend and Seasonality](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/). See also the [TSA Tutorial](https://thequackdaddy.github.io/statsmodels.github.io/devel/examples/notebooks/generated/exponential_smoothing.html). 

Section Goals: 

- Introduction to Statsmodels
- ETS Decomposition
- Moving Averages
- Holt Winters Methods
- Statsmodels Time Series Exercises

First we'll learn how to call a function test from Stasmodels and about the Hodrick-Prescott filter. **Time series data** has particular **properties**: 

- Trends:
  - Upward: slope is positive in general (if you would average it out)
  - Horizontal or Stationary
  - Downward: 
- Seasonality: Repeating trends
- Cyclical: Trends with no set repetition

The Hodrick-Prescott filter separates a time-series `y` into a trend component `tau` and a cyclical component `c`: 

```tex
y_t = \tau_t + c_t
```

The components are determined by minimizing the following quadratic loss / cost / error function, where lambda is a smoothing parameter:

```tex
\min _{\tau_{t}} \sum_{t=1}^{T} c_{t}^{2}+\lambda \sum_{t=1}^{T}\left[\left(\tau_{t}-\tau_{t-1}\right)-\left(\tau_{t-1}-\tau_{t-2}\right)\right]^{2}
```

The lambda value above handles variations in the growth rate of the trend component. 

Recommended lambda values for Hodrick-Prescott filter: 

- **Quarterly**: 1,600
- **Anually**: 6.25
- **Monthly**: 129,600 

## 5.2. ETS Models and Decomposition with ETS

**ETS Models** (Error-Trend-Seasonality). This general term stands for a variety of different models, e.g.: 

- Exponential Smoothing
- Trend Methods Models
- **ETS Decomposition**

Statsmodels provides a seasonal decomposition tool we can use to separate out the different components. We've already seen a simplistic example of this in the Introduction to Statsmodels section with the Hodrick-Prescott filter.

ETS (Error-Trend-Seasonality Models will take each of those terms for "smoothing" and may add them, multiply them, or even just leave some of them out. Based off these key factors, we can try to create a model to fit our data.

Visualizing the data based off its ETS is a good way to build an understanding of its behaviour.

There are two types of ETS models: 

- Additive model
- Multiplicative model 

We apply an **additive model** when it seems that the **trend** is more **linear** and the **seasonality** and trend components seem to be **constant** over time (e.g. every year we add 10,000 passengers).

A **multiplicative model** is more appropriate when we are increasing (or decreasing) at a **non-linear rate** (e.g. each year we double the amount of passengers).

## 5.3. EWMA Models

See [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html#pandas.DataFrame.ewm). See [Statsmodels Documentation](https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html). EWMA (Exponentially weighted moving average) models. We've previously seen how calculating simple moving averages can allow us to create a simple model that describes some trend level behavior of a time series. 

We could expand off the idea of an SMA (simple moving average) by utilizing a EWMA (Exponentially weighted moving average). As issue with SMA is that the entire model will be constrained to the same window size.

It would be nice if we could have more recent data be weighted more than older data. We do this by implementing a EWMA instead of SMA. Basic SMA has some cons:

- Smaller windows will lead to more noise, rather than signal
- It will always lag by the size of the window
- It will never reach to full peak or valley of the data due to the averaging.
- Does not really inform you about possible future behavior, all it really does is describe trends in your data.
- Extreme historical values can skew your SMA significantly


To help fix some of these issues, we can use an EWMA. EWMA will allow us to reduce the lag effect from SMA and it will put more weight on values that occurred more recently (by applying more weight to the more recent values, thus the name).

The amount of weight applied to the most recent values will depend on the actual parameters used in the EWMA and the number of periods given a window size. In general a weighted moving average is described by the following formula: 

```tex
y_t =   \frac{\sum\limits_{i=0}^t w_i x_{t-i}}{\sum\limits_{i=0}^t w_i}
```

We now need to define the weight function `w_i`. When the param `adjust = True` then: 

```tex
y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
+ (1 - \alpha)^t x_{0}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
+ (1 - \alpha)^t}
```

Otherwise: 

```tex
\begin{split}y_0 &= x_0 \\
y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,\end{split}
```

```tex
\begin{split}
  w_i = \begin{cases}
    \alpha (1 - \alpha)^i & \text{if } i < t \\
    (1 - \alpha)^i        & \text{if } i = t.
  \end{cases}
\end{split}
```

To adjust alpha we have three different params: 

```tex
\begin{split}\alpha =
 \begin{cases}
     \frac{2}{s + 1},               & \text{for span}\ s \geq 1\\
     \frac{1}{1 + c},               & \text{for center of mass}\ c \geq 0\\
     1 - \exp^{\frac{\log 0.5}{h}}, & \text{for half-life}\ h > 0
 \end{cases}
\end{split}
```

## 5.4. Holt-Winters Method

Previously with EWMA we applied Simple Exponential Smoothing using just one smoothing factor alpha. This failed to account for other contributing factors like trend and seasonality.

The Holt-Winters seasonal method comprises of the forecast equation and three smoothing equations.

One for the level `l_t`, one for the trend `b_t`, and one for the seasonal component `s_t`, with corresponding smoothing parameters alpha, beta, and gamma.

There are two variations to this method that differ in the nature of the seasonal component: the **additive method** is preferred when the seasonal variations are roughly constant through the series, while the **multiplicative method** is preferred when the seasonal variations are changing proportional to the level of the series.

We start with a **Single Exponential Smoothing**: 

```tex
y_0 = X_0
y_t = (1-\alpha)y_{t-1} + \alpha x_t
```

We can expand this to a **Double Exponential Smoothing**. In Double Exponential Smoothing (aka Holt's Method) we introduc a new smoothing factor beta that addresses trend:

```tex
\text{level: } l_t =(1-\alpha)l_{t-1}1+\alpha x,,
\text{trend: } b_t = (1-\beta)b_{t-1}+ \beta(l_t-t_{t-1})
\text{fitted model: } y_t = l_t + b_t
\text{forecasting model: } \hat{y}_{t+h} = l_t + hb_t,~ h \text{ is the periods into the future}
```

In this method we do not take into account a seasonal component. With Triple Exponential Smoothing (aka the Holt-Winters Method) we introduce a smoothing factor gamma that addresses seasonality:

```tex
\begin{aligned}
\text{level: }l_{t} &=(1-\alpha) l_{t-1}+\alpha x_{t}, \\
\text{trend: }b_{t} &=(1-\beta) b_{t-1}+\beta\left(l_{t}-l_{t-1}\right) \\
\text{seasonal: }c_{t} &=(1-\gamma) c_{t-L}+\gamma\left(x_{t}-l_{t-1}-b_{t-1}\right) \\
\text{fitted model: }y_{t} &=\left(l_{t}+b_{t}\right) c_{t} \\
\text{forecasting model: }\hat{y}_{t+m} &=\left(l_{t}+m b_{t}\right) c_{t-L+1+(m-1) \bmod L},~m \text{ is the periods into the future}
\end{aligned}
```

Here L represents the number of divisions per cycle. In our case looking at monthly data that displays a repeating pattern each year, we would use L=12. 

We need to set the frequency of the index to use this method. See [Offset Aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html). See also this [github thread](https://github.com/unit8co/darts/issues/241) if the methods does not converge. 

# 6. General Forecasting Models

## 6.1. Libraries for this section: 

```python
# Forecasting Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Metrics: Evaluating Predicitions 
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Autocorrelation
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
# import warnings
# warnings.filterwarnings('ignore')
# Lag Plot
from pandas.plotting import lag_plot
```

Section Overview: 

- Introduction to Forecasting
- ACF and PACF plots
- Auto Regression AR
- Descriptive Statistics and Tests
- Choosing ARIMA orders
- ARIMA based models

Standard Forecasting Procedure: 

- Choose a **Model**
- **Split** data into train and test sets
- **Fit** model on training set
- **Evaluate** model on test set
- **Re-fit** model on entire data set
- **Forecast** for future data

## 6.2. Intro to Forecasting Models


### 6.2.1. Test Train Split

Test sets will be the most recent end of the data (see [sklearn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html))

The size of the test set is typically about 20% of the total sample, although this value depends on how long the sample is and how far ahead you want to forecast. **The test set should ideally be at least as large as the maximum forecast horizon required**. Keep in mind, the longer the forecast horizon, the more likely your prediction becomes less accurate.

### 6.2.2. Evaluating Predictions

After we fit model on the training data, we forecast to match up to the test data dates. Then we can compare our results for evaluation.

You may have heard of some evaluation metrics like accuracy or recall. These sort of metrics aren't useful for time series forecasting problems, we need metrics designed for **continuous** values!

Let's discuss some of the most common evaluation metrics for regression:

- Mean Absolute Error (**MAE**)
- Mean Squared Error (**MSE**)
- Root Mean Square Error (**RMSE**)

Whenever we perform a forecast for a continuous value on a test set, we have two values:

- `y` the value of the observation (in the test data)
- `y_hat` the predicted value from our forecast

Mean Absolute Error (MAE) This is the mean of the absolute value of errors: Easy to understand. 

```tex
\frac{1}{n} \sum_{i=1}^{n}\left|y_{i}-\hat{y}_{i}\right|
```

An issue with MAE though, is that simply averaging the residuals won't alert us if the forecast was really off fo a few points. We want to be aware of any prediction errors that are very large (even if there only a few). 

Mean Squared Error (MSE): This is the mean of the squared errors. Larger errors are noted more than with MAE making MSE more popular.

```tex
\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}
```

There is an issue with MSE however! Because we squared the residual, the units are now also squared which is hard to interpret!

Root Mean Square Error (RMSE): This is the root of the mean of the squared errors. Most popular (has same units as y). 

```tex  
\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}}
```

## 6.3. Auto-Corr Function (ACF) and Partial Auto-Corr Function (PACF)

Let's learn about 2 very useful plot types 

- ACF AutoCorrelation Function Plot
- PACF Partial AutoCorrelation Function Plot

To understand these plots, we first need to understand **correlation**. Correlation is a measure of the strength of the linear relationship between two variables. The closer the correlation is to +1, the stronger the positive linear relationship The closer the correlation is to-1, the stronger the negative linear relationship. And the closer the correlation is to zero, the weaker the linear relationship. 

An **autocorrelation** plot (also known as a Correlogram) shows the correlation of the series with itself, lagged by t time units. So the y axis the correlation and the x axis is the number of time units of lag.

Imagine we had some sales data. We can compare the standard sales data against the sales data shifted by 1 time step. This answers the question, **"How correlated are today's sales to yesterday's sales?**"

We consider `y = Sales on Day t` vs `x = Sales on Day t-1` and calculate the correlation. We then plot the corr value on the plot `y = Autocorr` vs `x = Shift`. 

An autocorrelation plot shows the correlation of the series with itself, lagged by X time units. You go on and do this for all possible time lags x and this defines the plot. Typical examples: 

- Gradual decline: Common for seasonal data. The distance between peaks represent the season period
- Sharp Drop-off: After a couple of shifts the corr rapidly drops

It makes sense that in general there is a decline of some sort, the further away you get with the shift, the less likely the time series would be correlated with itself.

For the partial autocorr plot, we essentially plot out the relationship between `y = previous day's residuals` vs the `real values of the current day`. In general we expect the partial autocorrelation to drop off quite quickly.

The ACF describes the autocorrelation between an observation and another observation at a prior time step that includes direct and indirect dependence information. The PACF only describes the direct relationship between an observation and its lag.

These two plots can help choose order parameters for ARIMA based models. Later on, we will see that it is usually much easier to perform a **grid search** of the parameter values, rather than attempt to read these plots directly.

## 6.4. ARIMA

Continue here! 

# 7. Misc 

[regex101](https://regex101.com/)

**Downgrading Jupyter Lab and Conda Packages**

We downgrade jupyter lab from 3.2 to 3.1 to increase contextual help performance. We also install seaborn and a formatter. 

```shell
conda install -c conda-forge jupyterlab=3.1.19
conda install -c conda-forge seaborn==0.11.0
conda install -c conda-forge jupyterlab_code_formatter
```
 
We also install seaborn. 


**If-Name-Equals_Main Idiom**

From [video](https://www.youtube.com/watch?v=g_wlZ9IhbTs). We should be using the following idiom in all python (non-library) scripts: 

```python
def main(): 
	print('Hello World')

if __name__ == '__main__': 
	main()
```

**Dictionaries**

6 Different ways to create Dictionaries ([source](https://thispointer.com/python-6-different-ways-to-create-dictionaries/))

**Stack Overflow Threads**

- [Change order of x-axis labels](https://stackoverflow.com/questions/47255746/change-order-on-x-axis-for-matplotlib-chart)

## 7.1. How to Decompose Time Series Data into Trend and Seasonality

Time series decomposition involves thinking of a series as a combination of the following components. 

- level
- trend
- seasonality
- noise

