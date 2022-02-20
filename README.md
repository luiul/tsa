<!-- title: Time Series Analysis and Forecasting with Python -->
<!-- omit in toc -->
# Time Series Analysis and Forecasting with Python

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
- [3. Misc](#3-misc)

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
cond = df['W'] > 0
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
```

## 2.7. Common Operations

Continue here! 

# 3. Misc 

**Downgrading Jupyter Lab and Conda Packages**

We downgrade jupyter lab from 3.2 to 3.1 to increase contextual help performance. 

```shell
conda install -c conda-forge jupyterlab=3.1.19
```

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
