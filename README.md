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
<summary>Click here to see Course Curriculum</summary>

1. Course Goals
   1. Understand how to use Python to work with Time Series data
   2. Use Pandas and Statsmodels to visualize Time Series data
   3. Be able to use a wide variety of forecasting techniques on Time Series data
2. Set Up and Installation
   1. Install Anaconda and Python 
   2. Set up Virtual Environment
   3. Understand Jupyter Notebook
3. NumPy Basics: Quick section on NumPy basics and how to manipulate data with it
4. Pandas Basics: Pandas is used for data analysis and data exploration. We'll learn how to use this library since its fundamental to handling our data sources
5. Pandas Visualization: Pandas also comes with a lot of built in visualization tools which we will explore to quickly view time series data on a chart
6. Time Series with Pandas: After learning the Pandas Fundamentals we will focus on specialized tools within pandas specifically designed to work with time stamped data
7. Time Series Analysis with Statsmodels: Statsmodels is a statistical library for Python that contains an entire library of time series statistical analysis tools. This section will be an introduction to use Statsmodels for basic time series analysis
8. General Forecasting Models: In this section we'll dive deep into various forecasting models based on ARIMA (AutoRegressive Integrated Moving Averages)
9. Deep Learning and Prophet: The final sections will show the latest state of the art methods for forecasting, including Recurrent Neural Networks and Facebook's Prophet library

</details>

<!-- omit in toc -->
## Table of Contents
<!-- toc here -->
- [1. NumPy](#1-numpy)
  - [1.1. Broadcasting](#11-broadcasting)
  - [1.2. Random Numbers](#12-random-numbers)
  - [1.3. Indexing and Selection](#13-indexing-and-selection)
  - [1.4. NumPy Operations](#14-numpy-operations)
- [2. Misc](#2-misc)


# 1. NumPy
NumPy is a numerical processing library that can efficiently handle large data sets stored as arrays. Later we will learn about Pandas, which is built directly off of the NumPy library. It provides (see [Repo](https://github.com/numpy/numpy)):

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

NumPy operations are usually done on pairs of arrays on an element-by-element basis. In the simplest case, the two arrays must have exactly the same shape

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

Continue here! 

# 2. Misc 

We downgrade jupyter lab from 3.2 to 3.1 to make the contextual help less slow. 

```shell
conda install -c conda-forge jupyterlab=3.1.19
```