---
title: Computational Investing Part I (Coursera) - Week 3
date: 2013-11-27 00:00
status: draft
category: reference/moocs
parent: computational-investing-part-1
tags:
  - MOOC
  - Investing
---

## Week 3

#### Overview

* Was very happy to be introduced to Numpy
* There are, however, better resources for learning Numpy
* Really enjoyed playing with QSTK -- getting a taste of what a quant framework is like

#### 1.1

* Numpy tutorial
* Easier to go through tutorial via Wiki
* A real eye-opener. Seriously, one of the best things I've ever done. How was I ever going to get introduced to Numpy?

#### 1.2

* Slicing arrays
* Indexing using an array of indices
* Operations on arrays. Work out how many elements in the array is better than average.

#### 1.3

* Performing basic operations on matrices

```python
In [1]: squareArray * 2
Out[1]:
array([[ 2,  4,  6],
       [ 8, 10, 12],
       [14, 16, 18]])
```

* Matrix multiplication

```python
In [1]: matA = np.array( [[1,2], [3,4] ] )

In [1]: matB = np.array( [[5,6], [7,8] ] )

In [1]: matA * matB
Out[1]:
array([[ 5, 12],
       [21, 32]])
```

#### 2.1

* QSTK overview
* Set a start and end date

```python
In [2]: import QSTK.qstkutil.qsdateutil as du

In [5]: import datetime as dt

In [8]: ls_symbols = ['AAPL', 'GLD', 'GOOG', '$SPX', 'XOM']

In [9]: dt_start = dt.datetime(2010, 1, 1)

In [10]: dt_end = dt.datetime(2010, 1, 15)

In [11]: dt_timeofday = dt.timedelta(hours=16)

In [12]: ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

In [13]: ldt_timestamps
Out[13]:
[Timestamp('2010-01-04 16:00:00', tz=None),
 Timestamp('2010-01-05 16:00:00', tz=None),
 Timestamp('2010-01-06 16:00:00', tz=None),
 Timestamp('2010-01-07 16:00:00', tz=None),
 Timestamp('2010-01-08 16:00:00', tz=None),
 Timestamp('2010-01-11 16:00:00', tz=None),
 Timestamp('2010-01-12 16:00:00', tz=None),
 Timestamp('2010-01-13 16:00:00', tz=None),
 Timestamp('2010-01-14 16:00:00', tz=None)]
```

* "I wish I could edit this out and start over again." You kinda can, dude. :)
* Get data object

```python
In [14]: c_dataobj = da.DataAccess('Yahoo')

In [15]: c_dataobj
Out[15]: <QSTK.qstkutil.DataAccess.DataAccess at 0xa3d14cc>

In [16]: ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

In [17]: ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
In [20]: d_data = dict(zip(ls_keys, ldf_data))
In [22]: d_data['close']
Out[22]:
                       AAPL     GLD    GOOG     $SPX    XOM
2010-01-04 16:00:00  213.10  109.80  626.75  1132.99  64.55
2010-01-05 16:00:00  213.46  109.70  623.99  1136.52  64.80
2010-01-06 16:00:00  210.07  111.51  608.26  1137.14  65.36
2010-01-07 16:00:00  209.68  110.82  594.10  1141.69  65.15
2010-01-08 16:00:00  211.07  111.37  602.02  1144.98  64.89
2010-01-11 16:00:00  209.21  112.85  601.11  1146.98  65.62
2010-01-12 16:00:00  206.83  110.49  590.48  1136.22  65.29
2010-01-13 16:00:00  209.75  111.54  587.09  1145.68  65.03
2010-01-14 16:00:00  208.53  112.03  589.85  1148.46  65.04

```

* Plot data

```python
In [16]: na_price = d_data['close'].values

In [17]: na_price
Out[17]:
array([[  213.1 ,   109.8 ,   626.75,  1132.99,    64.55],
       [  213.46,   109.7 ,   623.99,  1136.52,    64.8 ],
       [  210.07,   111.51,   608.26,  1137.14,    65.36],
       [  209.68,   110.82,   594.1 ,  1141.69,    65.15],
       [  211.07,   111.37,   602.02,  1144.98,    64.89],
       [  209.21,   112.85,   601.11,  1146.98,    65.62],
       [  206.83,   110.49,   590.48,  1136.22,    65.29],
       [  209.75,   111.54,   587.09,  1145.68,    65.03],
       [  208.53,   112.03,   589.85,  1148.46,    65.04]])

In [18]: plt.clf()

In [19]: plt.plot(ldt_timestamps, na_price)
Out[19]:
[<matplotlib.lines.Line2D at 0xaf4abac>,
 <matplotlib.lines.Line2D at 0xb25632c>,
 <matplotlib.lines.Line2D at 0xb2564ac>,
 <matplotlib.lines.Line2D at 0xb25662c>,
 <matplotlib.lines.Line2D at 0xb2567ac>]

In [20]: plt.legend(ls_symbols)
Out[20]: <matplotlib.legend.Legend at 0xb25a56c>

In [21]: plt.ylabel('Adjusted Close')
Out[21]: <matplotlib.text.Text at 0xaf52eec>

In [22]: plt.xlabel('Date')
Out[22]: <matplotlib.text.Text at 0xaf4a70c>

In [23]: plt.show()
```

#### 2.2

* Normalising the data by comparing it with the first price

```python
In [26]: na_normalized_price = na_price / na_price[0,:]

In [27]: na_normalized_price
Out[27]:
array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
       [ 1.00168935,  0.99908925,  0.99559633,  1.00311565,  1.00387297],
       [ 0.98578132,  1.01557377,  0.9704986 ,  1.00366287,  1.01254841],
       [ 0.9839512 ,  1.00928962,  0.94790586,  1.0076788 ,  1.00929512],
       [ 0.99047396,  1.01429872,  0.96054248,  1.01058262,  1.00526723],
       [ 0.98174566,  1.02777778,  0.95909055,  1.01234786,  1.0165763 ],
       [ 0.97057719,  1.00628415,  0.94213004,  1.00285086,  1.01146398],
       [ 0.98427968,  1.01584699,  0.93672118,  1.01120045,  1.0074361 ],
       [ 0.97855467,  1.02030965,  0.94112485,  1.01365414,  1.00759101]])
In [28]: plt.plot(ldt_timestamps, na_normalized_price)
Out[28]:
[<matplotlib.lines.Line2D at 0xb49b8ac>,
 <matplotlib.lines.Line2D at 0xb5197cc>,
 <matplotlib.lines.Line2D at 0xb51994c>,
 <matplotlib.lines.Line2D at 0xb519acc>,
 <matplotlib.lines.Line2D at 0xb519c4c>]

In [29]: plt.show()
```
