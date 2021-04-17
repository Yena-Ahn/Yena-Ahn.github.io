---
layout: post
title:  "Housing Prices Competition for Kaggle Learn Users"
date:   2021-04-16 13:14:00 +0900
categories: Python Pandas
comments: true
---
Another regression challenge here! This competition is to predict house prices using given data. Let's build base lines first.

```python
import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
```python
/kaggle/input/home-data-for-ml-course/sample_submission.csv
/kaggle/input/home-data-for-ml-course/sample_submission.csv.gz
/kaggle/input/home-data-for-ml-course/train.csv.gz
/kaggle/input/home-data-for-ml-course/data_description.txt
/kaggle/input/home-data-for-ml-course/test.csv.gz
/kaggle/input/home-data-for-ml-course/train.csv
/kaggle/input/home-data-for-ml-course/test.csv
```
```python
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
```
```python
pd.options.display.max_columns = 999
display(train, test)
```
**train.csv** <br/>
![Output1](https://user-images.githubusercontent.com/75198944/115101495-98506280-9f7f-11eb-91eb-cdde1b679c50.png) <br/>
**test.csv** <br/>
![Output2](https://user-images.githubusercontent.com/75198944/115101496-99818f80-9f7f-11eb-8ebb-0705c971d8cc.png)<br/>

```python
alldata = pd.concat([train, test])
```

###Preprocessing
Have a look at the data - there are two points we have to handle so the model can learn better.
* It is important to check how old a house is when it is sold. Create a new column *Old* which shows how old a house is from when it is constructed to when it is sold.
  ```python
  alldata['Old'] = alldata['YrSold'] - alldata['YearBuilt']
  ```
* It is also important to acknowledge overall quality and condition rating together in order to improve training.
  ```python
  alldata['Overall'] = alldata['OverallQual'] + alldata['OverallCond']
  ```

![Output3](https://user-images.githubusercontent.com/75198944/115102632-13694700-9f87-11eb-8a6c-5e32b8a97a2f.png)
<br/>

Now drop all the useless columns.
```python
alldata2 = alldata.drop(columns = ['SalePrcie', 'Id'])
```

We have categorial columns so we need *Label Encoder* to convert them into numeric values.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c = alldata2.columns[alldata2.dtypes == object]
for i in c:
    alldata2[i] = le.fit_transform(alldata2[i])
```

We see some *na* values too in this data set, therefore we need to convert those to -1.

```python
alldata2 = alldata2.fillna(-1)
```

...and split the data into train and test set.

```python
train2 = alldata2[:len(train)]
test2 = alldata2[len(test):]
```

### Model training
This time we are going to use a different model called *XGboost*. 

```Python
from xgboost import XGBRegressor
xgb = XGBRegressor(learning_rate = 0.1)
xgb.fit(train2, train['SalePrice'])
output = xgb.predict(test2)
```
```python
array([122587.88, 153517.38, 184783.62, ..., 157521.  , 115749.16,
       220430.73], dtype=float32)
```
### Submission
```python
sub = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')
sub['SalePrice'] = output
sub.to_csv('sub.csv', index = False)
```

![Output4](https://user-images.githubusercontent.com/75198944/115102636-15330a80-9f87-11eb-8345-faf88650fafe.png)
