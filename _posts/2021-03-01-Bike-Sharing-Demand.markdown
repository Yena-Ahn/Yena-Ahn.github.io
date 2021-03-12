---
layout: post
title:  "Bike Sharing Demand - P"
date:   2021-03-01 16:00:00 +0900
categories: Python Pandas
comments: true
---

```Markdown
The title with 'P' stands for 'Practice' and this challenges will always be done with the previous challenge I tried but in the same process.
```

This challenge will be done under the same way with the previous post - **'Walmart Recruiting - Store Sales Forecasting'**. 

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

Output is:
```python
/kaggle/input/bike-sharing-demand/sampleSubmission.csv
/kaggle/input/bike-sharing-demand/train.csv
/kaggle/input/bike-sharing-demand/test.csv
```

### Reading given data sets
```python
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
train
test
```

**train.csv:**<br/>
![Output1](/assets/2021-03-01/B1.png)<br/>

**test.csv:**<br/>
![Output2](https://user-images.githubusercontent.com/75198944/110883422-cb6f5880-8326-11eb-9aae-89c5d515e2cb.png)<br/>

### Data Preprocessing
```python
train2 = train.drop(columns = ['datetime', 'casual', 'registered', 'count'])
test2 = test.drop(columns = 'datetime')
train2
test2
```
**train2.csv:**<br/>
![Output3](https://user-images.githubusercontent.com/75198944/110883428-cd391c00-8326-11eb-8edf-a739a16bbab0.png)<br/>
**test2.csv:**<br/>   
![Output4](https://user-images.githubusercontent.com/75198944/110883435-cf02df80-8326-11eb-8044-2553b4c379e8.png)<br/>

### Training a Model
```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs = 4)
rf.fit(train2, train['count'])
output = rf.predict(test2)
output
```
**Output**
```python
array([183.87666667,  61.98916667,  61.98916667, ...,  74.62333333,
        59.89066667,  20.19583333])
```

### Submission
```python
sub = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')
sub
```
**Output**<br/>
![Output5](https://user-images.githubusercontent.com/75198944/110883451-d629ed80-8326-11eb-8d0b-95274dd169ee.png)<br/>


Put the output into the submission file.

```python
sub['count'] = output
sub
```
![Output6](https://user-images.githubusercontent.com/75198944/110883461-d88c4780-8326-11eb-9c93-e714a3190640.png)

And... Get ready to submit!

```python
sub.to_csv('sub.csv', index = False)
```

### Let's see what my score is...
![Score1](/assets/2021-03-01/BS1.png)
![Score2](https://user-images.githubusercontent.com/75198944/110883465-db873800-8326-11eb-9a42-16b07e925942.png)