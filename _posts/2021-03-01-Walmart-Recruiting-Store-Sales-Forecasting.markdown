---
layout: post
title:  "Walmart Recruiting - Store Sales Forecasting"
date:   2021-03-01 01:08:00 +0900
categories: Python Pandas
comments: true
---

The very first Kaggle challenge I've tried is *'Walmart Recruiting - Store Sales Forecasting'*.

This challenge is to predict weekly sales of Walmart from 2012-11-02 to 2013-07-26.

#### What I realised to perform **data preprocessing**:
1. Only numeric figures can be detected and used for compute to process and learn data. Therefore, those figures that are *not* numeric should either be converted to numbers or be eliminated. (Elimination is not a good way of preprocessing though - it will lower the accuracy of prediction.) ~~I am going to use elimination anyways though - it's easier for now~~ 
2. The size of train and test data should be the same. Not only csv files but images or recordings from *deep learning* should have the same sizes. Therefore, we need to make both train and test data have the same numbers of columns. This can be done simply by eliminating useless data. (There is no *useless* data though - but easier to elminate on beginners' level)

### To start with...
On kaggle, there is a code palette we can use called **'Notebook'**. We do not have to download such large number of different python libraries on our own computer as this great tool of kaggle - Notebook - provides us everything we need to solve its challenges. (Only things we need are a laptop and the internet connection.) Anyways, let's start with reading out data given on this challenge.

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

```

We are using ***numpy*** and ***pandas*** to make a model for data prediction. These code lines are already given on the notebook.

```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

This code block will give all the directories available on this challenge. We will be using these files when training the model and submitting the test file.

When you press *'Shift + Enter'*, each code block runs.

```python
/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip
/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip
/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv
/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip
/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip
```

This output is given when running the code block above.

### Reading Data Sets
Now we know what data sets we have to look for.

Let's see what *train.csv* looks like.

```python
train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
train
```

Press *'Shift + Enter'* to see the output - and this output is given.

![Output1](/assets/2021-03-01/W1.png)

The columns consists of **[Store, Dept, Date, Weekly_Sales, IsHoliday]**. There are *421570* rows.

What does *test.csv* look like then?

```python
test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
test
```

![Output2](/assets/2021-03-01/W2.png)

*test.csv* has 4 columns; **[Store, Dept, Date, IsHoliday]**. There are *115064* rows.

On *test.csv*, there is ***NO*** *Weekly_Sales* which we have to predict from training the model - *train.csv*.

### Data Preprocessing
When preprocessing data, we need to make sure the size of data be the same to allow the model to train and there is **NO** other types of context than numeric information. Therefore, I'll eliminate **[Date, Weekly_sales]** columns on *train.csv* and assign it to a variable ***train2.csv***.

```python
train2 = train.drop(columns=['Date', 'Weekly_Sales'])
train2
```

![Output3](../assets/2021-03-01/W3.png)

Same process for ***test.csv*** data set - elmination of **[Date]** column and assignment to a variable ***'test2.csv'***.

```python
test2 = test.drop(columns = 'Date')
test2
```

![Output4](../assets/2021-03-01/W4.png)

And... done for data preprocessing!

### Training a model
I'm going to use this decent machine learning ensemble model to train and predict data sets - Random Forest. This model can be imported from ***sklearn*** and I will be using ***RandomForestRegressor*** as this challenge is about regression.

```python
from sklearn.ensemble import RandomForestRegressor 
rf = RandomForestRegressor(n_jobs = 4)
```

***n_jobs = 4*** is to make the model work faster by using all the 'CPU's in the model.

```python
rf.fit(train2,train['Weekly_Sales'])
output = rf.predict(test2)
output
```

Using the preprocessed data - *train2.csv* - put into the model with columns we have to predict - *'Weekly_Sales'*.

Then, we can get an output of prediction of *'test2.csv'*.

```python
array([22244.55823874, 22244.55823874, 22244.55823874, ...,
         567.26728439,   567.26728439,   567.26728439])
```

### Submission
As we got an output, we have to put this into the submission file.

First, read the file and see what it looks like.

```python
sub = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')
sub
```

![Output5](/assets/2021-03-01/W5.png)

Then we write the output we got on the file.

```python
sub['Weekly_Sales'] = output
sub
```

and... this is the submission file we have to submit.

![Output6](../assets/2021-03-01/W6.png)

Import this file into **csv** file and delete indexes just to eliminate an occurence of errors.

```python
sub.to_csv('sub.csv',index=False)
```

**DONE!!!**

----
This was the simplest way of solving this challenge, but hope I get better and better in the future by trying more challenges and learning deeper of machine learning. 

**BYE** :)
