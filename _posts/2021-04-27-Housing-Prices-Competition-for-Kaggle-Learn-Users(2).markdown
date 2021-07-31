---
layout: post
title:  "Housing Prices Competition for Kaggle Learn Users (2)"
date:   2021-04-27 22:00:00 +0900
categories: Python Pandas Seaborn 
comments: true
---

This time I'm going to try getting higher score from this challenge by using an ensemble model (tree model + linear model) instead of just using a tree model.

There are also some more preprocessing steps to be done for this.

As we always did before, we have to read csv files first.

```python
import numpy as np 
import pandas as pd 
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
