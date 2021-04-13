---
layout: post
title:  "Don't Get Kicked!"
date:   2021-03-31 12:30:00 +0900
categories: Python Pandas Seaborn 
comments: true
---

This challenge is to predict if the used car at an auto auction is a *Kick* (Bad buy) - the risk of that vehicle might have serious issues which prevent it from being sold to customers. 

```python
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
```python
/kaggle/input/DontGetKicked/example_entry.csv
/kaggle/input/DontGetKicked/training.zip
/kaggle/input/DontGetKicked/Carvana_Data_Dictionary.txt
/kaggle/input/DontGetKicked/test.zip
/kaggle/input/DontGetKicked/training.csv
/kaggle/input/DontGetKicked/test.csv
```
###Build base lines
```python
train = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
test = pd.read_csv('/kaggle/input/DontGetKicked/test.csv')
```
```python
display(train, test)
```
**train.csv** <br/>
![Output1](https://user-images.githubusercontent.com/75198944/113552187-bee1d580-9630-11eb-9b92-c4c5e28a6e39.png)
![Output2](https://user-images.githubusercontent.com/75198944/113552197-c1442f80-9630-11eb-9213-fc726ddb0d47.png)
<br/>
**test.csv** <br/>
![Output3](https://user-images.githubusercontent.com/75198944/113552199-c1dcc600-9630-11eb-80b9-d6f307574fc2.png)
![Output4](https://user-images.githubusercontent.com/75198944/113552203-c2755c80-9630-11eb-9d19-115a6d160883.png)

As it is too uncomfortable writing the same code for both data sets when data preprocessing, we can simply put these data sets together and make as a one data set to reduce redundancy.

```python
alldata = pd.concat([train, test])
```

### To start with
For data preprocessing, we need to convert all *object* type columns to *numeric* type - integer or float. 

#### AgeuponOutcome
This column contains time which can be converted to smaller measures such as day.

Before converting each columns, we need to figure out what information it contains.

```python
alldata['AgeuponOutcome'].unique()
```
*unique()* shows only unique values in the column.

```python
array(['1 year', '2 years', '3 weeks', '1 month', '5 months', '4 years',
       '3 months', '2 weeks', '2 months', '10 months', '6 months',
       '5 years', '7 years', '3 years', '4 months', '12 years', '9 years',
       '6 years', '1 weeks', '11 years', '4 weeks', '7 months', '8 years',
       '11 months', '4 days', '9 months', '8 months', '15 years',
       '10 years', '1 week', '0 years', '14 years', '3 days', '6 days',
       '5 days', '5 weeks', '2 days', '16 years', '1 day', '13 years',
       nan, '17 years', '18 years', '19 years', '20 years', '22 years'],
      dtype=object)
```
As we see from the result, *day* would be the optimal measure for each columns to be converted as it is the smallest.

Therefore, we need to define a *function* that can remove strings after numbers and convert it into *days*.

```python
def Age(i):
    if pd.isnull():
        return -1
    num = int(i.split()[0])
    if 'year' in i:
        return num * 365
    elif 'month' in i:
        return num * 30
    elif 'week' in i:
        return week * 7
    else:
        return num
```
... And apply the function to the column *AgeuponOutcome*.
```python
alldata['AgeuponOutcome'] = alldata['AgeuponOutcome'].apply(Age)
```
We can now see *AgeuponOutcome* column has changed.
<br/>
![Output5](https://user-images.githubusercontent.com/75198944/114509652-b9a80a80-9c70-11eb-8ca4-e36284d7d8f5.png) <br/>

#### Dates 
As we always do, it's time to split *DateTime* column into each time.

```python
alldata['DateTime'] = pd.to_datetime(alldata['DateTime'])
```
```python
alldata['Year'] = alldata['DateTime'].dt.year
alldata['Month'] = alldata['DateTime'].dt.month
alldata['Day'] = alldata['DateTime'].dt.day
alldata['Hour'] = alldata['DateTime'].dt.hour
alldata['DayofWeek'] = alldata['DateTime'].dt.dayofweek
alldata['Week'] = alldata['DateTime'].dt.week
alldata['Minute'] = alldata['DateTime'].dt.minute
alldata['Time'] = alldata['DateTime'].dt.date - alldata['DateTime'].dt.date.min()
```
*Time* column contains strings (*days*) though, so they need to be eliminated.

```python
alldata['Time'] = alldata['Time'].apply(lambda x:x.days)
```

To check if all these times can be helpful for training, we can simply draw a graph.

As this challenge is a classification problem and the answer column - 'OutcomeType' - is a categorial column, *count plot* should be useful.

```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
sns.countplot(alldata['Hour'], hue = alldata['OutcomeType'])
```
**Hour** <br/>
![Output6](https://user-images.githubusercontent.com/75198944/114517030-e52ef300-9c78-11eb-98a1-4fc06cb7a52c.png)
<br/>

The *adaption* rate is relatively high at 5pm - 6pm and the *transfer* rate is relatively high at 12am and 9am. 

```python
plt.figure(figsize=(12,8))
sns.countplot(alldata['Minute'], hue = alldata['OutcomeType'])
```

**Minute**<br/>
![Output7](https://user-images.githubusercontent.com/75198944/114517041-e6f8b680-9c78-11eb-9f01-b7c20518270f.png) <br/>

Most of the *transfer* happens exactly at each time (00 minute).

```python
plt.figure(figsize=(12,8))
sns.countplot(alldata['DayofWeek'], hue = alldata['OutcomeType'])
```

**Day of Week**<br/>
![Output8](https://user-images.githubusercontent.com/75198944/114517043-e7914d00-9c78-11eb-9787-4e3f23700439.png)
<br/>

Most people adopt animals on weekends.

```python
plt.figure(figsize=(12,8))
sns.countplot(alldata['Week'], hue = alldata['OutcomeType'])
```

**Week** <br/>
![Output9](https://user-images.githubusercontent.com/75198944/114517045-e829e380-9c78-11eb-8e6a-a78b0c8f9223.png)
<br/>

When it is the start of year or the end of year, *adoption* is high. Other outcomes are also different each week. 

#### Drop 
Remove columns we do not need or need to be eliminated such as an answer column.
```python
alldata2 = alldata.drop(columns = ['DateTime', 'AnimalID','OutcomeType','OutcomeSubtype','ID'])
```

### Final steps before model training
Categorial columns need to be encoded into numeric objects so that a model we will use later can learn and predict.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c = alldata2.columns[alldata2.dtypes == object]
for i in c:
    alldata2[i] = le.fit_transform(alldata2[i])
```

Below is the data set after completion of preprocessing. <br/>

![Output10](https://user-images.githubusercontent.com/75198944/114520118-10671180-9c7c-11eb-9018-511e1912e073.png)
<br/>

Almost done! 

Now we need to separate *alldata2* into *train2* and *test2* data sets.

```python
train2 = alldata2[:len(train)]
test2 = alldata2[len(train):]
```

### Model training
This time I am going to use a different model - CatBoost.
Boosting model is useful for this kind of challenge where it has many categorial columns.

```python
from catboost import CatBoostClassifier
cbc = CatBoostClassifier(verbose=50)
cbc.fit(train2, train['OutcomeType'])
output = cbc.predict_proba(test2)
```
```python
Learning rate set to 0.093562
0:	learn: 1.4724106	total: 23.4ms	remaining: 23.4s
50:	learn: 0.7823152	total: 987ms	remaining: 18.4s
100:	learn: 0.7454686	total: 2s	remaining: 17.8s
150:	learn: 0.7225850	total: 2.99s	remaining: 16.8s
200:	learn: 0.7019554	total: 4.07s	remaining: 16.2s
250:	learn: 0.6858341	total: 5.08s	remaining: 15.2s
300:	learn: 0.6688145	total: 6.11s	remaining: 14.2s
350:	learn: 0.6552639	total: 7.12s	remaining: 13.2s
400:	learn: 0.6411333	total: 8.16s	remaining: 12.2s
450:	learn: 0.6280911	total: 9.42s	remaining: 11.5s
500:	learn: 0.6162696	total: 10.5s	remaining: 10.4s
550:	learn: 0.6042127	total: 11.4s	remaining: 9.32s
600:	learn: 0.5927829	total: 12.5s	remaining: 8.28s
650:	learn: 0.5817213	total: 13.5s	remaining: 7.22s
700:	learn: 0.5718307	total: 14.5s	remaining: 6.19s
750:	learn: 0.5621950	total: 15.5s	remaining: 5.15s
800:	learn: 0.5516297	total: 16.6s	remaining: 4.11s
850:	learn: 0.5424182	total: 17.6s	remaining: 3.08s
900:	learn: 0.5333513	total: 18.6s	remaining: 2.04s
950:	learn: 0.5240316	total: 19.6s	remaining: 1.01s
999:	learn: 0.5147336	total: 20.6s	remaining: 0us
array([[2.10440287e-02, 2.28812812e-03, 7.99573244e-02, 1.70381373e-01,
        7.26329146e-01],
       [7.11690336e-01, 2.19766427e-04, 7.92554303e-03, 2.35654019e-01,
        4.45103357e-02],
       [4.25415924e-01, 1.11783072e-03, 9.45465653e-03, 1.13142795e-01,
        4.50868793e-01],
       ...,
       [5.73090721e-04, 4.04161882e-03, 5.76239033e-03, 1.04827288e-03,
        9.88574627e-01],
       [4.01602361e-01, 1.68657486e-03, 8.29771407e-03, 5.39613815e-01,
        4.87995344e-02],
       [9.26913865e-02, 1.63824204e-03, 1.39206120e-01, 6.34170099e-01,
        1.32294152e-01]])
```

### Submission

It's time to submit and check what my score is!

```python
sub = pd.read_csv('/kaggle/input/shelter-animal-outcomes/sample_submission.csv.gz')
sub.iloc[:,1:] = output
sub.to_csv('sub.csv', index = False)
```

**sub.csv** <br/>
![Output11](https://user-images.githubusercontent.com/75198944/114526728-35f71980-9c82-11eb-8cbb-6d2210e3f226.png) <br/>

## Score
<br/>

![Output12](https://user-images.githubusercontent.com/75198944/114527187-a1d98200-9c82-11eb-8985-b9d0d838e1a4.png)

![Output13](https://user-images.githubusercontent.com/75198944/114527192-a30aaf00-9c82-11eb-88ad-7e9a74980b76.png)
<br/>

Around 300th place :)

