---
layout: post
title:  "Adding Separate Date Columns - Walmart & Bike Sharing"
date:   2021-03-19 01:08:00 +0900
categories: Python Pandas Seaborn
comments: true
---

It's time to get a better score from both challenges I tried last time. 

##Walmart Recruiting
On the previous step, I deleted all the date column as it cannot be identified as a numeric column. However, it's time to change it to a date figure and extract a year, a month, and a day using *pandas*. The original date column should be deleted though as it still contains objects. I am going to add new columns of each extracted date instead.

So, the first step is to create **pandas objects** of both train and test data sets which change the date to a **pandas object** and assign it to a *Date* column.

```python
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
```
Then add a new column called *Year, Month,* and *Day* on both *train* and *test* data sets and extract each *Year, Month and Day* from the *pandas* object. (Remember to add on at each time and check if the information given is useful for the prediction.)

```python
train['Year'] = train['Date'].dt.year
test['Year'] = test['Date'].dt.year
```
```python
train['Month'] = train['Date'].dt.month
test['Month'] = test['Date'].dt.month
```
```python
train['Day'] = train['Date'].dt.day
test['Day'] = test['Date'].dt.day
```
Especially the reason why the *Day* column is important for the output is because there is a difference of sales when it is the start of a month and the end of a month.

We can check this out through grouping *train* set by *Day* column and showing the mean *Weekly_Sales* of each day.

```python
train.groupby('Day')['Weekly_Sales'].mean()
```

Then we get the output of below:
```python
Day
1     15438.693561
2     16282.825224
3     16243.591840
4     16262.123390
5     16152.185352
6     16540.152518
7     15851.075764
8     16026.260550
9     16563.236628
10    16684.753968
11    15946.176870
12    15769.983633
13    15358.628217
14    14843.544775
15    15476.565690
16    16400.916009
17    16890.881861
18    15914.848035
19    15634.121093
20    15236.067426
21    14987.074671
22    15696.588300
23    17183.530448
24    17732.369226
25    16816.726353
26    16742.479327
27    14908.182971
28    15056.847856
29    15117.011770
30    15038.411334
31    14833.557736
Name: Weekly_Sales, dtype: float64
```

As we can see from the output above, the Sales is higher on the start of a month than the end of a month.

___
I bet I'd get a better score with the added columns and information, so let's check out! <br/>

![Output1](https://user-images.githubusercontent.com/75198944/111740334-e8c49980-88c7-11eb-977f-067a60a33a16.png)
<br/>
... and this is my last score. <br/>

![Output2](https://user-images.githubusercontent.com/75198944/111740493-33461600-88c8-11eb-96aa-e3478314103c.png)



So I got a better score than the previous try!!

---
##Bike Sharing Demand

Let's do the same steps to this challenge as well.

This data set contains *date* and *time*.

```python
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])
```

As we see the data set, the sales would be affected by the year and hours - and this can be seen from a graph.

```python
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(14,8))
sns.boxplot(train['year'],train['count'])
```
<br/>

![Output3](https://user-images.githubusercontent.com/75198944/112266636-a0d2b780-8cb7-11eb-9f2d-f469c06b758b.png)

<br/>

So add columns *hour* and *year* on *test* and *train* sets.

```python
train['hour'] = train['datetime'].dt.hour
test['hour'] = test['datetime'].dt.hour
train['year'] = train['datetime'].dt.year
test['year'] = test['datetime'].dt.year
```

My new score for this challenge is...
<br/>

![Output4](https://user-images.githubusercontent.com/75198944/112267292-91a03980-8cb8-11eb-9357-0ee26a1005ca.png)

![Output5](https://user-images.githubusercontent.com/75198944/112267296-92d16680-8cb8-11eb-9e71-f46f7e9c4350.png)