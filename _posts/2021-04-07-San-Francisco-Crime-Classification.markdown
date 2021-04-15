---
layout: post
title:  "San Francisco Crime Classification"
date:   2021-04-07 18:09:00 +0900
categories: Python Pandas
comments: true
---
Another classification challenge is here! Just like what I did before, I need to build base lines first.

```python
import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
```python
/kaggle/input/sf-crime/train.csv.zip
/kaggle/input/sf-crime/sampleSubmission.csv.zip
/kaggle/input/sf-crime/test.csv.zip
```
```python
train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
test = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip')
display(train, test)
```
**train.csv & test.csv** <br/>
![Output1](https://user-images.githubusercontent.com/75198944/114528687-31336500-9c84-11eb-9c81-27b9d6c4fe4f.png)
![Output2](https://user-images.githubusercontent.com/75198944/114528698-342e5580-9c84-11eb-9a38-46068f18573c.png) <br/>

Now join two data sets together and start preprocessing.

```python 
alldata = pd.concat([train, test])
```

#### Dates 
```python
alldata['Dates'] = pd.to_datetime(alldata['Dates'])
alldata['Year'] = alldata['Dates'].dt.year
alldata['Month'] = alldata['Dates'].dt.month
alldata['Day'] = alldata['Dates'].dt.day
alldata['Week'] = alldata['Dates'].dt.week
alldata['Hour'] = alldata['Dates'].dt.hour
alldata['Minute'] = alldata['Dates'].dt.minute
alldata['Time'] = (alldata['Dates'].dt.date - alldata['Dates'].dt.date.min()).apply(lambda x:x.days)
```
![Output3](https://user-images.githubusercontent.com/75198944/114529773-42c93c80-9c85-11eb-8e66-6876b59e4dca.png) <br/>

#### Address
Here in this challenge, the *Address* column has too many unique values which let *Label Encoding* redundant on later steps. 

As we look closely into the *Adress* column, some have the same streets but different building number. If we simply convert them into different numbers with *Label Encoder*, the model won't know whether the place is nearby other places or not. This is why we need a *text mining* here using ***Tfid***.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
text = tfidf.fit_transform(alldata['Address'])
```
```python
<1762311x2192 sparse matrix of type '<class 'numpy.float64'>'
	with 7886561 stored elements in Compressed Sparse Row format>
```

There are about 2200 unique words in *Address* column and they are stored in sparse matrix.

#### Drop 
```python
alldata2 = alldata.drop(columns = ['Dates', 'Category','Descript','Id', 'Resolution'])
```
Drop all the useless columns and we are ready to perform *label encoding*. <br/>

![Output4](https://user-images.githubusercontent.com/75198944/114529782-43fa6980-9c85-11eb-843a-5fe40bfb02de.png) <br/>

#### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c = alldata2.columns[alldata2.dtypes == object]
for i in c:
    alldata2[i] = le.fit_transform(alldata2[i])
```
<br/>

![Output5](https://user-images.githubusercontent.com/75198944/114529792-45c42d00-9c85-11eb-8c75-223be031b0de.png) <br/>

### Check Your Score without Submission
```python
train2 = alldata2[:len(train)]
test2 = alldata2[len(train):]
```
Now we have *train2* set and *test2* set.

Before putting into model, we have to split the train set and the test set to let model give out the predicted score of our prediction output. This step is important as we cannot always submit the file everytime we change or add something; there may be a restriction of the number of submission. 

```python
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train2, train['Category'], test_size = 0.2, random_state = 42, stratify = train['Category'])
```
It will randomly select rows from train set and the answer column so that later when we get a prediction output, the model can automatically gives out the score we would get.

### Model training 
```python
from catboost import CatBoostClassifier
cbc = CatBoostClassifier(task_type = 'GPU', verbose = 30, learning_rate = 0.1, iterations = 10000)
cbc.fit(x_train, y_train, eval_set = (x_valid, y_valid), early_stopping_rounds = 30)
output = cbc.predict_proba(test2)
```
Here we have more options for *Catboost* model. 
* *task_type* is to select which accelerator to use such as *CPU* or *GPU*. This one has to use *GPU* as the model takes too much time to learn when using *CPU*.
* *verbose* is to set how many steps to show while learning. I put it to 30 so the steps will be shown after every 30 steps of trying.
* Smaller the *learning_rate*, more precise the output. However, it should not be too small as it would take forever to learn. We have to find an optimal learning rate but it should be at most 0.1 so the output can be more precise.
* *iterations* sets how many times the model will learn. It was 1000 times before I set it to 10000 and I found out the model can come out with better outputs if it runs more times. 
  
To get a predicted score for our output that model comes out, we need two train sets we split on earlier stage - *x_train, y_train*. *x_train* and *x_valid* are used for the actual learning of the model while *y_train* is used to get a score based on *y_valid*.
* *early_stopping_rounds* sets the number of times the model repeats to learn when it cannot get a better result.
  

**Output**
```python
0:	learn: 3.3022052	test: 3.3029319	best: 3.3029319 (0)	total: 82.2ms	remaining: 13m 42s
30:	learn: 2.4269292	test: 2.4306131	best: 2.4306131 (30)	total: 2.15s	remaining: 11m 31s
60:	learn: 2.3749358	test: 2.3815664	best: 2.3815664 (60)	total: 4.12s	remaining: 11m 10s
90:	learn: 2.3496837	test: 2.3591225	best: 2.3591225 (90)	total: 6.16s	remaining: 11m 11s
120:	learn: 2.3324685	test: 2.3446636	best: 2.3446636 (120)	total: 8.29s	remaining: 11m 16s
150:	learn: 2.3175884	test: 2.3321271	best: 2.3321271 (150)	total: 10.2s	remaining: 11m 8s
180:	learn: 2.3063894	test: 2.3234678	best: 2.3234678 (180)	total: 12.2s	remaining: 11m 2s
210:	learn: 2.2953632	test: 2.3148760	best: 2.3148760 (210)	total: 14.2s	remaining: 10m 57s
240:	learn: 2.2868100	test: 2.3088271	best: 2.3088271 (240)	total: 16.1s	remaining: 10m 52s
270:	learn: 2.2785820	test: 2.3033622	best: 2.3033622 (270)	total: 18.3s	remaining: 10m 57s
300:	learn: 2.2718910	test: 2.2994119	best: 2.2994119 (300)	total: 20.3s	remaining: 10m 54s
330:	learn: 2.2653969	test: 2.2957354	best: 2.2957354 (330)	total: 22.3s	remaining: 10m 51s
360:	learn: 2.2594343	test: 2.2924709	best: 2.2924709 (360)	total: 24.3s	remaining: 10m 48s
390:	learn: 2.2534382	test: 2.2892539	best: 2.2892539 (390)	total: 26.2s	remaining: 10m 45s
420:	learn: 2.2480913	test: 2.2863832	best: 2.2863832 (420)	total: 28.3s	remaining: 10m 44s
450:	learn: 2.2429471	test: 2.2839104	best: 2.2839104 (450)	total: 30.5s	remaining: 10m 45s
480:	learn: 2.2376018	test: 2.2811035	best: 2.2811035 (480)	total: 32.9s	remaining: 10m 50s
510:	learn: 2.2330133	test: 2.2792858	best: 2.2792858 (510)	total: 34.9s	remaining: 10m 47s
540:	learn: 2.2283837	test: 2.2772472	best: 2.2772472 (540)	total: 36.9s	remaining: 10m 44s
570:	learn: 2.2240691	test: 2.2756641	best: 2.2756641 (570)	total: 38.9s	remaining: 10m 42s
600:	learn: 2.2196148	test: 2.2742761	best: 2.2742761 (600)	total: 41.1s	remaining: 10m 42s
630:	learn: 2.2154322	test: 2.2725713	best: 2.2725713 (630)	total: 43.1s	remaining: 10m 39s
660:	learn: 2.2114333	test: 2.2714848	best: 2.2714848 (660)	total: 45.1s	remaining: 10m 36s
690:	learn: 2.2072454	test: 2.2700658	best: 2.2700658 (690)	total: 47.1s	remaining: 10m 34s
720:	learn: 2.2032742	test: 2.2688577	best: 2.2688577 (720)	total: 49.1s	remaining: 10m 31s
750:	learn: 2.1993757	test: 2.2678165	best: 2.2678165 (750)	total: 51.3s	remaining: 10m 31s
780:	learn: 2.1959206	test: 2.2670020	best: 2.2670020 (780)	total: 53.3s	remaining: 10m 28s
810:	learn: 2.1922868	test: 2.2660389	best: 2.2660389 (810)	total: 55.3s	remaining: 10m 26s
840:	learn: 2.1886714	test: 2.2651136	best: 2.2651136 (840)	total: 57.3s	remaining: 10m 23s
870:	learn: 2.1852971	test: 2.2643297	best: 2.2643297 (870)	total: 59.3s	remaining: 10m 21s
900:	learn: 2.1818304	test: 2.2633812	best: 2.2633812 (900)	total: 1m 1s	remaining: 10m 20s
930:	learn: 2.1783464	test: 2.2625964	best: 2.2625906 (928)	total: 1m 3s	remaining: 10m 18s
960:	learn: 2.1748787	test: 2.2619231	best: 2.2619231 (960)	total: 1m 6s	remaining: 10m 20s
990:	learn: 2.1714878	test: 2.2613741	best: 2.2613741 (990)	total: 1m 8s	remaining: 10m 18s
1020:	learn: 2.1680806	test: 2.2607507	best: 2.2607507 (1020)	total: 1m 10s	remaining: 10m 16s
1050:	learn: 2.1647303	test: 2.2599831	best: 2.2599831 (1050)	total: 1m 12s	remaining: 10m 15s
1080:	learn: 2.1612622	test: 2.2594521	best: 2.2594521 (1080)	total: 1m 14s	remaining: 10m 13s
1110:	learn: 2.1582002	test: 2.2588760	best: 2.2588760 (1110)	total: 1m 16s	remaining: 10m 10s
1140:	learn: 2.1551261	test: 2.2583087	best: 2.2583087 (1140)	total: 1m 18s	remaining: 10m 8s
1170:	learn: 2.1517573	test: 2.2576871	best: 2.2576857 (1168)	total: 1m 20s	remaining: 10m 5s
1200:	learn: 2.1487917	test: 2.2571570	best: 2.2571570 (1200)	total: 1m 22s	remaining: 10m 3s
1230:	learn: 2.1457899	test: 2.2565475	best: 2.2565315 (1226)	total: 1m 24s	remaining: 10m 2s
1260:	learn: 2.1427151	test: 2.2562370	best: 2.2562260 (1259)	total: 1m 26s	remaining: 9m 59s
1290:	learn: 2.1396966	test: 2.2558811	best: 2.2558811 (1290)	total: 1m 28s	remaining: 9m 57s
1320:	learn: 2.1367553	test: 2.2555225	best: 2.2555225 (1320)	total: 1m 30s	remaining: 9m 55s
1350:	learn: 2.1337406	test: 2.2550632	best: 2.2550632 (1350)	total: 1m 32s	remaining: 9m 53s
1380:	learn: 2.1309076	test: 2.2546977	best: 2.2546977 (1380)	total: 1m 34s	remaining: 9m 52s
1410:	learn: 2.1280504	test: 2.2543623	best: 2.2543623 (1410)	total: 1m 37s	remaining: 9m 52s
1440:	learn: 2.1251924	test: 2.2541133	best: 2.2541025 (1438)	total: 1m 39s	remaining: 9m 50s
1470:	learn: 2.1224183	test: 2.2537590	best: 2.2537590 (1470)	total: 1m 41s	remaining: 9m 48s
1500:	learn: 2.1195741	test: 2.2535394	best: 2.2535394 (1500)	total: 1m 43s	remaining: 9m 45s
1530:	learn: 2.1168616	test: 2.2533531	best: 2.2533531 (1530)	total: 1m 45s	remaining: 9m 44s
1560:	learn: 2.1141135	test: 2.2531627	best: 2.2531627 (1560)	total: 1m 47s	remaining: 9m 42s
1590:	learn: 2.1112906	test: 2.2528161	best: 2.2528161 (1590)	total: 1m 49s	remaining: 9m 39s
1620:	learn: 2.1083735	test: 2.2526362	best: 2.2526362 (1620)	total: 1m 51s	remaining: 9m 37s
1650:	learn: 2.1056368	test: 2.2524107	best: 2.2524016 (1641)	total: 1m 53s	remaining: 9m 35s
1680:	learn: 2.1029920	test: 2.2522367	best: 2.2522194 (1678)	total: 1m 55s	remaining: 9m 32s
1710:	learn: 2.1003603	test: 2.2519349	best: 2.2519345 (1709)	total: 1m 57s	remaining: 9m 31s
1740:	learn: 2.0977622	test: 2.2517039	best: 2.2516891 (1739)	total: 1m 59s	remaining: 9m 28s
1770:	learn: 2.0952084	test: 2.2514636	best: 2.2514558 (1764)	total: 2m 1s	remaining: 9m 26s
1800:	learn: 2.0926258	test: 2.2513560	best: 2.2513348 (1793)	total: 2m 4s	remaining: 9m 24s
1830:	learn: 2.0899984	test: 2.2509997	best: 2.2509997 (1830)	total: 2m 6s	remaining: 9m 22s
1860:	learn: 2.0876178	test: 2.2508156	best: 2.2508058 (1858)	total: 2m 8s	remaining: 9m 20s
1890:	learn: 2.0849367	test: 2.2505940	best: 2.2505940 (1890)	total: 2m 10s	remaining: 9m 20s
1920:	learn: 2.0823700	test: 2.2505036	best: 2.2505029 (1919)	total: 2m 12s	remaining: 9m 17s
1950:	learn: 2.0797567	test: 2.2502737	best: 2.2502737 (1950)	total: 2m 14s	remaining: 9m 15s
1980:	learn: 2.0771697	test: 2.2501849	best: 2.2501559 (1970)	total: 2m 16s	remaining: 9m 13s
bestTest = 2.250155885
bestIteration = 1970
Shrink model to first 1971 iterations.
array([[6.62720102e-04, 9.17315390e-02, 3.37921897e-06, ...,
        2.88443534e-01, 2.03894249e-02, 8.97947809e-03],
       [4.55542064e-04, 8.35300564e-02, 1.08853938e-06, ...,
        8.61068238e-03, 5.91659347e-02, 2.38095891e-02],
       [6.89969908e-03, 1.93533803e-01, 4.53274316e-05, ...,
        4.02361286e-02, 2.18574961e-02, 5.16495668e-03],
       ...,
       [8.48513375e-04, 1.36428296e-01, 3.08420104e-03, ...,
        5.42652446e-02, 1.82447062e-02, 1.66174739e-03],
       [8.36087676e-04, 4.51490050e-02, 6.33007050e-04, ...,
        1.85334062e-02, 1.17697316e-02, 1.55268214e-03],
       [5.89302810e-04, 2.78448388e-02, 4.06237938e-03, ...,
        3.00339220e-02, 5.03306628e-03, 2.66342980e-04]])
```
Now we see the best score is about 2.25 and there were 1971 iterations.

### Submission
```python
sub = pd.read_csv('/kaggle/input/sf-crime/sampleSubmission.csv.zip')
sub.iloc[:,1:] = output
sub.to_csv('sub.csv', index = False)
```
**sub.csv** <br/>
![Output6](https://user-images.githubusercontent.com/75198944/114529795-46f55a00-9c85-11eb-849b-68ba42e8ab2b.png) <br/>

### Score
![Output7](https://user-images.githubusercontent.com/75198944/114888470-3a6b3000-9e44-11eb-850c-ef86e00b0063.png) <br/>
![Output8](https://user-images.githubusercontent.com/75198944/114888847-8ddd7e00-9e44-11eb-8e23-a7ad9383c81d.png) <br/>

The actual score is about 0.1 higher than the predicted score we've got on the previous stage.

