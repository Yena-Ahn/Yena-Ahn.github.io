---
layout: post
title:  "Two Sigma Connect Rental Listing Inquiries"
date:   2021-05-08 14:10:00 +0900
categories: Python Pandas 
comments: true
---

This challenge is to predict how popular an apartment rental listing is based on information given like photos, description, price, etc. There are three levels of interest - low, medium, and high - that we will need to predict and submit.

### Preprocessing

Given file directories are:
```python
/kaggle/input/two-sigma-connect-rental-listing-inquiries/images_sample.zip
/kaggle/input/two-sigma-connect-rental-listing-inquiries/Kaggle-renthop.torrent
/kaggle/input/two-sigma-connect-rental-listing-inquiries/sample_submission.csv.zip
/kaggle/input/two-sigma-connect-rental-listing-inquiries/test.json.zip
/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json.zip
```

The files given on this challenge are *json* files, so we need to open these using *read_json* instead of *read_csv*.

```python
train = pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json.zip')
test = pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/test.json.zip')
```

```python
display(train, test)
```
<!--- 
output has to be added here --->

Combine these two data sets into *alldata*.

```python
alldata = pd.concat([train, test])
alldata
```
<!--- output here>



