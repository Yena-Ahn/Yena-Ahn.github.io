---
layout: post
title:  "Pandas and Matplotlib(1)"
date:   2020-12-03 11:16:00 +0900
categories: Pyhton Pandas Matplotlib
comments: true
---
**Pandas** does have its own data visualisation tool, however, it is not enough to visualise more advanced data set. **Matplotlib** is a great tool to visualise data and it is often used with *pandas*.

##Line Plot

A line plot shows linear or curve relationships between continuous data.

Data visualisation always starts from importing modules like *pandas* and *matplotlib*. 

```python
import pandas as pd
import matplotlib.pyplot as plt
```

The data set used in this post is *Inter-Provincial Migration* data set by Statistics Korea.

```python
df = pd.read_excel('Inter-Provincial Migration.xlsx', fillna=0, header=0)

