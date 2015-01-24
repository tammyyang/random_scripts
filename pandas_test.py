#!/usr/bin/env python
#
import sys
import os
import time
import datetime

import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

names = ['bug', 'status', 'importance', 'title', 'open', 'tag', 'GM', 'link']
Today = "07/25/2013"

indf = pd.read_csv('hychentestdb.csv', names=names, index_col=0)
data = {}
for ix in indf.index:
    WI = (lambda x: {"Critical": 1.2, "High": 1.0, "Medium": -0.2, "Low":
-0.3}.get(indf.ix[ix]['importance']))(ix)

    WH = 0.0
    for tag in indf.ix[ix]['tag']:
        if tag == 'stella-oem-highlight':
            WH = 1.1

    ttoday = time.strptime(Today, "%m/%d/%Y")
    tGM = time.strptime(indf.ix[ix]['GM'], "%m/%d/%Y")
    topen = time.strptime(indf.ix[ix]['open'].split(" ")[0], "%Y-%m-%d")
    nodays = (datetime.date(ttoday[0], ttoday[1],
ttoday[2])-datetime.date(topen[0], topen[1], topen[2])).days
    ngdays = (datetime.date(ttoday[0], ttoday[1],
ttoday[2])-datetime.date(tGM[0], tGM[1], tGM[2])).days

    WO = (lambda x : x/7*0.1 if x < 35  else
0.5)(nodays)
    WG1 = (lambda x : 0.4 if x < 7  else 0)(ngdays)
    WG2 = (lambda x : 0.2 if x < 14  else 0)(ngdays)

    data[ix] = [WI+WH+WO+WG1+WG2, indf.ix[ix]['status'], WI, WH, WO, WG1, WG2, indf.ix[ix]['link'], indf.ix[ix]['title']]

outdf = DataFrame(data, index=['HPS', 'Status', 'WI', 'WH', 'WO', 'WG1',
'WG2', 'Link', 'Title'])
outdf = outdf.T.sort_index(by='HPS', ascending = False)
outdf.to_csv('tammytestdb.csv')

#plt.plot(outdf.index, outdf['HPS'], 'ro')
plt.hist(outdf['HPS'])
#plt.xlabel('my data', fontsize=14, color='red')
plt.xlabel('HPS')
#plt.show()
plt.savefig('hps.png')
