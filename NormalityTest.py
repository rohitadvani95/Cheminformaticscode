import pandas as pd
from pandas import DataFrame
import scipy as sp 
import numpy as np
import scipy.stats as stats
import operator
import csv
from csv import DictReader
import re

import scipy as sp

from scipy.stats import anderson
import time

with open(r'C:\Users\kirby\OneDrive\Documents\Finalskindata.csv', 'rU') as infile:
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
            for header, value in row.items():
              try:
                data[header].append(value)
              except KeyError:
                data[header] = [value]
                

group1 = data['excurineneg']

print(group1)
groupstrat = []
for data['excurineneg'] in group1:

        if (len(data['excurineneg'])) > 1:
                groupstrat.append(float(data['excurineneg']))
                


print(groupstrat)


NormTest = sp.stats.normaltest(groupstrat, axis=0, nan_policy='propagate')


print(NormTest)


 
