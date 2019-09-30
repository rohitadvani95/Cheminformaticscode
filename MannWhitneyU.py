import pandas as pd

import scipy as sp 
import numpy as np
import scipy.stats as stats

import csv

with open(r'C:\Users\INSERTFILEPATH.csv', 'rU') as infile:
  reader = csv.DictReader(infile)
  data = {}
  for row in reader:
    for header, value in row.items():
      try:
        data[header].append(value)
      except KeyError:
        data[header] = [value]

group1 = data['MW Positive Skin']
group2 = data['MW Negative Skin']



u_statistic, pVal = stats.mannwhitneyu(group1, group2)

print ('P value:')
print (pVal) 


