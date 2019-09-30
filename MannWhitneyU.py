import pandas as pd

import scipy as sp 
import numpy as np
import scipy.stats as stats

import csv

# open the file in universal line ending mode to avoid "file does not exist error"

with open(r'C:\Users\kirby\OneDrive\Documents\MidJuneData.csv', 'rU') as infile:
  # read the file as a dictionary for each row ({header : value})
  reader = csv.DictReader(infile)
  data = {}
  for row in reader:
    for header, value in row.items():
      try:
        data[header].append(value)
      except KeyError:
        data[header] = [value]

# extract the variables you want
group1 = data['MW Positive Skin']
group2 = data['MW Negative Skin']



u_statistic, pVal = stats.mannwhitneyu(group1, group2)

print ('P value:')
print (pVal) 


