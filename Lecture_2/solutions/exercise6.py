#!/usr/bin/env python
import pandas as pd

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                        na_values='?', comment='\t',
                        sep=' ', skipinitialspace=True)


print('Mean:')
print(raw_dataset.mean())

print("\nFilter by cylinders == 3")
r = raw_dataset[raw_dataset['Cylinders'] == 3]
print(r)
