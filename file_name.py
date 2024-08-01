import os
import pandas as pd

old_data = 'data\test2.csv'

old_df = pd.read_csv(old_data)
old_df.info()