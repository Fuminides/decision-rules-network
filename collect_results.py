import pandas as pd
import os
results_folder = 'results/'

results = pd.DataFrame()
for file in os.listdir(results_folder):
    if file.endswith('.csv'):
        df = pd.read_csv(results_folder + file, sep=';', index_col=0)
        results = pd.concat([results, df], axis=0)

print(results.mean())