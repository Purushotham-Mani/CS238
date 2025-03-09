import pandas as pd

samples = pd.read_csv(".\medium.csv").to_numpy()

print(samples.shape)