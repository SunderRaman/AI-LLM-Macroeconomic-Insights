import pandas as pd

df = pd.read_csv("data/processed/wb_macro_clean.csv")
print (df['country'].value_counts())
