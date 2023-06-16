import pandas as pd

# Load the dataset
df = pd.read_csv("winequality-red.csv", delimiter=";")
#df = pd.read_csv("winequality-white.csv", delimiter=";")

# Make 'quality' the first column
cols = df.columns.tolist()
cols = [cols[-1]] + cols[:-1]
df = df[cols]

# Normalize - 0/1 
for col in df.columns[1:]:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

df.info()

df.to_csv("preprocessed_winequality-red.csv", sep=" ", index=False, header=False) 
#df.to_csv("preprocessed_winequality-white.csv", sep=" ", index=False, header=False)
