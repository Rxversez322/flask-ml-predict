import pandas as pd

df = pd.read_csv(
    "AI/heart.data.txt",
    sep=",",
    na_values="?"
)

print(df.shape)
print(df.head())
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

print(df["target"].value_counts())

df.to_csv("heartbinary.csv", index=False)
