import pandas as pd

df = pd.read_csv("data/transformed_df.csv", index_col=0)


df = df.drop(columns=['num__PAINTED_SAFETY_ZONE_UNITS'])


df.to_csv("data/cleaned.csv", index=False)