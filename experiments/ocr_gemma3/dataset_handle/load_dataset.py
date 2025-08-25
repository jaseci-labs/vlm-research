import pandas as pd

# need to laod the csv files
train_df = pd.read_csv("nanonets_kie_splits/train.csv")

print("Train Data:")
print(train_df.head(1))