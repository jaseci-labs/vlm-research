import pandas as pd

df = pd.read_csv("wandb_export_2025-07-16T18_40_18.647+05_30.csv")

column_names = ["inference_time",
                "memory_usage",
                "score",
                ]

for column in column_names:
    valid_values = pd.to_numeric(df[column], errors='coerce').dropna()

    if len(valid_values) == 0:
        print("No numeric data to calculate average.")
    else:
        average = valid_values.mean()
        print(f"Average of '{column}': {average}")
