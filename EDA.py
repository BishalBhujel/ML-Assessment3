import os
import pandas as pd

dataset_path = "/Users/bishalbhujel/Desktop/ML_Assessment3/datasets"
print(os.listdir(dataset_path))

for file_name in os.listdir(dataset_path):
    df = pd.read_csv(os.path.join(dataset_path, file_name), sep=';')
    print(f"\nDataset: {file_name}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.to_list())
    print("Data types:")
    print(df.dtypes)
    print("\nTarget distribution:")
    print(df['y'].value_counts())