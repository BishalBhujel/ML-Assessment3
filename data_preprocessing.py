import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset_path = os.getcwd() + "/datasets"
datasets = {}

for file_name in os.listdir(dataset_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path, sep=';')
        print(f"\nDataset: {file_name}")

        # Convert target variable
        df['y'] = df['y'].map({'no': 0, 'yes': 1})

        # Separate features and target
        X = df.drop('y', axis=1)
        y = df['y']

        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Feature scaling
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Store processed data
        datasets[file_name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        print("Training data:", X_train.shape)
        print("Testing data:", X_test.shape)