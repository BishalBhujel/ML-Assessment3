from data_preprocessing import datasets

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import matplotlib.pyplot as plt


# Dataset names
dataset_names = [
    'bank.csv',
    'bank-full.csv'
]


for dataset_name in dataset_names:

    print(f"\nDataset: {dataset_name}")

    # Get training data
    X_train = datasets[dataset_name]['X_train']
    y_train = datasets[dataset_name]['y_train']

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Read original dataset
    df = pd.read_csv(
        f"datasets/{dataset_name}",
        sep=';'
    )

    # Get feature names after encoding
    X = df.drop('y', axis=1)

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    feature_names = X.columns

    # Create feature importance dataframe
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })

    # Sort by importance
    importance = importance.sort_values(
        by='Importance',
        ascending=False
    )

    # Select top 10 features
    top_features = importance.head(10)

    # Print top 10 features
    print("\nTop 10 Important Features:")

    print(
        top_features.to_string(index=False)
    )

    # -----------------------------
    # Feature Importance Graph
    # -----------------------------

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features['Feature'][::-1],
        top_features['Importance'][::-1]
    )

    plt.title(
        f"Top 10 Feature Importance - {dataset_name}"
    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.tight_layout()

    # Save graph
    plt.savefig(
        f"./results/{dataset_name.replace('.csv', '')}_feature_importance.png",
        dpi=300
    )

    plt.show()