from data_preprocessing import datasets

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd


results = []


for dataset_name in datasets:

    print(f"\nDataset: {dataset_name}")

    X_train = datasets[dataset_name]['X_train']
    X_test = datasets[dataset_name]['X_test']
    y_train = datasets[dataset_name]['y_train']
    y_test = datasets[dataset_name]['y_test']

    # Create Naïve Bayes model
    model = GaussianNB()

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    # Store results
    results.append({
        'Dataset': dataset_name,
        'Model': 'Naïve Bayes',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })


# Save results to CSV
results_df = pd.DataFrame(results)

results_df.to_csv(
    "./results/naive_bayes_results.csv",
    index=False
)