from data_preprocessing import datasets

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd


results = []


for dataset_name in datasets:

    print(f"\nDataset: {dataset_name}")

    X_train = datasets[dataset_name]['X_train']
    X_test = datasets[dataset_name]['X_test']
    y_train = datasets[dataset_name]['y_train']
    y_test = datasets[dataset_name]['y_test']

    # Create ANN model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Make predictions
    y_pred = model.predict(
        X_test,
        verbose=0
    )

    y_pred = (y_pred > 0.5).astype(int)

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
        'Model': 'ANN',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })


# Save results to CSV
results_df = pd.DataFrame(results)

results_df.to_csv(
    "./results/ann_results.csv",
    index=False
)