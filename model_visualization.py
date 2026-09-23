import pandas as pd
import matplotlib.pyplot as plt


# Read model results
decision_tree = pd.read_csv("./results/decision_tree_results.csv")
random_forest = pd.read_csv("./results/random_forest_results.csv")
naive_bayes = pd.read_csv("./results/naive_bayes_results.csv")
ann = pd.read_csv("./results/ann_results.csv")


# Combine all model results
results = pd.concat(
    [
        decision_tree,
        random_forest,
        naive_bayes,
        ann
    ],
    ignore_index=True
)


# Convert metrics to percentages
results['Accuracy'] = results['Accuracy'] * 100
results['Precision'] = results['Precision'] * 100
results['Recall'] = results['Recall'] * 100
results['F1-Score'] = results['F1-Score'] * 100


# Dataset names
datasets = [
    'bank.csv',
    'bank-full.csv'
]


# -----------------------------
# Accuracy Comparison
# -----------------------------

for dataset_name in datasets:

    dataset_results = results[
        results['Dataset'] == dataset_name
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        dataset_results['Model'],
        dataset_results['Accuracy']
    )

    plt.title(
        f"Model Accuracy Comparison - {dataset_name}"
    )

    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)

    for i, value in enumerate(dataset_results['Accuracy']):
        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha='center'
        )

    plt.tight_layout()

    plt.savefig(
        f"./results/{dataset_name.replace('.csv', '')}_accuracy_comparison.png",
        dpi=300
    )

    plt.show()


# -----------------------------
# Precision Comparison
# -----------------------------

for dataset_name in datasets:

    dataset_results = results[
        results['Dataset'] == dataset_name
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        dataset_results['Model'],
        dataset_results['Precision']
    )

    plt.title(
        f"Model Precision Comparison - {dataset_name}"
    )

    plt.xlabel("Model")
    plt.ylabel("Precision (%)")
    plt.ylim(0, 100)

    for i, value in enumerate(dataset_results['Precision']):
        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha='center'
        )

    plt.tight_layout()

    plt.savefig(
        f"./results/{dataset_name.replace('.csv', '')}_precision_comparison.png",
        dpi=300
    )

    plt.show()


# -----------------------------
# Recall Comparison
# -----------------------------

for dataset_name in datasets:

    dataset_results = results[
        results['Dataset'] == dataset_name
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        dataset_results['Model'],
        dataset_results['Recall']
    )

    plt.title(
        f"Model Recall Comparison - {dataset_name}"
    )

    plt.xlabel("Model")
    plt.ylabel("Recall (%)")
    plt.ylim(0, 100)

    for i, value in enumerate(dataset_results['Recall']):
        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha='center'
        )

    plt.tight_layout()

    plt.savefig(
        f"./results/{dataset_name.replace('.csv', '')}_recall_comparison.png",
        dpi=300
    )

    plt.show()


# -----------------------------
# F1-Score Comparison
# -----------------------------

for dataset_name in datasets:

    dataset_results = results[
        results['Dataset'] == dataset_name
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        dataset_results['Model'],
        dataset_results['F1-Score']
    )

    plt.title(
        f"Model F1-Score Comparison - {dataset_name}"
    )

    plt.xlabel("Model")
    plt.ylabel("F1-Score (%)")
    plt.ylim(0, 100)

    for i, value in enumerate(dataset_results['F1-Score']):
        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha='center'
        )

    plt.tight_layout()

    plt.savefig(
        f"./results/{dataset_name.replace('.csv', '')}_f1_comparison.png",
        dpi=300
    )

    plt.show()


# -----------------------------
# Precision vs Recall
# -----------------------------

for dataset_name in datasets:

    dataset_results = results[
        results['Dataset'] == dataset_name
    ]

    x = range(len(dataset_results))

    plt.figure(figsize=(9, 5))

    plt.bar(
        [i - 0.2 for i in x],
        dataset_results['Precision'],
        width=0.4,
        label='Precision'
    )

    plt.bar(
        [i + 0.2 for i in x],
        dataset_results['Recall'],
        width=0.4,
        label='Recall'
    )

    plt.xticks(
        x,
        dataset_results['Model']
    )

    plt.title(
        f"Precision vs Recall - {dataset_name}"
    )

    plt.xlabel("Model")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"./results/{dataset_name.replace('.csv', '')}_precision_recall.png",
        dpi=300
    )

    plt.show()


# -----------------------------
# Print Final Results
# -----------------------------

print("\nFinal Model Results:")

print(
    results[
        [
            'Dataset',
            'Model',
            'Accuracy',
            'Precision',
            'Recall',
            'F1-Score'
        ]
    ].to_string(index=False)
)