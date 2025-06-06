import matplotlib.pyplot as plt
import numpy as np

def plot_changes(original, counterfactual, feature_names, threshold=0.01):
    changes = [(i, name, original[i], counterfactual[i])
               for i, name in enumerate(feature_names)
               if abs(original[i] - counterfactual[i]) > threshold]

    if not changes:
        print("No significant changes to plot.")
        return

    _, names, orig_vals, cf_vals = zip(*changes)
    x = np.arange(len(names))
    bar_width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width/2, orig_vals, width=bar_width, label='Original')
    plt.bar(x + bar_width/2, cf_vals, width=bar_width, label='Counterfactual')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Feature Value')
    plt.title('Original vs Counterfactual Changes')
    plt.legend()
    plt.tight_layout()
    plt.show()
