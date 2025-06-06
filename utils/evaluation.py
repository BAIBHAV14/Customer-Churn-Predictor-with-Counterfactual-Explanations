import numpy as np

def summarize_changes(original, counterfactual, feature_names, threshold=0.01):
    diffs = []
    for i, (o, c) in enumerate(zip(original, counterfactual)):
        if abs(o - c) > threshold:
            name = feature_names[i]
            diffs.append(f"{name}: {o:.2f} → {c:.2f} (Δ={c - o:.2f})")
    return diffs

def compute_proximity_metrics(original, counterfactual, feature_names):
    l1_distance = np.sum(np.abs(original - counterfactual))
    binary_changes = 0
    for i, name in enumerate(feature_names):
        if 'Gender_' in name or 'MaritalStatus_' in name or 'Preferred' in name:
            if original[i] != counterfactual[i]:
                binary_changes += 1
    return l1_distance, binary_changes

def evaluate_counterfactual(original, counterfactual, feature_names, feature_mins, feature_maxs, immutable_features):
    sparsity = np.sum(np.abs(original - counterfactual) > 0.01)
    realism = np.all((counterfactual >= feature_mins) & (counterfactual <= feature_maxs))

    immutable_indices = [i for i, f in enumerate(feature_names) if f in immutable_features]
    actionability = all(abs(original[i] - counterfactual[i]) <= 0.01 for i in immutable_indices)

    return {
        "Sparsity": sparsity,
        "Realistic?": realism,
        "Actionable?": actionability
    }
