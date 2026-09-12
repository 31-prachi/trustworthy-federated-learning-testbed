from dataclasses import dataclass
import numpy as np


FEATURE_NAMES = [
    "cpu_usage", "memory_usage", "disk_usage", "response_time_ms",
    "error_count", "failed_logins", "active_sessions", "queue_depth",
    "network_latency_ms", "restart_count",
]


@dataclass
class Client:
    client_id: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def predict(x, weights, bias):
    return (sigmoid(x @ weights + bias) >= 0.5).astype(int)


def train_logistic(x, y, weights, bias, epochs=5, lr=0.08):
    w = weights.copy()
    b = float(bias)
    n = len(y)
    for _ in range(epochs):
        probabilities = sigmoid(x @ w + b)
        error = probabilities - y
        w -= lr * (x.T @ error) / n
        b -= lr * float(error.mean())
    return w, b


def metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = float((y_true == y_pred).mean())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def make_clients(n_clients=8, samples_per_client=180, features=10, seed=42):
    """Create synthetic, non-IID operational data for independent organisations.

    Values are standardized for training. No real people or company records are used.
    """
    rng = np.random.default_rng(seed)
    true_w = rng.normal(size=features)
    clients = []
    for client_id in range(n_clients):
        shift = rng.normal(scale=1.2, size=features)
        x = rng.normal(size=(samples_per_client, features)) + shift
        class_bias = (client_id - (n_clients - 1) / 2) * 0.22
        probability = sigmoid(x @ true_w + class_bias + rng.normal(scale=0.45, size=samples_per_client))
        y = rng.binomial(1, probability)
        split = int(samples_per_client * 0.75)
        clients.append(Client(client_id, x[:split], y[:split], x[split:], y[split:]))
    return clients


def evaluate_global(clients, weights, bias):
    y_true = np.concatenate([c.y_test for c in clients])
    y_pred = np.concatenate([predict(c.x_test, weights, bias) for c in clients])
    result = metrics(y_true, y_pred)
    per_client = [metrics(c.y_test, predict(c.x_test, weights, bias))["accuracy"] for c in clients]
    result["worst_client_accuracy"] = float(min(per_client))
    result["mean_client_accuracy"] = float(np.mean(per_client))
    return result


def centralized_baseline(clients, features, epochs=80, lr=0.08):
    x = np.concatenate([c.x_train for c in clients])
    y = np.concatenate([c.y_train for c in clients])
    w, b = train_logistic(x, y, np.zeros(features), 0.0, epochs, lr)
    return evaluate_global(clients, w, b)


def local_only_baseline(clients, features, epochs=80, lr=0.08):
    scores = []
    for c in clients:
        w, b = train_logistic(c.x_train, c.y_train, np.zeros(features), 0.0, epochs, lr)
        scores.append(metrics(c.y_test, predict(c.x_test, w, b))["accuracy"])
    return {"mean_client_accuracy": float(np.mean(scores)), "worst_client_accuracy": float(min(scores))}


def _clip_update(delta_w, delta_b, clip_norm):
    vector = np.append(delta_w, delta_b)
    norm = float(np.linalg.norm(vector))
    scale = min(1.0, clip_norm / max(norm, 1e-12))
    return delta_w * scale, float(delta_b * scale), norm, scale


def _secure_aggregate_simulation(weighted_updates, rng):
    """Mask updates with cancelling pairwise masks.

    This demonstrates the secure-aggregation idea in one process. It is not a
    network protocol or production cryptographic implementation.
    """
    masked = [update.copy() for update in weighted_updates]
    for left in range(len(masked)):
        for right in range(left + 1, len(masked)):
            mask = rng.normal(0.0, 5.0, size=masked[left].shape)
            masked[left] += mask
            masked[right] -= mask
    return np.sum(masked, axis=0)


def membership_inference_auc(clients, weights, bias):
    """Illustrative confidence-based membership attack (0.5=random, 1=perfect)."""
    train_x = np.concatenate([c.x_train for c in clients])
    train_y = np.concatenate([c.y_train for c in clients])
    test_x = np.concatenate([c.x_test for c in clients])
    test_y = np.concatenate([c.y_test for c in clients])

    def confidence(x, y):
        probability = sigmoid(x @ weights + bias)
        return np.where(y == 1, probability, 1.0 - probability)

    member_scores = confidence(train_x, train_y)
    nonmember_scores = confidence(test_x, test_y)
    # Pairwise AUC avoids an external metrics dependency.
    comparisons = (member_scores[:, None] > nonmember_scores[None, :]).mean()
    ties = (member_scores[:, None] == nonmember_scores[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def fedavg(clients, features, rounds=40, availability=0.65, min_clients=2,
           local_epochs=5, lr=0.08, seed=42, clip_norm=1.0,
           noise_multiplier=0.0, secure_aggregation=False, return_model=False):
    rng = np.random.default_rng(seed)
    privacy_rng = np.random.default_rng(seed + 99_991)
    w = np.zeros(features)
    b = 0.0
    participation = np.zeros(len(clients), dtype=int)
    history = []
    communication_bytes = 0
    for round_id in range(1, rounds + 1):
        available = [c for c in clients if rng.random() < availability]
        if len(available) < min_clients:
            ranked = rng.permutation(clients).tolist()
            available = ranked[:min_clients]
        updates = []
        unclipped_norms = []
        clipped_updates = 0
        for c in available:
            local_w, local_b = train_logistic(c.x_train, c.y_train, w, b, local_epochs, lr)
            delta_w, delta_b, norm, scale = _clip_update(
                local_w - w, local_b - b, clip_norm
            )
            updates.append((len(c.y_train), delta_w, delta_b))
            unclipped_norms.append(norm)
            clipped_updates += int(scale < 1.0)
            participation[c.client_id] += 1
            communication_bytes += 2 * (features + 1) * 8  # download and upload, float64
        total = sum(size for size, _, _ in updates)
        vectors = [size * np.append(delta_w, delta_b)
                   for size, delta_w, delta_b in updates]
        if secure_aggregation:
            mask_rng = np.random.default_rng(seed * 10_000 + round_id)
            aggregate = _secure_aggregate_simulation(vectors, mask_rng) / total
        else:
            aggregate = np.sum(vectors, axis=0) / total
        if noise_multiplier > 0:
            noise_std = noise_multiplier * clip_norm / len(updates)
            aggregate += privacy_rng.normal(0.0, noise_std, size=aggregate.shape)
        w += aggregate[:-1]
        b += float(aggregate[-1])
        score = evaluate_global(clients, w, b)
        score.update({
            "round": round_id,
            "participants": len(available),
            "clipped_updates": clipped_updates,
            "mean_unclipped_update_norm": float(np.mean(unclipped_norms)),
        })
        history.append(score)
    final = history[-1].copy()
    final.update({
        "communication_bytes": int(communication_bytes),
        "participation_counts": participation.tolist(),
        "participation_gap": int(participation.max() - participation.min()),
        "privacy_mechanism": {
            "clipping_norm": float(clip_norm),
            "noise_multiplier": float(noise_multiplier),
            "secure_aggregation_simulated": bool(secure_aggregation),
            "formal_privacy_accounting": False,
        },
        "membership_attack_auc": membership_inference_auc(clients, w, b),
    })
    if return_model:
        return final, history, (w.copy(), float(b))
    return final, history
