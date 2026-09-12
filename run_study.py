"""Run an availability/privacy study across multiple random seeds."""
import json
from pathlib import Path
import numpy as np

from fl_testbed.core import make_clients, centralized_baseline, local_only_baseline, fedavg

AVAILABILITY_LEVELS = [0.30, 0.50, 0.65, 0.80, 1.00]
SEEDS = [7, 21, 42, 84, 101]


def main():
    records = []
    for availability in AVAILABILITY_LEVELS:
        for seed in SEEDS:
            clients = make_clients(seed=seed)
            central = centralized_baseline(clients, 10)
            local = local_only_baseline(clients, 10)
            standard, _ = fedavg(
                clients, 10, availability=availability, seed=seed,
                clip_norm=1.0, noise_multiplier=0.0,
                secure_aggregation=False,
            )
            private, _ = fedavg(
                clients, 10, availability=availability, seed=seed,
                clip_norm=1.0, noise_multiplier=0.15,
                secure_aggregation=True,
            )
            records.append({
                "availability": availability,
                "seed": seed,
                "centralized_accuracy": central["accuracy"],
                "local_mean_accuracy": local["mean_client_accuracy"],
                "standard_fedavg_accuracy": standard["accuracy"],
                "standard_attack_auc": standard["membership_attack_auc"],
                "private_fedavg_accuracy": private["accuracy"],
                "worst_client_accuracy": private["worst_client_accuracy"],
                "membership_attack_auc": private["membership_attack_auc"],
                "participation_gap": private["participation_gap"],
            })

    summary = []
    for availability in AVAILABILITY_LEVELS:
        group = [r for r in records if r["availability"] == availability]
        row = {"availability": availability, "runs": len(group)}
        for key in ["standard_fedavg_accuracy", "standard_attack_auc",
                    "private_fedavg_accuracy", "worst_client_accuracy",
                    "membership_attack_auc", "participation_gap"]:
            values = np.array([r[key] for r in group], dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1))
        summary.append(row)

    output = {"seeds": SEEDS, "individual_runs": records, "summary": summary}
    path = Path("results/multi_seed_study.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Availability | Standard acc | Protected acc | Worst client | Attack AUC standard -> protected")
    for row in summary:
        print(
            f"{row['availability']:>11.0%} | "
            f"{row['standard_fedavg_accuracy_mean']:.3f} | "
            f"{row['private_fedavg_accuracy_mean']:.3f} | "
            f"{row['worst_client_accuracy_mean']:.3f} | "
            f"{row['standard_attack_auc_mean']:.3f} -> {row['membership_attack_auc_mean']:.3f}"
        )
    print(f"\nSaved study evidence to {path}")


if __name__ == "__main__":
    main()
