import argparse
import json
from pathlib import Path
from fl_testbed.core import make_clients, centralized_baseline, local_only_baseline, fedavg
from fl_testbed.reporting import write_html_report


def main():
    parser = argparse.ArgumentParser(description="Federated learning under intermittent client availability")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--samples-per-client", type=int, default=180)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--availability", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=0.15)
    parser.add_argument("--secure-aggregation", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0 < args.availability <= 1:
        parser.error("--availability must be in (0, 1]")

    clients = make_clients(args.clients, args.samples_per_client, args.features, args.seed)
    centralized = centralized_baseline(clients, args.features)
    local_only = local_only_baseline(clients, args.features)
    federated, history = fedavg(
        clients, args.features, rounds=args.rounds,
        availability=args.availability, seed=args.seed, clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        secure_aggregation=args.secure_aggregation,
    )
    report = {
        "configuration": vars(args),
        "centralized": centralized,
        "local_only": local_only,
        "fedavg": federated,
        "round_history": history,
    }
    path = Path("results/latest_results.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path = write_html_report(report)
    print(json.dumps({k: v for k, v in report.items() if k != "round_history"}, indent=2))
    print(f"\nSaved complete history to {path}")
    print(f"Saved plain-English report to {html_path}")


if __name__ == "__main__":
    main()
