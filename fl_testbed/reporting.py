from html import escape
from pathlib import Path


def _percent(value):
    return f"{100 * value:.1f}%"


def write_html_report(report, path="results/report.html"):
    config = report["configuration"]
    fedavg = report["fedavg"]
    centralized = report["centralized"]
    local = report["local_only"]
    privacy = fedavg["privacy_mechanism"]
    accuracy_status = "good" if fedavg["accuracy"] >= local["mean_client_accuracy"] else "warn"
    fairness_gap = fedavg["mean_client_accuracy"] - fedavg["worst_client_accuracy"]
    fairness_status = "good" if fairness_gap <= 0.05 else "warn"
    attack = fedavg["membership_attack_auc"]
    attack_status = "good" if attack <= 0.55 else "warn"

    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{_percent(values['accuracy'])}</td>"
        f"<td>{_percent(values['worst_client_accuracy'])}</td></tr>"
        for name, values in [
            ("Centralized", centralized),
            ("Federated (FedAvg)", fedavg),
        ]
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Federated Learning Privacy Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:40px auto;padding:0 20px;color:#172033;background:#f6f8fb}}
h1{{color:#123b5d}} .subtitle{{font-size:1.15rem;color:#536174}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:16px;margin:24px 0}}
.card{{background:white;border-radius:12px;padding:18px;box-shadow:0 2px 10px #1b315012;border-top:5px solid #738399}}
.good{{border-top-color:#198754}} .warn{{border-top-color:#e29a16}} .value{{font-size:1.8rem;font-weight:700;margin:8px 0}}
table{{width:100%;border-collapse:collapse;background:white}} th,td{{padding:12px;text-align:left;border-bottom:1px solid #dfe5ec}}
.note{{background:#fff7df;border-left:5px solid #e29a16;padding:14px;margin:20px 0}} code{{background:#e9eef4;padding:2px 5px}}
</style></head><body>
<h1>Privacy-Enhanced Federated Learning Report</h1>
<p class="subtitle">Can organisations learn together without sharing their raw operational records?</p>
<div class="grid">
 <div class="card {accuracy_status}"><b>Federated accuracy</b><div class="value">{_percent(fedavg['accuracy'])}</div><span>Overall predictions that were correct.</span></div>
 <div class="card {fairness_status}"><b>Least-served client</b><div class="value">{_percent(fedavg['worst_client_accuracy'])}</div><span>Performance for the client receiving the weakest result.</span></div>
 <div class="card {attack_status}"><b>Privacy attack AUC</b><div class="value">{attack:.3f}</div><span>0.5 is random guessing; lower is preferable.</span></div>
 <div class="card"><b>Participation gap</b><div class="value">{fedavg['participation_gap']}</div><span>Difference between most and least active clients.</span></div>
</div>
<h2>What was tested?</h2>
<p>{config['clients']} simulated organisations trained an incident-risk model for {config['rounds']} rounds. Each organisation retained its raw data and shared only protected model updates.</p>
<table><thead><tr><th>Method</th><th>Overall accuracy</th><th>Worst-client accuracy</th></tr></thead><tbody>{rows}
<tr><td>Local only</td><td>{_percent(local['mean_client_accuracy'])} (mean)</td><td>{_percent(local['worst_client_accuracy'])}</td></tr></tbody></table>
<h2>Privacy controls</h2>
<ul><li>Update clipping norm: <code>{privacy['clipping_norm']}</code></li>
<li>Gaussian noise multiplier: <code>{privacy['noise_multiplier']}</code></li>
<li>Secure aggregation simulation: <b>{'enabled' if privacy['secure_aggregation_simulated'] else 'disabled'}</b></li></ul>
<h2>Plain-English conclusion</h2>
<p>The clients successfully produced one shared model without pooling their raw records. Overall accuracy alone is not enough: worst-client performance and participation imbalance reveal who may benefit less.</p>
<div class="note"><b>Important limitation:</b> This is an educational simulation using synthetic data. Clipping and Gaussian noise are implemented, but no formal privacy budget is claimed. Secure aggregation is simulated in one process and is not production cryptography.</div>
</body></html>"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
