"""
Contract clause extraction evaluation pipeline.

Generates synthetic labeled contracts, simulates three LLM extractors,
and computes a full evaluation report. Saves all outputs to data/.

Run: python pipeline.py
"""

import json
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ── Clause taxonomy ────────────────────────────────────────────────────────────

CLAUSES = {
    "limitation_of_liability": {"risk": "high",   "name": "Limitation of Liability"},
    "indemnification":         {"risk": "high",   "name": "Indemnification"},
    "ip_ownership":            {"risk": "high",   "name": "IP Ownership"},
    "data_privacy":            {"risk": "high",   "name": "Data Privacy & Security"},
    "confidentiality":         {"risk": "high",   "name": "Confidentiality"},
    "auto_renewal":            {"risk": "medium", "name": "Auto-Renewal"},
    "termination":             {"risk": "medium", "name": "Termination"},
    "dispute_resolution":      {"risk": "medium", "name": "Dispute Resolution"},
    "price_change":            {"risk": "medium", "name": "Price Change"},
    "assignment":              {"risk": "medium", "name": "Assignment"},
    "governing_law":           {"risk": "low",    "name": "Governing Law"},
    "payment_terms":           {"risk": "low",    "name": "Payment Terms"},
    "audit_rights":            {"risk": "low",    "name": "Audit Rights"},
    "non_solicitation":        {"risk": "low",    "name": "Non-Solicitation"},
}

RED_FLAGS = {
    "limitation_of_liability": ["cap below 12 months fees", "no carve-out for gross negligence"],
    "indemnification":         ["unilateral indemnification", "uncapped indemnification"],
    "ip_ownership":            ["vendor retains deliverables", "AI model trained on client data"],
    "data_privacy":            ["no DPA", "breach notification over 72 hours"],
    "confidentiality":         ["no time limit", "residuals clause"],
    "auto_renewal":            ["short cancellation window", "price increase on renewal"],
    "termination":             ["termination fee", "cure period under 15 days"],
    "dispute_resolution":      ["mandatory arbitration", "class action waiver"],
    "price_change":            ["uncapped price increase", "less than 30 days notice"],
    "assignment":              ["assignment without consent", "no change-of-control protection"],
    "governing_law":           ["foreign jurisdiction"],
    "payment_terms":           ["suspension for disputed invoices"],
    "audit_rights":            ["no audit rights"],
    "non_solicitation":        ["duration over 12 months"],
}

CONTRACT_TYPES = ["nda", "saas", "msa"]
CLAUSE_SETS = {
    "nda":  ["confidentiality", "ip_ownership", "governing_law", "non_solicitation", "termination"],
    "saas": list(CLAUSES.keys()),
    "msa":  ["limitation_of_liability", "indemnification", "confidentiality", "ip_ownership",
              "termination", "governing_law", "dispute_resolution", "payment_terms",
              "data_privacy", "audit_rights", "assignment"],
}

# ── Synthetic contract generation ──────────────────────────────────────────────

def generate_contracts(n=20):
    contracts = []
    for i in range(n):
        ctype = CONTRACT_TYPES[i % 3]
        is_risky = i >= n // 2
        clause_ids = CLAUSE_SETS[ctype]
        sections = []
        for cid in clause_ids:
            flags = []
            if is_risky and random.random() < 0.55:
                flags = random.sample(RED_FLAGS[cid], k=min(1, len(RED_FLAGS[cid])))
            sections.append({
                "clause_id": cid,
                "clause_name": CLAUSES[cid]["name"],
                "risk": CLAUSES[cid]["risk"],
                "red_flags": flags,
            })
        contracts.append({
            "id": f"{ctype.upper()}-{i+1:03d}",
            "type": ctype,
            "is_risky": is_risky,
            "sections": sections,
        })
    return contracts

# ── LLM simulation ─────────────────────────────────────────────────────────────

MODELS = {
    "gpt4_zeroshot":  {"recall": 0.76, "fp_rate": 0.07, "conf_mu": 0.80, "flag_recall": 0.65},
    "claude_fewshot": {"recall": 0.87, "fp_rate": 0.04, "conf_mu": 0.86, "flag_recall": 0.79},
    "ensemble_v1":    {"recall": 0.93, "fp_rate": 0.02, "conf_mu": 0.92, "flag_recall": 0.91},
}

# Clauses that are harder to detect
HARD = {"auto_renewal": 0.80, "price_change": 0.82, "assignment": 0.83}

# Clauses models commonly confuse
CONFUSE = [
    ("limitation_of_liability", "indemnification"),
    ("termination", "dispute_resolution"),
    ("confidentiality", "ip_ownership"),
]

def confuse(cid):
    for a, b in CONFUSE:
        if cid == a: return b
        if cid == b: return a
    return cid

def simulate(contracts, model):
    p = MODELS[model]
    results = []
    for contract in contracts:
        for sec in contract["sections"]:
            cid = sec["clause_id"]
            recall = p["recall"] * HARD.get(cid, 1.0)

            if random.random() > recall:
                results.append({"contract_id": contract["id"], "type": contract["type"],
                                 "true": cid, "pred": None, "conf": 0.0,
                                 "true_flags": sec["red_flags"], "pred_flags": [],
                                 "outcome": "fn"})
                continue

            # Miscategorization (~5% base rate)
            miscat = random.random() < 0.05
            pred_cid = confuse(cid) if miscat else cid
            conf = float(np.clip(np.random.normal(p["conf_mu"], 0.08), 0.3, 0.99))
            pred_flags = [f for f in sec["red_flags"] if random.random() < p["flag_recall"]]

            results.append({"contract_id": contract["id"], "type": contract["type"],
                             "true": cid, "pred": pred_cid, "conf": round(conf, 4),
                             "true_flags": sec["red_flags"], "pred_flags": pred_flags,
                             "outcome": "tp" if not miscat else "miscat"})

        # False positives
        if random.random() < p["fp_rate"]:
            fp_cid = random.choice(list(CLAUSES.keys()))
            results.append({"contract_id": contract["id"], "type": contract["type"],
                             "true": None, "pred": fp_cid,
                             "conf": round(float(np.random.uniform(0.45, 0.65)), 4),
                             "true_flags": [], "pred_flags": [], "outcome": "fp"})
    return results

# ── Evaluation ─────────────────────────────────────────────────────────────────

RISK_W = {"high": 3.0, "medium": 2.0, "low": 1.0}

def evaluate(results):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    for r in results:
        if r["outcome"] == "fn":
            fn[r["true"]] += 1
        elif r["outcome"] == "fp":
            fp[r["pred"]] += 1
        elif r["outcome"] == "tp":
            tp[r["true"]] += 1
        else:  # miscat
            fp[r["pred"]] += 1
            fn[r["true"]] += 1

    per_clause = {}
    wf1_num = wf1_den = 0.0
    all_f1 = []

    for cid, meta in CLAUSES.items():
        p_ = tp[cid] / (tp[cid] + fp[cid]) if (tp[cid] + fp[cid]) else 0.0
        r_ = tp[cid] / (tp[cid] + fn[cid]) if (tp[cid] + fn[cid]) else 0.0
        f1 = 2*p_*r_/(p_+r_) if (p_+r_) else 0.0
        support = tp[cid] + fn[cid]
        w = RISK_W[meta["risk"]]
        per_clause[cid] = {"name": meta["name"], "risk": meta["risk"],
                            "precision": round(p_,4), "recall": round(r_,4),
                            "f1": round(f1,4), "tp": tp[cid], "fp": fp[cid],
                            "fn": fn[cid], "support": support}
        if support:
            all_f1.append(f1)
            wf1_num += f1 * w * support
            wf1_den += w * support

    # Red flag metrics
    rf_tp = rf_fp = rf_fn = 0
    for r in results:
        if r["outcome"] in ("tp", "miscat"):
            t = set(r["true_flags"]); p = set(r["pred_flags"])
            rf_tp += len(t & p); rf_fp += len(p - t); rf_fn += len(t - p)
    rf_prec = rf_tp/(rf_tp+rf_fp) if (rf_tp+rf_fp) else 0.0
    rf_rec  = rf_tp/(rf_tp+rf_fn) if (rf_tp+rf_fn) else 0.0
    rf_f1   = 2*rf_prec*rf_rec/(rf_prec+rf_rec) if (rf_prec+rf_rec) else 0.0

    # Calibration (ECE)
    confs = [r["conf"] for r in results if r["conf"] > 0]
    correct = [1 if r["outcome"]=="tp" else 0 for r in results if r["conf"] > 0]
    ece = _ece(confs, correct)

    return {
        "per_clause": per_clause,
        "macro_precision": round(float(np.mean([v["precision"] for v in per_clause.values()])),4),
        "macro_recall":    round(float(np.mean([v["recall"]    for v in per_clause.values()])),4),
        "macro_f1":        round(float(np.mean(all_f1)),4) if all_f1 else 0.0,
        "risk_weighted_f1": round(wf1_num/wf1_den,4) if wf1_den else 0.0,
        "red_flag_precision": round(rf_prec,4),
        "red_flag_recall":    round(rf_rec,4),
        "red_flag_f1":        round(rf_f1,4),
        "ece": round(ece,4),
        "total_tp": sum(tp.values()), "total_fp": sum(fp.values()), "total_fn": sum(fn.values()),
    }

def _ece(confs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        mask = [(bins[i] <= c < bins[i+1]) for c in confs]
        if not any(mask): continue
        idxs = [j for j,m in enumerate(mask) if m]
        avg_conf = np.mean([confs[j] for j in idxs])
        avg_acc  = np.mean([labels[j] for j in idxs])
        ece += len(idxs)/n * abs(avg_conf - avg_acc)
    return float(ece)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    Path("data").mkdir(exist_ok=True)

    contracts = generate_contracts(n=20)
    with open("data/contracts.json","w") as f: json.dump(contracts,f,indent=2)
    print(f"Generated {len(contracts)} contracts")

    all_results = {}
    eval_summary = {}

    for model in MODELS:
        random.seed(42); np.random.seed(42)
        preds = simulate(contracts, model)
        all_results[model] = preds
        metrics = evaluate(preds)
        eval_summary[model] = metrics

        print(f"  {model:20s}  F1={metrics['macro_f1']:.3f}  "
              f"RW-F1={metrics['risk_weighted_f1']:.3f}  "
              f"RedFlag-F1={metrics['red_flag_f1']:.3f}  "
              f"ECE={metrics['ece']:.3f}")

    with open("data/results.json","w") as f:
        json.dump({"predictions": all_results, "metrics": eval_summary}, f, indent=2)
    print("Saved to data/results.json")

if __name__ == "__main__":
    main()
