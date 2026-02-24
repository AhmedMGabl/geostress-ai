"""Focused test for v3.30.0 endpoints: Model Leaderboard + Data Augmentation Advisor."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8103"


def api(method, path, body=None, timeout=120):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {label}" + (f" ({detail})" if detail else ""))
        passed += 1
    else:
        print(f"  FAIL: {label}" + (f" ({detail})" if detail else ""))
        failed += 1


# ── [100] Model Performance Leaderboard ─────────────────
print("\n[100] Model Performance Leaderboard")
lb = api("POST", "/api/analysis/model-leaderboard", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", lb is not None)
check("Has well", lb.get("well") == "3P")
check("Has n_samples", isinstance(lb.get("n_samples"), int) and lb["n_samples"] > 0)
check("Has n_models", isinstance(lb.get("n_models"), int) and lb["n_models"] >= 4)
check("Has n_classes", isinstance(lb.get("n_classes"), int) and lb["n_classes"] >= 2)
check("Has class_names", isinstance(lb.get("class_names"), list))
check("Has leaderboard", isinstance(lb.get("leaderboard"), list) and len(lb["leaderboard"]) >= 4)

m0 = lb["leaderboard"][0]
check("M has model", isinstance(m0.get("model"), str))
check("M has rank", m0.get("rank") == 1)
check("M has accuracy", isinstance(m0.get("accuracy"), (int, float)))
check("M has balanced_accuracy", isinstance(m0.get("balanced_accuracy"), (int, float)) and m0["balanced_accuracy"] > 0)
check("M has mean_f1", isinstance(m0.get("mean_f1"), (int, float)))
check("M has mean_confidence", m0.get("mean_confidence") is None or isinstance(m0["mean_confidence"], (int, float)))
check("M has per_class", isinstance(m0.get("per_class"), list) and len(m0["per_class"]) >= 2)

pc0 = m0["per_class"][0]
check("PC has class", isinstance(pc0.get("class"), str))
check("PC has precision", isinstance(pc0.get("precision"), (int, float)))
check("PC has recall", isinstance(pc0.get("recall"), (int, float)))
check("PC has f1", isinstance(pc0.get("f1"), (int, float)))
check("PC has support", isinstance(pc0.get("support"), int))

check("M has worst_class", isinstance(m0.get("worst_class"), str))
check("M has worst_f1", isinstance(m0.get("worst_f1"), (int, float)))

check("Has agreed_features", isinstance(lb.get("agreed_features"), list))
if lb["agreed_features"]:
    af0 = lb["agreed_features"][0]
    check("AF has feature", isinstance(af0.get("feature"), str))
    check("AF has avg_importance", isinstance(af0.get("avg_importance"), (int, float)))
    check("AF has importances", isinstance(af0.get("importances"), dict))

check("Has recommendations", isinstance(lb.get("recommendations"), list) and len(lb["recommendations"]) > 0)
check("Has plot", isinstance(lb.get("plot"), str) and len(lb["plot"]) > 100)
check("Has stakeholder_brief", isinstance(lb.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(lb["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", lb["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# Sorted correctly (rank 1 should have highest balanced_accuracy)
if len(lb["leaderboard"]) >= 2:
    check("Sorted correctly", lb["leaderboard"][0]["balanced_accuracy"] >= lb["leaderboard"][1]["balanced_accuracy"])


# ── [101] Data Augmentation Advisor ─────────────────────
print("\n[101] Data Augmentation Advisor")
da = api("POST", "/api/analysis/data-augmentation-advisor", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", da is not None)
check("Has well", da.get("well") == "3P")
check("Has n_samples", isinstance(da.get("n_samples"), int) and da["n_samples"] > 0)
check("Has n_classes", isinstance(da.get("n_classes"), int))
check("Has class_analysis", isinstance(da.get("class_analysis"), list) and len(da["class_analysis"]) >= 2)

ca0 = da["class_analysis"][0]
check("CA has class", isinstance(ca0.get("class"), str))
check("CA has count", isinstance(ca0.get("count"), int))
check("CA has pct_of_total", isinstance(ca0.get("pct_of_total"), (int, float)))
check("CA has ratio_to_mean", isinstance(ca0.get("ratio_to_mean"), (int, float)))
check("CA has status", ca0.get("status") in ("ADEQUATE", "UNDERSAMPLED", "CRITICAL"))
check("CA has additional_needed", isinstance(ca0.get("additional_needed"), int))

check("Has n_undersampled", isinstance(da.get("n_undersampled"), int))
check("Has depth_gaps", isinstance(da.get("depth_gaps"), list))
check("Has smote_analysis", da.get("smote_analysis") is not None)

if isinstance(da.get("smote_analysis"), dict) and "baseline_accuracy" in da["smote_analysis"]:
    sm = da["smote_analysis"]
    check("SMOTE has baseline", isinstance(sm.get("baseline_accuracy"), (int, float)))
    check("SMOTE has augmented", isinstance(sm.get("augmented_accuracy"), (int, float)))
    check("SMOTE has improvement", isinstance(sm.get("improvement"), (int, float)))
    check("SMOTE has recommended", isinstance(sm.get("recommended"), bool))
    check("SMOTE has n_synthetic", isinstance(sm.get("n_synthetic_added"), int))

check("Has recommendations", isinstance(da.get("recommendations"), list) and len(da["recommendations"]) > 0)
r0 = da["recommendations"][0]
check("Rec has priority", r0.get("priority") in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
check("Rec has category", isinstance(r0.get("category"), str))
check("Rec has action", isinstance(r0.get("action"), str))
check("Rec has impact", isinstance(r0.get("impact"), str))

check("Has plot", isinstance(da.get("plot"), str) and len(da["plot"]) > 100)
check("Has stakeholder_brief", isinstance(da.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(da["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", da["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# Check negative examples recommendation exists
rec_cats = [r["category"] for r in da["recommendations"]]
check("Has negative examples rec", any("Negative" in c for c in rec_cats))


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.30.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
