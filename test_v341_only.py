"""Focused test for v3.41.0 endpoints: Feature Ranking + Cluster Stability + Well Similarity + Prediction Timeline + Augmentation Preview."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8125"


def api(method, path, body=None, timeout=300):
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


# ── [140] Feature Importance Ranking ────────────────────
print("\n[140] Feature Importance Ranking")
fr = api("POST", "/api/analysis/feature-ranking", {"source": "demo", "well": "3P"})
check("Status 200", fr is not None)
check("Has well", fr.get("well") == "3P")
check("Has n_features", isinstance(fr.get("n_features"), int) and fr["n_features"] >= 10)
check("Has n_samples", isinstance(fr.get("n_samples"), int) and fr["n_samples"] > 0)
check("Has n_classes", isinstance(fr.get("n_classes"), int) and fr["n_classes"] >= 2)
check("Has features list", isinstance(fr.get("features"), list) and len(fr["features"]) >= 10)
f0 = fr["features"][0]
check("Feature has name", isinstance(f0.get("feature"), str))
check("Feature has consensus_score", isinstance(f0.get("consensus_score"), (int, float)))
check("Feature has rf_importance", isinstance(f0.get("rf_importance"), (int, float)))
check("Feature has permutation_importance", isinstance(f0.get("permutation_importance"), (int, float)))
check("Feature has correlation", isinstance(f0.get("correlation"), (int, float)))
check("Feature has agreement", isinstance(f0.get("agreement"), int) and 0 <= f0["agreement"] <= 3)
check("Has top_5", isinstance(fr.get("top_5"), list) and len(fr["top_5"]) == 5)
check("Has recommendations", isinstance(fr.get("recommendations"), list) and len(fr["recommendations"]) > 0)
check("Has plot", isinstance(fr.get("plot"), str) and len(fr["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fr.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(fr["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(fr.get("elapsed_s"), (int, float)))

# 6P test
fr2 = api("POST", "/api/analysis/feature-ranking", {"source": "demo", "well": "6P"})
check("6P works", fr2 is not None and fr2.get("well") == "6P")


# ── [141] Cluster Stability Analysis ───────────────────
print("\n[141] Cluster Stability Analysis")
cs = api("POST", "/api/analysis/cluster-stability", {"source": "demo", "well": "3P", "k_max": 8})
check("Status 200", cs is not None)
check("Has well", cs.get("well") == "3P")
check("Has best_k", isinstance(cs.get("best_k"), int) and cs["best_k"] >= 2)
check("Has best_silhouette", isinstance(cs.get("best_silhouette"), (int, float)))
check("Has elbow_k", isinstance(cs.get("elbow_k"), int))
check("Has stability", cs.get("stability") in ("HIGH", "MODERATE", "LOW"))
check("Has k_results", isinstance(cs.get("k_results"), list) and len(cs["k_results"]) >= 2)
kr0 = cs["k_results"][0]
check("Result has k", isinstance(kr0.get("k"), int))
check("Result has silhouette", isinstance(kr0.get("silhouette"), (int, float)))
check("Result has inertia", isinstance(kr0.get("inertia"), (int, float)))
check("Has recommendations", isinstance(cs.get("recommendations"), list) and len(cs["recommendations"]) > 0)
check("Has plot", isinstance(cs.get("plot"), str) and len(cs["plot"]) > 100)
check("Has stakeholder_brief", isinstance(cs.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(cs["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(cs.get("elapsed_s"), (int, float)))

# 6P test
cs2 = api("POST", "/api/analysis/cluster-stability", {"source": "demo", "well": "6P"})
check("6P works", cs2 is not None and cs2.get("well") == "6P")


# ── [142] Well Similarity Matrix ───────────────────────
print("\n[142] Well Similarity Matrix")
ws = api("POST", "/api/analysis/well-similarity", {"source": "demo"})
check("Status 200", ws is not None)
check("Has n_wells", isinstance(ws.get("n_wells"), int) and ws["n_wells"] >= 2)
check("Has wells list", isinstance(ws.get("wells"), list) and len(ws["wells"]) >= 2)
check("Has pairs", isinstance(ws.get("pairs"), list) and len(ws["pairs"]) >= 1)
p0 = ws["pairs"][0]
check("Pair has well_a", isinstance(p0.get("well_a"), str))
check("Pair has well_b", isinstance(p0.get("well_b"), str))
check("Pair has similarity", isinstance(p0.get("similarity"), (int, float)) and 0 <= p0["similarity"] <= 1)
check("Pair has avg_wasserstein", isinstance(p0.get("avg_wasserstein"), (int, float)))
check("Pair has type_overlap", isinstance(p0.get("type_overlap"), str))
check("Pair has can_share_model", isinstance(p0.get("can_share_model"), bool))
check("Has similarity_matrix", isinstance(ws.get("similarity_matrix"), list))
check("Has recommendations", isinstance(ws.get("recommendations"), list) and len(ws["recommendations"]) > 0)
check("Has plot", isinstance(ws.get("plot"), str) and len(ws["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ws.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(ws["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(ws.get("elapsed_s"), (int, float)))


# ── [143] Prediction Timeline ──────────────────────────
print("\n[143] Prediction Timeline")
pt = api("POST", "/api/analysis/prediction-timeline", {"source": "demo", "well": "3P"})
check("Status 200", pt is not None)
check("Has well", pt.get("well") == "3P")
check("Has n_samples", isinstance(pt.get("n_samples"), int) and pt["n_samples"] > 0)
check("Has mean_confidence", isinstance(pt.get("mean_confidence"), (int, float)) and 0 <= pt["mean_confidence"] <= 1)
check("Has accuracy_pct", isinstance(pt.get("accuracy_pct"), (int, float)))
check("Has n_low_confidence", isinstance(pt.get("n_low_confidence"), int))
check("Has n_high_confidence", isinstance(pt.get("n_high_confidence"), int))
check("Has timeline", isinstance(pt.get("timeline"), list) and len(pt["timeline"]) > 0)
t0_entry = pt["timeline"][0]
check("Entry has predicted_class", isinstance(t0_entry.get("predicted_class"), str))
check("Entry has true_class", isinstance(t0_entry.get("true_class"), str))
check("Entry has confidence", isinstance(t0_entry.get("confidence"), (int, float)))
check("Entry has correct", isinstance(t0_entry.get("correct"), bool))
check("Entry has zone", t0_entry.get("zone") in ("LOW", "MODERATE", "HIGH"))
check("Has recommendations", isinstance(pt.get("recommendations"), list) and len(pt["recommendations"]) > 0)
check("Has plot", isinstance(pt.get("plot"), str) and len(pt["plot"]) > 100)
check("Has stakeholder_brief", isinstance(pt.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(pt["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(pt.get("elapsed_s"), (int, float)))

# 6P test
pt2 = api("POST", "/api/analysis/prediction-timeline", {"source": "demo", "well": "6P"})
check("6P works", pt2 is not None and pt2.get("well") == "6P")


# ── [144] Augmentation Preview ─────────────────────────
print("\n[144] Data Augmentation Preview")
ap = api("POST", "/api/analysis/augmentation-preview", {"source": "demo", "well": "3P"})
check("Status 200", ap is not None)
check("Has well", ap.get("well") == "3P")
check("Has n_samples_original", isinstance(ap.get("n_samples_original"), int) and ap["n_samples_original"] > 0)
check("Has n_classes", isinstance(ap.get("n_classes"), int) and ap["n_classes"] >= 2)
check("Has imbalance_ratio", isinstance(ap.get("imbalance_ratio"), (int, float)))
check("Has baseline_accuracy", isinstance(ap.get("baseline_accuracy"), (int, float)))
check("Has best_method", isinstance(ap.get("best_method"), str))
check("Has best_accuracy", isinstance(ap.get("best_accuracy"), (int, float)))
check("Has best_improvement", isinstance(ap.get("best_improvement"), (int, float)))
check("Has strategies list", isinstance(ap.get("strategies"), list) and len(ap["strategies"]) == 3)
s0 = ap["strategies"][0]
check("Strategy has method", isinstance(s0.get("method"), str))
check("Strategy has n_synthetic", isinstance(s0.get("n_synthetic"), int))
check("Strategy has baseline_accuracy", isinstance(s0.get("baseline_accuracy"), (int, float)))
check("Strategy has augmented_accuracy", isinstance(s0.get("augmented_accuracy"), (int, float)))
check("Strategy has improvement_pct", isinstance(s0.get("improvement_pct"), (int, float)))
check("Has class_distribution", isinstance(ap.get("class_distribution"), list) and len(ap["class_distribution"]) >= 2)
check("Has recommendations", isinstance(ap.get("recommendations"), list) and len(ap["recommendations"]) > 0)
check("Has plot", isinstance(ap.get("plot"), str) and len(ap["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ap.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(ap["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(ap.get("elapsed_s"), (int, float)))

# 6P test
ap2 = api("POST", "/api/analysis/augmentation-preview", {"source": "demo", "well": "6P"})
check("6P works", ap2 is not None and ap2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.41.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
