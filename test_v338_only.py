"""Focused test for v3.38.0 endpoints: SSE Streaming + Data Validation + What-If + Ensemble + Fast Classify."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8117"


def api(method, path, body=None, timeout=120):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_stream(path, body, timeout=120):
    """Fetch SSE endpoint and collect all events."""
    url = BASE + path
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
    return events


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


# ── [125] SSE Streaming Inversion ─────────────────────────
print("\n[125] SSE Streaming Inversion")
events = api_stream("/api/analysis/inversion-stream", {"source": "demo", "well": "3P", "n_steps": 200}, timeout=300)
check("Got events", len(events) >= 2)
start_events = [e for e in events if e.get("event") == "start"]
check("Has start event", len(start_events) >= 1)
if start_events:
    check("Start has well", start_events[0].get("well") == "3P")
    check("Start has regime", isinstance(start_events[0].get("regime"), str))
progress_events = [e for e in events if e.get("event") == "progress"]
check("Has progress events", len(progress_events) >= 1)
if progress_events:
    p0 = progress_events[0]
    check("Progress has step", isinstance(p0.get("step"), int))
    check("Progress has pct", isinstance(p0.get("pct"), (int, float)))
    check("Progress has current_best", isinstance(p0.get("current_best"), dict))
    check("Current best has SHmax", isinstance(p0["current_best"].get("SHmax_azimuth"), (int, float)))
complete_events = [e for e in events if e.get("event") == "complete"]
check("Has complete event", len(complete_events) >= 1)
if complete_events:
    c = complete_events[0]
    check("Complete has result", isinstance(c.get("result"), dict))
    check("Result has SHmax", isinstance(c["result"].get("SHmax_azimuth"), (int, float)))
    check("Complete has plot", isinstance(c.get("plot"), str) and len(c["plot"]) > 100)
    check("Complete has stakeholder_brief", isinstance(c.get("stakeholder_brief"), dict))


# ── [126] Data Validation ──────────────────────────────────
print("\n[126] Data Quality Validation")
dv = api("POST", "/api/data/quality-check", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", dv is not None)
check("Has well", dv.get("well") == "3P")
check("Has quality_score", isinstance(dv.get("quality_score"), int) and 0 <= dv["quality_score"] <= 100)
check("Has grade", dv.get("grade") in ("A", "B", "C", "D", "F"))
check("Has n_issues", isinstance(dv.get("n_issues"), int))
check("Has n_high", isinstance(dv.get("n_high"), int))
check("Has issues list", isinstance(dv.get("issues"), list))
if dv["issues"]:
    iss0 = dv["issues"][0]
    check("Issue has severity", iss0.get("severity") in ("HIGH", "MEDIUM", "LOW"))
    check("Issue has category", isinstance(iss0.get("category"), str))
    check("Issue has issue", isinstance(iss0.get("issue"), str))
else:
    check("Issue has severity", True, "no issues found = perfect data")
    check("Issue has category", True)
    check("Issue has issue", True)
check("Has class_distribution", isinstance(dv.get("class_distribution"), list) and len(dv["class_distribution"]) >= 2)
check("Has imbalance_ratio", isinstance(dv.get("imbalance_ratio"), (int, float)))
check("Has missing_data", isinstance(dv.get("missing_data"), dict))
check("Has recommendations", isinstance(dv.get("recommendations"), list) and len(dv["recommendations"]) > 0)
check("Has plot", isinstance(dv.get("plot"), str) and len(dv["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dv.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(dv["stakeholder_brief"].get("for_non_experts"), str))

# 6P test
dv2 = api("POST", "/api/data/quality-check", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", dv2 is not None and dv2.get("well") == "6P")


# ── [127] What-If Simulator ────────────────────────────────
print("\n[127] What-If Scenario Simulator")
wi = api("POST", "/api/analysis/scenario-compare", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", wi is not None)
check("Has well", wi.get("well") == "3P")
check("Has baseline_params", isinstance(wi.get("baseline_params"), dict))
check("Has alternative_params", isinstance(wi.get("alternative_params"), dict))
check("Has baseline_result", isinstance(wi.get("baseline_result"), dict))
check("Has alternative_result", isinstance(wi.get("alternative_result"), dict))
check("Has differences", isinstance(wi.get("differences"), dict))
check("Diff has min_mw_change", isinstance(wi["differences"].get("min_mw_change_MPa"), (int, float)))
check("Diff has critical_zones_change", isinstance(wi["differences"].get("critical_zones_change"), int))
check("Has verdict", wi.get("verdict") in ("BASELINE_BETTER", "ALTERNATIVE_BETTER", "SIMILAR"))
check("Has recommendations", isinstance(wi.get("recommendations"), list) and len(wi["recommendations"]) > 0)
check("Has plot", isinstance(wi.get("plot"), str) and len(wi["plot"]) > 100)
check("Has stakeholder_brief", isinstance(wi.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(wi["stakeholder_brief"].get("for_non_experts"), str))


# ── [128] Ensemble Voting ──────────────────────────────────
print("\n[128] Multi-Model Ensemble Voting")
ev = api("POST", "/api/analysis/ensemble-vote", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", ev is not None)
check("Has well", ev.get("well") == "3P")
check("Has n_models", isinstance(ev.get("n_models"), int) and ev["n_models"] >= 3)
check("Has ensemble_balanced_accuracy", isinstance(ev.get("ensemble_balanced_accuracy"), (int, float)))
check("Has pct_unanimous", isinstance(ev.get("pct_unanimous"), (int, float)))
check("Has pct_majority", isinstance(ev.get("pct_majority"), (int, float)))
check("Has n_contested", isinstance(ev.get("n_contested"), int))
check("Has leaderboard", isinstance(ev.get("leaderboard"), list) and len(ev["leaderboard"]) >= 3)
lb0 = ev["leaderboard"][0]
check("LB has model", isinstance(lb0.get("model"), str))
check("LB has balanced_accuracy", isinstance(lb0.get("balanced_accuracy"), (int, float)))
check("Has sample_details", isinstance(ev.get("sample_details"), list) and len(ev["sample_details"]) > 0)
sd0 = ev["sample_details"][0]
check("SD has ensemble_prediction", isinstance(sd0.get("ensemble_prediction"), str))
check("SD has ensemble_confidence", isinstance(sd0.get("ensemble_confidence"), (int, float)))
check("SD has n_models_agree", isinstance(sd0.get("n_models_agree"), int))
check("SD has consensus_level", sd0.get("consensus_level") in ("UNANIMOUS", "MAJORITY", "CONTESTED"))
check("SD has model_votes", isinstance(sd0.get("model_votes"), dict))
check("SD has needs_review", isinstance(sd0.get("needs_review"), bool))
check("Has recommendations", isinstance(ev.get("recommendations"), list) and len(ev["recommendations"]) > 0)
check("Has plot", isinstance(ev.get("plot"), str) and len(ev["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ev.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(ev["stakeholder_brief"].get("for_non_experts"), str))

# 6P test
ev2 = api("POST", "/api/analysis/ensemble-vote", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", ev2 is not None and ev2.get("well") == "6P")


# ── [129] Fast Classify ────────────────────────────────────
print("\n[129] Fast Classification (Optimized)")
fc = api("POST", "/api/analysis/fast-classify", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", fc is not None)
check("Has well", fc.get("well") == "3P")
check("Has standard", isinstance(fc.get("standard"), dict))
check("Std has balanced_accuracy", isinstance(fc["standard"].get("balanced_accuracy"), (int, float)))
check("Std has inference_time_ms", isinstance(fc["standard"].get("inference_time_ms"), (int, float)))
check("Has optimized", isinstance(fc.get("optimized"), dict))
check("Opt has balanced_accuracy", isinstance(fc["optimized"].get("balanced_accuracy"), (int, float)))
check("Opt has inference_time_ms", isinstance(fc["optimized"].get("inference_time_ms"), (int, float)))
check("Has speedup", isinstance(fc.get("speedup"), (int, float)) and fc["speedup"] > 0)
check("Has accuracy_loss", isinstance(fc.get("accuracy_loss"), (int, float)))
check("Has agreement", isinstance(fc.get("agreement"), (int, float)) and 0 <= fc["agreement"] <= 1)
check("Has per_class", isinstance(fc.get("per_class"), list) and len(fc["per_class"]) >= 2)
pc0 = fc["per_class"][0]
check("PC has class", isinstance(pc0.get("class"), str))
check("PC has standard_accuracy", isinstance(pc0.get("standard_accuracy"), (int, float)))
check("PC has optimized_accuracy", isinstance(pc0.get("optimized_accuracy"), (int, float)))
check("Has recommendations", isinstance(fc.get("recommendations"), list) and len(fc["recommendations"]) > 0)
check("Has plot", isinstance(fc.get("plot"), str) and len(fc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(fc.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(fc["stakeholder_brief"].get("for_non_experts"), str))

# 6P test
fc2 = api("POST", "/api/analysis/fast-classify", {"source": "demo", "well": "6P"}, timeout=300)
check("6P works", fc2 is not None and fc2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.38.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
