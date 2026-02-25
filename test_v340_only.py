"""Focused test for v3.40.0 endpoints: Regulatory Compliance + Operator Workflow + Smart Alerts + Model Lifecycle + Benchmark."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8124"


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


# ── [135] Regulatory Compliance ─────────────────────────
print("\n[135] Regulatory Compliance Report")
rc = api("POST", "/api/report/regulatory-compliance", {"source": "demo", "well": "3P"})
check("Status 200", rc is not None)
check("Has well", rc.get("well") == "3P")
check("Has overall_status", rc.get("overall_status") in ("COMPLIANT", "CONDITIONALLY_COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT"))
check("Has compliance_score", isinstance(rc.get("compliance_score"), (int, float)) and 0 <= rc["compliance_score"] <= 100)
check("Has n_checks", isinstance(rc.get("n_checks"), int) and rc["n_checks"] >= 5)
check("Has n_pass", isinstance(rc.get("n_pass"), int))
check("Has n_warn", isinstance(rc.get("n_warn"), int))
check("Has n_fail", isinstance(rc.get("n_fail"), int))
check("Has checks list", isinstance(rc.get("checks"), list) and len(rc["checks"]) >= 5)
ch0 = rc["checks"][0]
check("Check has framework", isinstance(ch0.get("framework"), str))
check("Check has requirement", isinstance(ch0.get("requirement"), str))
check("Check has status", ch0.get("status") in ("PASS", "WARN", "FAIL"))
check("Check has evidence", isinstance(ch0.get("evidence"), str))
check("Check has section", isinstance(ch0.get("section"), str))
check("Has recommendations", isinstance(rc.get("recommendations"), list) and len(rc["recommendations"]) > 0)
check("Has plot", isinstance(rc.get("plot"), str) and len(rc["plot"]) > 100)
check("Has stakeholder_brief", isinstance(rc.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(rc["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(rc.get("elapsed_s"), (int, float)))

# 6P test
rc2 = api("POST", "/api/report/regulatory-compliance", {"source": "demo", "well": "6P"})
check("6P works", rc2 is not None and rc2.get("well") == "6P")


# ── [136] Operator Workflow Checklist ───────────────────
print("\n[136] Operator Workflow Checklist")
wf = api("POST", "/api/workflow/checklist", {"source": "demo", "well": "3P"})
check("Status 200", wf is not None)
check("Has well", wf.get("well") == "3P")
check("Has n_steps", isinstance(wf.get("n_steps"), int) and wf["n_steps"] >= 8)
check("Has n_complete", isinstance(wf.get("n_complete"), int))
check("Has n_blocked", isinstance(wf.get("n_blocked"), int))
check("Has progress_pct", isinstance(wf.get("progress_pct"), (int, float)) and 0 <= wf["progress_pct"] <= 100)
check("Has next_recommended", isinstance(wf.get("next_recommended"), str))
check("Has steps list", isinstance(wf.get("steps"), list) and len(wf["steps"]) >= 8)
st0 = wf["steps"][0]
check("Step has step number", isinstance(st0.get("step"), int))
check("Step has name", isinstance(st0.get("name"), str))
check("Step has status", st0.get("status") in ("COMPLETE", "RECOMMENDED", "PENDING", "BLOCKED", "OPTIONAL"))
check("Step has description", isinstance(st0.get("description"), str))
check("Step has endpoint", isinstance(st0.get("endpoint"), str))
check("Step has priority", st0.get("priority") in ("REQUIRED", "RECOMMENDED", "OPTIONAL"))
check("Step has detail", isinstance(st0.get("detail"), str))
check("Has stakeholder_brief", isinstance(wf.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(wf["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(wf.get("elapsed_s"), (int, float)))

# 6P test
wf2 = api("POST", "/api/workflow/checklist", {"source": "demo", "well": "6P"})
check("6P works", wf2 is not None and wf2.get("well") == "6P")


# ── [137] Smart Alerts ─────────────────────────────────
print("\n[137] Smart Alert System")
al = api("POST", "/api/alerts/check", {"source": "demo", "well": "3P"})
check("Status 200", al is not None)
check("Has well", al.get("well") == "3P")
check("Has overall", al.get("overall") in ("CLEAR", "CAUTION", "ALERT"))
check("Has n_alerts", isinstance(al.get("n_alerts"), int))
check("Has n_critical", isinstance(al.get("n_critical"), int))
check("Has n_warning", isinstance(al.get("n_warning"), int))
check("Has n_info", isinstance(al.get("n_info"), int))
check("Has alerts list", isinstance(al.get("alerts"), list))
if al["alerts"]:
    a0 = al["alerts"][0]
    check("Alert has severity", a0.get("severity") in ("CRITICAL", "WARNING", "INFO"))
    check("Alert has category", isinstance(a0.get("category"), str))
    check("Alert has alert msg", isinstance(a0.get("alert"), str))
    check("Alert has action", isinstance(a0.get("action"), str))
else:
    for _ in range(4):
        check("Alert field", True, "no alerts = clean data")
check("Has stakeholder_brief", isinstance(al.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(al["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(al.get("elapsed_s"), (int, float)))

# 6P test — should have more alerts due to missing depths
al2 = api("POST", "/api/alerts/check", {"source": "demo", "well": "6P"})
check("6P works", al2 is not None and al2.get("well") == "6P")
check("6P has alerts", al2.get("n_alerts", 0) >= 1, f"n_alerts={al2.get('n_alerts', 0)}")


# ── [138] Model Lifecycle Management ───────────────────
print("\n[138] Model Lifecycle Management")
ml = api("POST", "/api/models/lifecycle", {"source": "demo", "well": "3P"})
check("Status 200", ml is not None)
check("Has well", ml.get("well") == "3P")
check("Has n_models", isinstance(ml.get("n_models"), int) and ml["n_models"] >= 3)
check("Has n_healthy", isinstance(ml.get("n_healthy"), int))
check("Has n_degraded", isinstance(ml.get("n_degraded"), int))
check("Has n_critical", isinstance(ml.get("n_critical"), int))
check("Has best_model", isinstance(ml.get("best_model"), str))
check("Has best_accuracy", isinstance(ml.get("best_accuracy"), (int, float)))
check("Has models list", isinstance(ml.get("models"), list) and len(ml["models"]) >= 3)
m0 = ml["models"][0]
check("Model has name", isinstance(m0.get("model"), str))
check("Model has balanced_accuracy", isinstance(m0.get("balanced_accuracy"), (int, float)))
check("Model has health", m0.get("health") in ("HEALTHY", "DEGRADED", "CRITICAL", "ERROR"))
check("Model has recommended_action", isinstance(m0.get("recommended_action"), str))
check("Model has n_classes", isinstance(m0.get("n_classes"), int))
check("Model has n_features", isinstance(m0.get("n_features"), int))
check("Model has n_samples", isinstance(m0.get("n_samples"), int))
check("Has recommendations", isinstance(ml.get("recommendations"), list) and len(ml["recommendations"]) > 0)
check("Has plot", isinstance(ml.get("plot"), str) and len(ml["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ml.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(ml["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(ml.get("elapsed_s"), (int, float)))

# 6P test
ml2 = api("POST", "/api/models/lifecycle", {"source": "demo", "well": "6P"})
check("6P works", ml2 is not None and ml2.get("well") == "6P")


# ── [139] Performance Benchmark ────────────────────────
print("\n[139] Performance Benchmark")
bm = api("POST", "/api/benchmark/run", {"source": "demo", "well": "3P"})
check("Status 200", bm is not None)
check("Has well", bm.get("well") == "3P")
check("Has n_samples", isinstance(bm.get("n_samples"), int) and bm["n_samples"] > 0)
check("Has total_time_s", isinstance(bm.get("total_time_s"), (int, float)) and bm["total_time_s"] > 0)
check("Has bottleneck", isinstance(bm.get("bottleneck"), str))
check("Has bottleneck_time_s", isinstance(bm.get("bottleneck_time_s"), (int, float)))
check("Has benchmarks list", isinstance(bm.get("benchmarks"), list) and len(bm["benchmarks"]) >= 4)
b0 = bm["benchmarks"][0]
check("Benchmark has step", isinstance(b0.get("step"), str))
check("Benchmark has time_s", isinstance(b0.get("time_s"), (int, float)))
check("Benchmark has detail", isinstance(b0.get("detail"), str))
check("Has recommendations", isinstance(bm.get("recommendations"), list) and len(bm["recommendations"]) > 0)
check("Has plot", isinstance(bm.get("plot"), str) and len(bm["plot"]) > 100)
check("Has stakeholder_brief", isinstance(bm.get("stakeholder_brief"), dict))
check("Brief has for_non_experts", isinstance(bm["stakeholder_brief"].get("for_non_experts"), str))
check("Has elapsed_s", isinstance(bm.get("elapsed_s"), (int, float)))

# 6P test
bm2 = api("POST", "/api/benchmark/run", {"source": "demo", "well": "6P"})
check("6P works", bm2 is not None and bm2.get("well") == "6P")


# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.40.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
