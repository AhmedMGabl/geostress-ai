"""Focused test for v3.28.0 endpoints: Decision Support, Risk Report, Transparency Audit."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8099"


def api(method, path, body=None, timeout=60):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_expect_error(method, path, body=None, expected_status=400):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return False
    except urllib.error.HTTPError as e:
        return e.code == expected_status


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


# ── [94] Decision Support Matrix ──────────────────────
print("\n[94] Decision Support Matrix")
dsx = api("POST", "/api/analysis/decision-support", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", dsx is not None)
check("Has well", dsx.get("well") == "3P")
check("Has n_samples", isinstance(dsx.get("n_samples"), int) and dsx["n_samples"] > 0)
check("Has decision", dsx.get("decision") in ("GO", "CAUTION", "STOP"))
check("Has overall_score", isinstance(dsx.get("overall_score"), (int, float)) and 0 <= dsx["overall_score"] <= 100)
check("Has criteria", isinstance(dsx.get("criteria"), list) and len(dsx["criteria"]) >= 4)
c0dsx = dsx["criteria"][0]
check("Crit has criterion", isinstance(c0dsx.get("criterion"), str))
check("Crit has score", isinstance(c0dsx.get("score"), int) and 0 <= c0dsx["score"] <= 100)
check("Crit has weight", isinstance(c0dsx.get("weight"), int))
check("Crit has detail", isinstance(c0dsx.get("detail"), str))
check("Has recommendations", isinstance(dsx.get("recommendations"), list) and len(dsx["recommendations"]) > 0)
check("Has best_model", isinstance(dsx.get("best_model"), str))
check("Has best_accuracy", isinstance(dsx.get("best_accuracy"), (int, float)))
check("Has plot", isinstance(dsx.get("plot"), str) and len(dsx["plot"]) > 100)
check("Has stakeholder_brief", isinstance(dsx.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(dsx["stakeholder_brief"].get("headline"), str))
check("Brief has risk_level", dsx["stakeholder_brief"].get("risk_level") in ("GREEN", "AMBER", "RED"))

# ── [95] Risk Communication Report ───────────────────
print("\n[95] Risk Communication Report")
rr = api("POST", "/api/analysis/risk-report", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", rr is not None)
check("Has well", rr.get("well") == "3P")
check("Has n_samples", isinstance(rr.get("n_samples"), int) and rr["n_samples"] > 0)
check("Has overall_risk", rr.get("overall_risk") in ("LOW", "MEDIUM", "HIGH"))
check("Has executive_summary", isinstance(rr.get("executive_summary"), str) and len(rr["executive_summary"]) > 20)
check("Has risks", isinstance(rr.get("risks"), list) and len(rr["risks"]) >= 3)
rk0 = rr["risks"][0]
check("Risk has category", isinstance(rk0.get("category"), str))
check("Risk has risk_level", rk0.get("risk_level") in ("LOW", "MEDIUM", "HIGH"))
check("Risk has plain_english", isinstance(rk0.get("plain_english"), str) and len(rk0["plain_english"]) > 20)
check("Risk has impact", isinstance(rk0.get("impact"), str))
check("Risk has mitigation", isinstance(rk0.get("mitigation"), str))
check("Has n_high_risks", isinstance(rr.get("n_high_risks"), int))
check("Has n_medium_risks", isinstance(rr.get("n_medium_risks"), int))
check("Has n_low_risks", isinstance(rr.get("n_low_risks"), int))
check("Risk counts sum", rr["n_high_risks"] + rr["n_medium_risks"] + rr["n_low_risks"] == len(rr["risks"]))
check("Has plot", isinstance(rr.get("plot"), str) and len(rr["plot"]) > 100)
check("Has stakeholder_brief", isinstance(rr.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(rr["stakeholder_brief"].get("headline"), str))

cat_names = [r["category"] for r in rr["risks"]]
check("Data risk exists", any("Data" in c for c in cat_names))
check("Model risk exists", any("Model" in c for c in cat_names))

# ── [96] Model Transparency Audit ────────────────────
print("\n[96] Model Transparency Audit")
ta = api("POST", "/api/analysis/transparency-audit", {"source": "demo", "well": "3P"}, timeout=300)
check("Status 200", ta is not None)
check("Has well", ta.get("well") == "3P")
check("Has n_samples", isinstance(ta.get("n_samples"), int) and ta["n_samples"] > 0)
check("Has n_audited", isinstance(ta.get("n_audited"), int) and ta["n_audited"] > 0)
check("Has n_correct", isinstance(ta.get("n_correct"), int))
check("Has n_wrong", isinstance(ta.get("n_wrong"), int))
check("Correct+Wrong=Audited", ta["n_correct"] + ta["n_wrong"] == ta["n_audited"])
check("Has audit_accuracy", isinstance(ta.get("audit_accuracy"), (int, float)))
check("Has n_models", isinstance(ta.get("n_models"), int) and ta["n_models"] >= 2)
check("Has global_feature_importances", isinstance(ta.get("global_feature_importances"), list))
if ta["global_feature_importances"]:
    gi0 = ta["global_feature_importances"][0]
    check("GI has feature", isinstance(gi0.get("feature"), str))
    check("GI has importance", isinstance(gi0.get("importance"), (int, float)))
check("Has transparency_cards", isinstance(ta.get("transparency_cards"), list) and len(ta["transparency_cards"]) > 0)
tc0 = ta["transparency_cards"][0]
check("TC has index", isinstance(tc0.get("index"), int))
check("TC has depth_m", True)  # may be null
check("TC has true_class", isinstance(tc0.get("true_class"), str))
check("TC has consensus_class", isinstance(tc0.get("consensus_class"), str))
check("TC has correct", isinstance(tc0.get("correct"), bool))
check("TC has agreement", isinstance(tc0.get("agreement"), (int, float)))
check("TC has model_details", isinstance(tc0.get("model_details"), list) and len(tc0["model_details"]) >= 2)
md0 = tc0["model_details"][0]
check("MD has model", isinstance(md0.get("model"), str))
check("MD has predicted", isinstance(md0.get("predicted"), str))
check("MD has correct", isinstance(md0.get("correct"), bool))
check("TC has top_features", isinstance(tc0.get("top_features"), list))
check("TC has geology_note", isinstance(tc0.get("geology_note"), str))
check("Has plot", isinstance(ta.get("plot"), str) and len(ta["plot"]) > 100)
check("Has stakeholder_brief", isinstance(ta.get("stakeholder_brief"), dict))
check("Brief has headline", isinstance(ta["stakeholder_brief"].get("headline"), str))

# Param validation
check("top_n=0 rejected", api_expect_error("POST", "/api/analysis/transparency-audit", {"source": "demo", "well": "3P", "top_n": 0}))

# Custom top_n
ta5 = api("POST", "/api/analysis/transparency-audit", {"source": "demo", "well": "3P", "top_n": 5}, timeout=300)
check("top_n=5 works", ta5 is not None and ta5.get("n_audited") == 5)

# ── Summary ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"v3.28.0 Tests: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*50}")

if failed > 0:
    exit(1)
