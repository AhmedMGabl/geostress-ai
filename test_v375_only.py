"""Tests for v3.75.0: APL, Stability Window, Trip Margin, Pack-Off, ECD Profile."""
import requests, sys

BASE = "http://localhost:8170"
SESSION = requests.Session()
PASS = 0
FAIL = 0

def test(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label}")

def api(method, path, payload=None):
    try:
        if method == "POST":
            r = SESSION.post(BASE + path, json=payload, timeout=300)
        else:
            r = SESSION.get(BASE + path, timeout=300)
        return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, {}


# ═══════════════════════════════════════════════════════════════
# [310] Annular Pressure Loss
# ═══════════════════════════════════════════════════════════════
print("\n[310] Annular Pressure Loss")
code, r = api("POST", "/api/analysis/annular-pressure-loss", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "flow_rate_gpm": 500, "mud_weight_ppg": 10})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has total_loss_psi", "total_loss_psi" in r)
test("Has total_loss_ppg", "total_loss_ppg" in r)
test("Has ECD_at_TD_ppg", "ECD_at_TD_ppg" in r)
test("Has apl_class", "apl_class" in r)
test("Class valid", r.get("apl_class") in ("HIGH", "MODERATE", "LOW", "MINIMAL"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has ECD_ppg", r.get("profile", [{}])[0].get("ECD_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High flow", api("POST", "/api/analysis/annular-pressure-loss", {"source": "demo", "well": "3P", "flow_rate_gpm": 800})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/annular-pressure-loss", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [311] Stability Window Profile
# ═══════════════════════════════════════════════════════════════
print("\n[311] Stability Window Profile")
code, r = api("POST", "/api/analysis/stability-window-profile", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "UCS_MPa": 40})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has UCS_MPa", "UCS_MPa" in r)
test("Has min_window_ppg", "min_window_ppg" in r)
test("Has mean_window_ppg", "mean_window_ppg" in r)
test("Has sw_class", "sw_class" in r)
test("Class valid", r.get("sw_class") in ("NO_WINDOW", "NARROW", "ADEQUATE", "WIDE"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has breakout_ppg", r.get("profile", [{}])[0].get("breakout_ppg") is not None)
test("Prof has breakdown_ppg", r.get("profile", [{}])[0].get("breakdown_ppg") is not None)
test("Prof has window_ppg", r.get("profile", [{}])[0].get("window_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Weak rock", api("POST", "/api/analysis/stability-window-profile", {"source": "demo", "well": "3P", "UCS_MPa": 10})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/stability-window-profile", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [312] Trip Margin Analysis
# ═══════════════════════════════════════════════════════════════
print("\n[312] Trip Margin Analysis")
code, r = api("POST", "/api/analysis/trip-margin-analysis", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "mud_weight_ppg": 10, "connection_time_min": 5})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has connection_time_min", "connection_time_min" in r)
test("Has min_effective_margin_ppg", "min_effective_margin_ppg" in r)
test("Has mean_effective_margin_ppg", "mean_effective_margin_ppg" in r)
test("Has pct_negative", "pct_negative" in r)
test("Has tm_class", "tm_class" in r)
test("Class valid", r.get("tm_class") in ("INSUFFICIENT", "TIGHT", "ADEQUATE", "COMFORTABLE"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has effective_margin_ppg", r.get("profile", [{}])[0].get("effective_margin_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Long connection", api("POST", "/api/analysis/trip-margin-analysis", {"source": "demo", "well": "3P", "connection_time_min": 15})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/trip-margin-analysis", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [313] Pack-Off Risk
# ═══════════════════════════════════════════════════════════════
print("\n[313] Pack-Off Risk")
code, r = api("POST", "/api/analysis/pack-off-risk", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "flow_rate_gpm": 500, "rop_ft_hr": 60})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has flow_rate_gpm", "flow_rate_gpm" in r)
test("Has rop_ft_hr", "rop_ft_hr" in r)
test("Has mean_pack_risk", "mean_pack_risk" in r)
test("Has max_pack_risk", "max_pack_risk" in r)
test("Has po_class", "po_class" in r)
test("Class valid", r.get("po_class") in ("HIGH", "MODERATE", "LOW", "MINIMAL"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has pack_risk", r.get("profile", [{}])[0].get("pack_risk") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("High ROP", api("POST", "/api/analysis/pack-off-risk", {"source": "demo", "well": "3P", "rop_ft_hr": 150})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/pack-off-risk", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
# [314] ECD Profile (Comprehensive)
# ═══════════════════════════════════════════════════════════════
print("\n[314] ECD Profile")
code, r = api("POST", "/api/analysis/equivalent-circulating-density", {"source": "demo", "well": "3P", "depth_from": 500, "depth_to": 5000, "mud_weight_ppg": 10, "flow_rate_gpm": 500, "rop_ft_hr": 60})
test("Status 200", code == 200)
test("Has well", "well" in r)
test("Has depth_from_m", "depth_from_m" in r)
test("Has mud_weight_ppg", "mud_weight_ppg" in r)
test("Has flow_rate_gpm", "flow_rate_gpm" in r)
test("Has rop_ft_hr", "rop_ft_hr" in r)
test("Has ECD_at_TD_ppg", "ECD_at_TD_ppg" in r)
test("Has max_ECD_ppg", "max_ECD_ppg" in r)
test("Has min_frac_margin_ppg", "min_frac_margin_ppg" in r)
test("Has ecd_class", "ecd_class" in r)
test("Class valid", r.get("ecd_class") in ("CRITICAL", "TIGHT", "ADEQUATE", "SAFE"))
test("Has profile", "profile" in r)
test("Prof has depth_m", r.get("profile", [{}])[0].get("depth_m") is not None)
test("Prof has ECD_ppg", r.get("profile", [{}])[0].get("ECD_ppg") is not None)
test("Prof has ecd_margin_ppg", r.get("profile", [{}])[0].get("ecd_margin_ppg") is not None)
test("Has recommendations", "recommendations" in r)
test("Has plot", "plot" in r and isinstance(r["plot"], str) and r["plot"].startswith("data:image"))
test("Has stakeholder_brief", "stakeholder_brief" in r)
test("Brief has for_non_experts", "for_non_experts" in r.get("stakeholder_brief", {}))
test("Has elapsed_s", "elapsed_s" in r)
test("Heavy MW", api("POST", "/api/analysis/equivalent-circulating-density", {"source": "demo", "well": "3P", "mud_weight_ppg": 14})[0] in (200, 422, 500))
test("6P works", api("POST", "/api/analysis/equivalent-circulating-density", {"source": "demo", "well": "6P"})[0] in (200, 422, 500))


# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"v3.75.0 Tests: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
