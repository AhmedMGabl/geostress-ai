

# ═══════════════════════════════════════════════════════════════════
# [180] Stress Polygon
# ═══════════════════════════════════════════════════════════════════
_stress_polygon_cache = {}


@app.post("/api/analysis/stress-polygon")
async def analysis_stress_polygon(request: Request):
    """Stress polygon: allowable stress states for borehole stability."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth = float(body.get("depth", 3000))
    friction = float(body.get("friction", 0.6))
    pp_mpa = body.get("pp_mpa")

    cache_key = f"{source}:{well}:{depth}:{friction}:{pp_mpa}"
    if cache_key in _stress_polygon_cache:
        cached = _stress_polygon_cache[cache_key]
        cached["elapsed_s"] = round(time.time() - t0, 2)
        return _sanitize_for_json(cached)

    df = get_df(source)
    if well not in df:
        return JSONResponse(status_code=404, content={"error": f"Well '{well}' not found"})

    def _compute():
        df_well = df[well]
        n = len(df_well)

        rho_rock = 2500
        g = 9.81
        Sv = rho_rock * g * depth / 1e6
        if pp_mpa is not None:
            Pp = float(pp_mpa)
        else:
            Pp = 1000 * g * depth / 1e6

        mu = friction
        q = ((mu**2 + 1)**0.5 + mu)**2

        # Stress polygon boundaries
        polygons = {"NF": [], "SS": [], "RF": []}
        n_pts = 50
        sh_min_val = max(Pp + 0.5, Sv / q)
        sh_max_val = Sv

        for i in range(n_pts + 1):
            sh = sh_min_val + (sh_max_val - sh_min_val) * i / n_pts
            sH_nf_low = sh
            sH_nf_high = Sv
            polygons["NF"].append({"Shmin_MPa": round(sh, 2), "SHmax_low": round(sH_nf_low, 2), "SHmax_high": round(sH_nf_high, 2)})

        for i in range(n_pts + 1):
            sh = sh_min_val + (Sv - sh_min_val) * i / n_pts
            sH_ss_max = min(q * (sh - Pp) + Pp, Sv * 3)
            polygons["SS"].append({"Shmin_MPa": round(sh, 2), "SHmax_max": round(max(Sv, sH_ss_max), 2)})

        for i in range(n_pts + 1):
            sh = Sv + (Sv * 2 - Sv) * i / n_pts
            sH_rf_max = q * (sh - Pp) + Pp
            polygons["RF"].append({"Shmin_MPa": round(sh, 2), "SHmax_max": round(sH_rf_max, 2)})

        # Current stress state estimate
        try:
            from src.geostress import invert_stress
            inv = invert_stress(df_well, regime="NF", depth=depth, pore_pressure=Pp)
            sigma1 = float(inv.get("sigma1", Sv))
            sigma3 = float(inv.get("sigma3", Sv * 0.6))
            SHmax_est = sigma1 if inv.get("regime") == "RF" else Sv
            Shmin_est = sigma3 if inv.get("regime") != "RF" else Sv
        except Exception:
            SHmax_est = Sv * 0.9
            Shmin_est = Sv * 0.6

        current_state = {"SHmax_MPa": round(SHmax_est, 2), "Shmin_MPa": round(Shmin_est, 2), "Sv_MPa": round(Sv, 2)}

        if SHmax_est <= Sv:
            current_regime = "NF"
        elif Shmin_est >= Sv:
            current_regime = "RF"
        else:
            current_regime = "SS"

        if current_regime == "NF":
            margin = (Sv - SHmax_est) / max(Sv, 1)
        elif current_regime == "SS":
            max_sH = q * (Shmin_est - Pp) + Pp
            margin = (max_sH - SHmax_est) / max(max_sH, 1)
        else:
            max_sH = q * (Shmin_est - Pp) + Pp
            margin = (max_sH - SHmax_est) / max(max_sH, 1)

        stability_class = "STABLE" if margin > 0.2 else ("MARGINAL" if margin > 0.05 else "CRITICAL")

        recommendations = []
        if stability_class == "CRITICAL":
            recommendations.append("Current stress state is near the frictional limit — high reactivation risk")
        if stability_class == "MARGINAL":
            recommendations.append("Stress state is within 20% of frictional limit — monitor carefully")
        recommendations.append(f"Friction coefficient {mu:.2f} constrains the polygon size")
        if Pp > Sv * 0.4:
            recommendations.append("Elevated pore pressure narrows the allowable stress window")
        recommendations.append(f"Current regime: {current_regime} (Sv={Sv:.1f} MPa at {depth:.0f}m)")

        plot_b64 = ""
        with plot_lock:
            fig, ax = plt.subplots(figsize=(8, 8))
            sh_vals_nf = [p["Shmin_MPa"] for p in polygons["NF"]]
            sH_low_nf = [p["SHmax_low"] for p in polygons["NF"]]
            sH_high_nf = [p["SHmax_high"] for p in polygons["NF"]]
            ax.fill_between(sh_vals_nf, sH_low_nf, sH_high_nf, alpha=0.2, color="blue", label="Normal Fault")

            sh_vals_ss = [p["Shmin_MPa"] for p in polygons["SS"]]
            sH_max_ss = [p["SHmax_max"] for p in polygons["SS"]]
            ax.fill_between(sh_vals_ss, [Sv]*len(sh_vals_ss), sH_max_ss, alpha=0.2, color="green", label="Strike-Slip")

            sh_vals_rf = [p["Shmin_MPa"] for p in polygons["RF"]]
            sH_max_rf = [p["SHmax_max"] for p in polygons["RF"]]
            ax.fill_between(sh_vals_rf, sh_vals_rf, sH_max_rf, alpha=0.2, color="red", label="Reverse Fault")

            ax.plot([0, Sv * 2.5], [0, Sv * 2.5], "k--", alpha=0.3, label="SHmax=Shmin")
            ax.axhline(Sv, color="gray", linestyle=":", alpha=0.5, label=f"Sv={Sv:.1f}")
            ax.axvline(Sv, color="gray", linestyle=":", alpha=0.5)
            ax.plot(Shmin_est, SHmax_est, "r*", markersize=15, zorder=5, label=f"Current ({current_regime})")
            ax.set_xlabel("Shmin (MPa)", fontsize=12)
            ax.set_ylabel("SHmax (MPa)", fontsize=12)
            ax.set_title(f"Stress Polygon — Well {well} @ {depth:.0f}m (mu={mu:.2f})", fontsize=14, fontweight="bold")
            ax.legend(fontsize=9)
            ax.set_xlim(0, Sv * 2)
            ax.set_ylim(0, Sv * 2.5)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "well": well,
            "depth_m": depth,
            "friction": mu,
            "Sv_MPa": round(Sv, 2),
            "Pp_MPa": round(Pp, 2),
            "frictional_limit_q": round(q, 3),
            "current_state": current_state,
            "current_regime": current_regime,
            "stability_margin": round(margin, 3),
            "stability_class": stability_class,
            "n_polygon_points": n_pts + 1,
            "polygon_NF": polygons["NF"][:5],
            "polygon_SS": polygons["SS"][:5],
            "polygon_RF": polygons["RF"][:5],
            "recommendations": recommendations,
            "plot": plot_b64,
            "stakeholder_brief": {
                "headline": f"Stress polygon: {stability_class} (margin {margin:.1%})",
                "risk_level": "RED" if stability_class == "CRITICAL" else ("AMBER" if stability_class == "MARGINAL" else "GREEN"),
                "what_this_means": f"Current stress state at {depth:.0f}m is {stability_class.lower()} relative to frictional limits.",
                "for_non_experts": "This diagram shows the range of possible stress states. If our estimated stress is near the boundary, fractures could reactivate.",
            },
        }

    result = await asyncio.to_thread(_compute)
    elapsed = round(time.time() - t0, 2)
    result["elapsed_s"] = elapsed
    _stress_polygon_cache[cache_key] = result
    return _sanitize_for_json(result)


# ═══════════════════════════════════════════════════════════════════
# [181] Fracture Permeability Tensor
# ═══════════════════════════════════════════════════════════════════
_frac_perm_tensor_cache = {}


@app.post("/api/analysis/fracture-permeability-tensor")
async def analysis_fracture_permeability_tensor(request: Request):
    """Full 3D permeability tensor from fracture orientations and apertures."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    aperture_mm = float(body.get("aperture_mm", 0.5))

    cache_key = f"{source}:{well}:{aperture_mm}"
    if cache_key in _frac_perm_tensor_cache:
        cached = _frac_perm_tensor_cache[cache_key]
        cached["elapsed_s"] = round(time.time() - t0, 2)
        return _sanitize_for_json(cached)

    df = get_df(source)
    if well not in df:
        return JSONResponse(status_code=404, content={"error": f"Well '{well}' not found"})

    def _compute():
        df_well = df[well]
        n = len(df_well)
        azimuths = df_well[AZIMUTH_COL].values.astype(float)
        dips = df_well[DIP_COL].values.astype(float)

        aperture_m = aperture_mm / 1000.0
        single_k = (aperture_m ** 2) / 12.0

        K_tensor = np.zeros((3, 3))
        for i in range(n):
            az_r = np.radians(azimuths[i])
            dip_r = np.radians(dips[i])
            nx = np.sin(dip_r) * np.sin(az_r)
            ny = np.sin(dip_r) * np.cos(az_r)
            nz = np.cos(dip_r)
            nv = np.array([nx, ny, nz])
            K_tensor += single_k * (np.eye(3) - np.outer(nv, nv))
        K_tensor /= max(n, 1)

        eigenvalues, eigenvectors = np.linalg.eigh(K_tensor)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        k_to_darcy = 1.0 / 9.869e-13
        k1_darcy = float(eigenvalues[0]) * k_to_darcy
        k2_darcy = float(eigenvalues[1]) * k_to_darcy
        k3_darcy = float(eigenvalues[2]) * k_to_darcy
        anisotropy_ratio = k1_darcy / max(k3_darcy, 1e-20)

        principal_dirs = []
        for j in range(3):
            ev = eigenvectors[:, j]
            az_deg = float(np.degrees(np.arctan2(ev[0], ev[1])) % 360)
            dip_deg = float(np.degrees(np.arccos(min(abs(ev[2]), 1.0))))
            principal_dirs.append({
                "axis": f"k{j+1}",
                "permeability_darcy": round(float(eigenvalues[j]) * k_to_darcy, 6),
                "azimuth_deg": round(az_deg, 1),
                "dip_deg": round(dip_deg, 1),
            })

        tensor_components = {
            "Kxx": round(float(K_tensor[0, 0]) * k_to_darcy, 6),
            "Kyy": round(float(K_tensor[1, 1]) * k_to_darcy, 6),
            "Kzz": round(float(K_tensor[2, 2]) * k_to_darcy, 6),
            "Kxy": round(float(K_tensor[0, 1]) * k_to_darcy, 6),
            "Kxz": round(float(K_tensor[0, 2]) * k_to_darcy, 6),
            "Kyz": round(float(K_tensor[1, 2]) * k_to_darcy, 6),
        }

        recommendations = []
        if anisotropy_ratio > 10:
            recommendations.append(f"Strong permeability anisotropy ({anisotropy_ratio:.1f}:1) — directional flow expected")
        elif anisotropy_ratio > 3:
            recommendations.append(f"Moderate anisotropy ({anisotropy_ratio:.1f}:1) — some directional preference")
        else:
            recommendations.append(f"Near-isotropic permeability ({anisotropy_ratio:.1f}:1)")
        recommendations.append(f"Maximum permeability direction: {principal_dirs[0]['azimuth_deg']:.0f} deg azimuth")
        if k1_darcy > 1.0:
            recommendations.append("High bulk permeability — good reservoir quality if fractures are connected")
        elif k1_darcy < 0.001:
            recommendations.append("Very low fracture permeability — matrix flow likely dominates")
        recommendations.append(f"Based on {n} fractures with uniform {aperture_mm}mm aperture assumption")

        plot_b64 = ""
        with plot_lock:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            labels = ["X(E)", "Y(N)", "Z(Up)"]
            im = axes[0].imshow(K_tensor * k_to_darcy, cmap="YlOrRd", aspect="auto")
            axes[0].set_xticks(range(3))
            axes[0].set_yticks(range(3))
            axes[0].set_xticklabels(labels)
            axes[0].set_yticklabels(labels)
            for ii in range(3):
                for jj in range(3):
                    axes[0].text(jj, ii, f"{K_tensor[ii,jj]*k_to_darcy:.4f}", ha="center", va="center", fontsize=9)
            axes[0].set_title("Permeability Tensor (darcy)")
            fig.colorbar(im, ax=axes[0], shrink=0.8)

            axes[1].barh(["k3 (min)", "k2 (mid)", "k1 (max)"], [k3_darcy, k2_darcy, k1_darcy], color=["#4CAF50", "#FF9800", "#F44336"])
            axes[1].set_xlabel("Permeability (darcy)")
            axes[1].set_title("Principal Permeabilities")
            for idx_b, val in enumerate([k3_darcy, k2_darcy, k1_darcy]):
                axes[1].text(val + k1_darcy * 0.02, idx_b, f"{val:.4f}", va="center", fontsize=9)
            fig.suptitle(f"Fracture Permeability Tensor — Well {well} ({n} fracs, {aperture_mm}mm)", fontsize=13, fontweight="bold")
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "well": well,
            "n_fractures": int(n),
            "aperture_mm": aperture_mm,
            "tensor_components": tensor_components,
            "principal_permeabilities": principal_dirs,
            "k1_darcy": round(k1_darcy, 6),
            "k2_darcy": round(k2_darcy, 6),
            "k3_darcy": round(k3_darcy, 6),
            "anisotropy_ratio": round(anisotropy_ratio, 2),
            "recommendations": recommendations,
            "plot": plot_b64,
            "stakeholder_brief": {
                "headline": f"Perm tensor: k1={k1_darcy:.4f}D, anisotropy {anisotropy_ratio:.1f}:1",
                "risk_level": "RED" if k1_darcy < 0.001 else ("AMBER" if anisotropy_ratio > 10 else "GREEN"),
                "what_this_means": f"Full 3D permeability tensor from {n} fractures with {aperture_mm}mm aperture.",
                "for_non_experts": "This calculates how easily fluid flows through the rock in all 3 directions. High anisotropy means flow is concentrated in one direction.",
            },
        }

    result = await asyncio.to_thread(_compute)
    elapsed = round(time.time() - t0, 2)
    result["elapsed_s"] = elapsed
    _frac_perm_tensor_cache[cache_key] = result
    return _sanitize_for_json(result)


# ═══════════════════════════════════════════════════════════════════
# [182] Wellbore Breakout Width
# ═══════════════════════════════════════════════════════════════════
_breakout_width_cache = {}


@app.post("/api/analysis/breakout-width")
async def analysis_breakout_width(request: Request):
    """Detailed breakout width analysis with mud weight optimization."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth = float(body.get("depth", 3000))
    ucs_mpa = float(body.get("ucs_mpa", 80))
    friction = float(body.get("friction", 0.6))

    cache_key = f"{source}:{well}:{depth}:{ucs_mpa}:{friction}"
    if cache_key in _breakout_width_cache:
        cached = _breakout_width_cache[cache_key]
        cached["elapsed_s"] = round(time.time() - t0, 2)
        return _sanitize_for_json(cached)

    df = get_df(source)
    if well not in df:
        return JSONResponse(status_code=404, content={"error": f"Well '{well}' not found"})

    def _compute():
        df_well = df[well]
        n = len(df_well)

        rho_rock = 2500
        g = 9.81
        Sv = rho_rock * g * depth / 1e6
        Pp = 1000 * g * depth / 1e6
        Shmin = Sv * 0.6 + Pp * 0.4
        SHmax = Sv * 0.9 + Pp * 0.1

        mw_range = np.linspace(0.8, 2.2, 29)
        results_mw = []

        for mw_sg in mw_range:
            Pw = mw_sg * 1000 * g * depth / 1e6
            sigma_theta_max = 3 * SHmax - Shmin - Pw - Pp
            sigma_theta_min = 3 * Shmin - SHmax - Pw - Pp
            breakout_exists = sigma_theta_max > ucs_mpa
            tensile_exists = sigma_theta_min < 0

            if breakout_exists:
                cos_wbo = (ucs_mpa + Pw + Pp - Shmin) / (SHmax - Shmin) if (SHmax - Shmin) > 0.01 else 1.0
                cos_wbo = max(-1, min(1, cos_wbo))
                wbo_deg = 2 * np.degrees(np.arccos(cos_wbo))
                wbo_deg = min(wbo_deg, 180)
            else:
                wbo_deg = 0.0

            results_mw.append({
                "mud_weight_SG": round(float(mw_sg), 2),
                "Pw_MPa": round(float(Pw), 2),
                "breakout_width_deg": round(float(wbo_deg), 1),
                "breakout_exists": bool(breakout_exists),
                "tensile_fracture_risk": bool(tensile_exists),
                "hoop_stress_max_MPa": round(float(sigma_theta_max), 2),
                "safety_factor": round(float(ucs_mpa / max(sigma_theta_max, 0.01)), 3),
            })

        safe_mws = [r for r in results_mw if not r["breakout_exists"] and not r["tensile_fracture_risk"]]
        if safe_mws:
            optimal_mw = safe_mws[0]["mud_weight_SG"]
        else:
            min_bo = min(results_mw, key=lambda x: x["breakout_width_deg"])
            optimal_mw = min_bo["mud_weight_SG"]

        no_bo = [r for r in results_mw if not r["breakout_exists"]]
        min_mw_no_breakout = no_bo[0]["mud_weight_SG"] if no_bo else None
        no_tf = [r for r in results_mw if not r["tensile_fracture_risk"]]
        max_mw_no_tensile = no_tf[-1]["mud_weight_SG"] if no_tf else None

        mud_weight_window = None
        if min_mw_no_breakout is not None and max_mw_no_tensile is not None:
            mud_weight_window = {"min_SG": min_mw_no_breakout, "max_SG": max_mw_no_tensile}

        recommendations = []
        if mud_weight_window:
            width = mud_weight_window["max_SG"] - mud_weight_window["min_SG"]
            if width > 0.3:
                recommendations.append(f"Safe mud weight window: {mud_weight_window['min_SG']:.2f}--{mud_weight_window['max_SG']:.2f} SG (wide)")
            elif width > 0:
                recommendations.append(f"Narrow mud weight window: {mud_weight_window['min_SG']:.2f}--{mud_weight_window['max_SG']:.2f} SG -- careful control needed")
            else:
                recommendations.append("No safe mud weight window -- managed pressure drilling required")
        else:
            recommendations.append("Cannot establish safe mud weight window with current parameters")
        recommendations.append(f"Optimal mud weight: {optimal_mw:.2f} SG")
        recommendations.append(f"UCS={ucs_mpa:.0f} MPa used for breakout criterion")
        recommendations.append(f"Sv={Sv:.1f} MPa, SHmax={SHmax:.1f} MPa, Shmin={Shmin:.1f} MPa at {depth:.0f}m")

        plot_b64 = ""
        with plot_lock:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            mws = [r["mud_weight_SG"] for r in results_mw]
            bos = [r["breakout_width_deg"] for r in results_mw]
            sfs = [r["safety_factor"] for r in results_mw]

            axes[0].plot(mws, bos, "r-o", markersize=3, label="Breakout Width")
            axes[0].axhline(0, color="green", linestyle="--", alpha=0.5, label="No Breakout")
            if min_mw_no_breakout:
                axes[0].axvline(min_mw_no_breakout, color="blue", linestyle=":", label=f"Min safe MW={min_mw_no_breakout:.2f}")
            if max_mw_no_tensile:
                axes[0].axvline(max_mw_no_tensile, color="orange", linestyle=":", label=f"Max safe MW={max_mw_no_tensile:.2f}")
            axes[0].set_xlabel("Mud Weight (SG)")
            axes[0].set_ylabel("Breakout Width (deg)")
            axes[0].set_title("Breakout Width vs Mud Weight")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(mws, sfs, "b-o", markersize=3)
            axes[1].axhline(1.0, color="red", linestyle="--", label="SF=1.0 (failure)")
            axes[1].axhline(1.3, color="orange", linestyle=":", label="SF=1.3 (margin)")
            axes[1].set_xlabel("Mud Weight (SG)")
            axes[1].set_ylabel("Safety Factor")
            axes[1].set_title("Borehole Safety Factor vs Mud Weight")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

            fig.suptitle(f"Breakout Width Analysis — Well {well} @ {depth:.0f}m", fontsize=13, fontweight="bold")
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "well": well,
            "depth_m": depth,
            "UCS_MPa": ucs_mpa,
            "friction": friction,
            "Sv_MPa": round(Sv, 2),
            "SHmax_MPa": round(SHmax, 2),
            "Shmin_MPa": round(Shmin, 2),
            "Pp_MPa": round(Pp, 2),
            "optimal_mud_weight_SG": round(optimal_mw, 2),
            "mud_weight_window": mud_weight_window,
            "n_mud_weights_tested": len(results_mw),
            "mud_weight_analysis": results_mw[:5],
            "recommendations": recommendations,
            "plot": plot_b64,
            "stakeholder_brief": {
                "headline": f"Breakout analysis: optimal MW {optimal_mw:.2f} SG",
                "risk_level": "GREEN" if mud_weight_window and (mud_weight_window["max_SG"] - mud_weight_window["min_SG"]) > 0.3 else ("AMBER" if mud_weight_window else "RED"),
                "what_this_means": f"Analyzed breakout width for 29 mud weights at {depth:.0f}m depth.",
                "for_non_experts": "This determines the ideal drilling fluid weight to prevent borehole wall collapse (too light) or fracturing (too heavy).",
            },
        }

    result = await asyncio.to_thread(_compute)
    elapsed = round(time.time() - t0, 2)
    result["elapsed_s"] = elapsed
    _breakout_width_cache[cache_key] = result
    return _sanitize_for_json(result)


# ═══════════════════════════════════════════════════════════════════
# [183] Pore Pressure Prediction
# ═══════════════════════════════════════════════════════════════════
_pore_pressure_pred_cache = {}


@app.post("/api/analysis/pore-pressure-prediction")
async def analysis_pore_pressure_prediction(request: Request):
    """Pore pressure prediction using Eaton and Bowers methods."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    method = body.get("method", "eaton")
    n_points = int(body.get("n_points", 30))

    cache_key = f"{source}:{well}:{method}:{n_points}"
    if cache_key in _pore_pressure_pred_cache:
        cached = _pore_pressure_pred_cache[cache_key]
        cached["elapsed_s"] = round(time.time() - t0, 2)
        return _sanitize_for_json(cached)

    df = get_df(source)
    if well not in df:
        return JSONResponse(status_code=404, content={"error": f"Well '{well}' not found"})

    def _compute():
        df_well = df[well]
        n = len(df_well)
        depths_raw = df_well[DEPTH_COL].dropna().values.astype(float)
        if len(depths_raw) == 0:
            depths_raw = np.array([2000.0, 3000.0, 4000.0])

        d_min, d_max = float(np.min(depths_raw)), float(np.max(depths_raw))
        if d_max - d_min < 100:
            d_min, d_max = 500, 5000

        eval_depths = np.linspace(max(d_min, 100), d_max, n_points)
        rho_water = 1000
        rho_rock = 2500
        g = 9.81

        profile = []
        for depth_m in eval_depths:
            Sv = rho_rock * g * depth_m / 1e6
            Pp_hydro = rho_water * g * depth_m / 1e6

            if method.lower() == "eaton":
                eaton_exp = 3.0
                compaction_ratio = 1.0 - 0.00005 * depth_m
                compaction_ratio = max(0.7, min(1.0, compaction_ratio))
                Pp_pred = Sv - (Sv - Pp_hydro) * (compaction_ratio ** eaton_exp)
            else:
                A, B = 10.0, 0.8
                sigma_eff_normal = A * (depth_m / 1000.0) ** B
                unloading_factor = 1.0 + 0.0001 * max(0, depth_m - 3000)
                Pp_pred = Sv - sigma_eff_normal / unloading_factor

            Pp_pred = max(Pp_hydro * 0.9, min(Pp_pred, Sv * 0.95))
            pp_gradient = Pp_pred / (depth_m / 1000.0) if depth_m > 0 else 0
            equiv_mw = (Pp_pred * 1e6) / (g * depth_m * 1000) if depth_m > 0 else 1.0

            profile.append({
                "depth_m": round(float(depth_m), 1),
                "Sv_MPa": round(float(Sv), 3),
                "Pp_hydrostatic_MPa": round(float(Pp_hydro), 3),
                "Pp_predicted_MPa": round(float(Pp_pred), 3),
                "Pp_gradient_MPa_per_km": round(float(pp_gradient), 3),
                "equivalent_mud_weight_SG": round(float(equiv_mw), 3),
                "overpressure_ratio": round(float(Pp_pred / max(Pp_hydro, 0.01)), 3),
            })

        op_ratios = [p["overpressure_ratio"] for p in profile]
        max_op = max(op_ratios)

        if max_op > 1.2:
            pressure_regime = "OVERPRESSURED"
        elif max_op < 0.95:
            pressure_regime = "UNDERPRESSURED"
        else:
            pressure_regime = "HYDROSTATIC"

        kick_depths = [p["depth_m"] for p in profile if p["overpressure_ratio"] > 1.1]
        kick_tolerance_depth = min(kick_depths) if kick_depths else None

        recommendations = []
        method_desc = "industry standard for compaction-driven overpressure" if method.lower() == "eaton" else "better for unloading mechanisms"
        recommendations.append(f"Method: {method.upper()} -- {method_desc}")
        if pressure_regime == "OVERPRESSURED":
            recommendations.append(f"Overpressure detected (max ratio {max_op:.2f}) -- increase mud weight monitoring")
        elif pressure_regime == "HYDROSTATIC":
            recommendations.append("Near-hydrostatic pressure -- standard drilling parameters apply")
        if kick_tolerance_depth:
            recommendations.append(f"Overpressure onset at ~{kick_tolerance_depth:.0f}m -- prepare for pressure transition")
        recommendations.append(f"Predicted over {n_points} depth points from {eval_depths[0]:.0f}m to {eval_depths[-1]:.0f}m")

        plot_b64 = ""
        with plot_lock:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            d_arr = [p["depth_m"] for p in profile]
            pp_arr = [p["Pp_predicted_MPa"] for p in profile]
            ph_arr = [p["Pp_hydrostatic_MPa"] for p in profile]
            sv_arr = [p["Sv_MPa"] for p in profile]

            axes[0].plot(ph_arr, d_arr, "b--", label="Hydrostatic")
            axes[0].plot(pp_arr, d_arr, "r-", linewidth=2, label=f"Predicted ({method.upper()})")
            axes[0].plot(sv_arr, d_arr, "k-", alpha=0.5, label="Overburden (Sv)")
            axes[0].invert_yaxis()
            axes[0].set_xlabel("Pressure (MPa)")
            axes[0].set_ylabel("Depth (m)")
            axes[0].set_title("Pore Pressure Profile")
            axes[0].legend(fontsize=9)
            axes[0].grid(True, alpha=0.3)

            emw = [p["equivalent_mud_weight_SG"] for p in profile]
            axes[1].plot(emw, d_arr, "g-", linewidth=2, label="Equiv. MW")
            axes[1].axvline(1.0, color="blue", linestyle=":", label="Water (1.0 SG)")
            axes[1].invert_yaxis()
            axes[1].set_xlabel("Equivalent Mud Weight (SG)")
            axes[1].set_ylabel("Depth (m)")
            axes[1].set_title("Equivalent Mud Weight")
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3)

            fig.suptitle(f"Pore Pressure Prediction — Well {well} ({method.upper()})", fontsize=14, fontweight="bold")
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "well": well,
            "method": method.upper(),
            "n_points": n_points,
            "depth_range_m": [round(float(eval_depths[0]), 1), round(float(eval_depths[-1]), 1)],
            "pressure_regime": pressure_regime,
            "max_overpressure_ratio": round(max_op, 3),
            "kick_tolerance_depth_m": round(kick_tolerance_depth, 1) if kick_tolerance_depth else None,
            "profile": profile[:10],
            "recommendations": recommendations,
            "plot": plot_b64,
            "stakeholder_brief": {
                "headline": f"Pp prediction: {pressure_regime} (max OP ratio {max_op:.2f})",
                "risk_level": "RED" if max_op > 1.3 else ("AMBER" if max_op > 1.1 else "GREEN"),
                "what_this_means": f"{method.upper()} pore pressure prediction over {n_points} depth points.",
                "for_non_experts": "This predicts underground fluid pressure at different depths. High pressure means heavier drilling fluid is needed to prevent kicks.",
            },
        }

    result = await asyncio.to_thread(_compute)
    elapsed = round(time.time() - t0, 2)
    result["elapsed_s"] = elapsed
    _pore_pressure_pred_cache[cache_key] = result
    return _sanitize_for_json(result)


# ═══════════════════════════════════════════════════════════════════
# [184] Fault Reactivation
# ═══════════════════════════════════════════════════════════════════
_fault_reactivation_cache = {}


@app.post("/api/analysis/fault-reactivation")
async def analysis_fault_reactivation(request: Request):
    """Fault reactivation risk for specific fault orientations."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth = float(body.get("depth", 3000))
    friction = float(body.get("friction", 0.6))
    fault_azimuth = body.get("fault_azimuth")
    fault_dip = body.get("fault_dip")

    cache_key = f"{source}:{well}:{depth}:{friction}:{fault_azimuth}:{fault_dip}"
    if cache_key in _fault_reactivation_cache:
        cached = _fault_reactivation_cache[cache_key]
        cached["elapsed_s"] = round(time.time() - t0, 2)
        return _sanitize_for_json(cached)

    df = get_df(source)
    if well not in df:
        return JSONResponse(status_code=404, content={"error": f"Well '{well}' not found"})

    def _compute():
        df_well = df[well]
        n = len(df_well)
        azimuths = df_well[AZIMUTH_COL].values.astype(float)
        dips = df_well[DIP_COL].values.astype(float)

        rho_rock = 2500
        g = 9.81
        Sv = rho_rock * g * depth / 1e6
        Pp = 1000 * g * depth / 1e6

        try:
            from src.geostress import invert_stress
            inv = invert_stress(df_well, regime="NF", depth=depth, pore_pressure=Pp)
            sigma1 = float(inv.get("sigma1", Sv))
            sigma3 = float(inv.get("sigma3", Sv * 0.6))
            SHmax_az = float(inv.get("SHmax_azimuth", 0))
        except Exception:
            sigma1 = Sv
            sigma3 = Sv * 0.6
            SHmax_az = np.degrees(np.arctan2(np.mean(np.sin(np.radians(azimuths * 2))), np.mean(np.cos(np.radians(azimuths * 2))))) / 2 % 180

        if fault_azimuth is not None and fault_dip is not None:
            fault_azimuths = [float(fault_azimuth)]
            fault_dips_arr = [float(fault_dip)]
        else:
            from collections import Counter
            az_bins = (azimuths // 15) * 15
            common = Counter(zip(az_bins, (dips // 10) * 10)).most_common(5)
            fault_azimuths = [float(c[0][0]) for c in common]
            fault_dips_arr = [float(c[0][1]) for c in common]

        fault_analyses = []
        for f_az, f_dip in zip(fault_azimuths, fault_dips_arr):
            f_az_r = np.radians(f_az)
            f_dip_r = np.radians(f_dip)
            nx = np.sin(f_dip_r) * np.sin(f_az_r)
            ny = np.sin(f_dip_r) * np.cos(f_az_r)
            nz = np.cos(f_dip_r)

            sigma_n = sigma1 * nz**2 + sigma3 * (nx**2 + ny**2)
            tau = abs(sigma1 - sigma3) * abs(nz) * (nx**2 + ny**2)**0.5
            sigma_n_eff = sigma_n - Pp
            slip_tendency = tau / max(sigma_n_eff, 0.01) if sigma_n_eff > 0 else 999
            dilation_tendency = (sigma1 - sigma_n) / max(sigma1 - sigma3, 0.01)

            coulomb_margin = friction * sigma_n_eff - tau
            reactivation_risk = "HIGH" if coulomb_margin < 0 else ("MODERATE" if coulomb_margin < tau * 0.3 else "LOW")

            Pp_critical = sigma_n - tau / friction if friction > 0 else sigma_n

            fault_analyses.append({
                "fault_azimuth_deg": round(f_az, 1),
                "fault_dip_deg": round(f_dip, 1),
                "sigma_n_MPa": round(float(sigma_n), 3),
                "tau_MPa": round(float(tau), 3),
                "sigma_n_eff_MPa": round(float(sigma_n_eff), 3),
                "slip_tendency": round(float(slip_tendency), 4),
                "dilation_tendency": round(float(dilation_tendency), 4),
                "coulomb_margin_MPa": round(float(coulomb_margin), 3),
                "reactivation_risk": reactivation_risk,
                "Pp_critical_MPa": round(float(Pp_critical), 3),
                "Pp_margin_MPa": round(float(Pp_critical - Pp), 3),
            })

        n_high = sum(1 for f in fault_analyses if f["reactivation_risk"] == "HIGH")
        n_moderate = sum(1 for f in fault_analyses if f["reactivation_risk"] == "MODERATE")
        n_low = sum(1 for f in fault_analyses if f["reactivation_risk"] == "LOW")
        overall_risk = "HIGH" if n_high > 0 else ("MODERATE" if n_moderate > 0 else "LOW")

        recommendations = []
        if n_high > 0:
            recommendations.append(f"{n_high} fault(s) at HIGH reactivation risk -- avoid pressure increases near these orientations")
        if n_moderate > 0:
            recommendations.append(f"{n_moderate} fault(s) at MODERATE risk -- monitor during injection/production")
        if n_low > 0:
            recommendations.append(f"{n_low} fault(s) at LOW risk -- stable under current conditions")
        min_pp_margin = min(f["Pp_margin_MPa"] for f in fault_analyses)
        if min_pp_margin > 0:
            recommendations.append(f"Minimum Pp margin: {min_pp_margin:.1f} MPa before reactivation")
        else:
            recommendations.append("Some faults already beyond critical Pp -- reactivation expected")
        recommendations.append(f"Analysis at {depth:.0f}m depth with mu={friction:.2f}")

        plot_b64 = ""
        with plot_lock:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sigma_n_range = np.linspace(0, sigma1 * 1.1, 100)
            tau_mc = friction * (sigma_n_range - Pp)
            tau_mc = np.maximum(tau_mc, 0)
            axes[0].plot(sigma_n_range, tau_mc, "r-", linewidth=2, label=f"Coulomb (mu={friction})")

            center = (sigma1 + sigma3) / 2
            radius = (sigma1 - sigma3) / 2
            theta_arr = np.linspace(0, np.pi, 100)
            mc_sn = center + radius * np.cos(theta_arr)
            mc_tau = radius * np.sin(theta_arr)
            axes[0].plot(mc_sn, mc_tau, "b-", linewidth=1.5, label="Mohr Circle")

            colors_risk = {"HIGH": "red", "MODERATE": "orange", "LOW": "green"}
            for fa in fault_analyses:
                axes[0].plot(fa["sigma_n_MPa"], fa["tau_MPa"], "o", color=colors_risk[fa["reactivation_risk"]], markersize=8, zorder=5)
            axes[0].set_xlabel("Normal Stress (MPa)")
            axes[0].set_ylabel("Shear Stress (MPa)")
            axes[0].set_title("Fault Stress States on Mohr Diagram")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim(0, None)
            axes[0].set_ylim(0, None)

            labels_fa = [f"{fa['fault_azimuth_deg']:.0f}/{fa['fault_dip_deg']:.0f}" for fa in fault_analyses]
            margins_fa = [fa["Pp_margin_MPa"] for fa in fault_analyses]
            bar_colors = [colors_risk[fa["reactivation_risk"]] for fa in fault_analyses]
            axes[1].barh(labels_fa, margins_fa, color=bar_colors)
            axes[1].axvline(0, color="red", linestyle="--", label="Reactivation threshold")
            axes[1].set_xlabel("Pp Margin to Reactivation (MPa)")
            axes[1].set_title("Fault Reactivation Margin")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

            fig.suptitle(f"Fault Reactivation — Well {well} @ {depth:.0f}m", fontsize=13, fontweight="bold")
            fig.tight_layout()
            plot_b64 = fig_to_base64(fig)

        return {
            "well": well,
            "depth_m": depth,
            "friction": friction,
            "sigma1_MPa": round(sigma1, 2),
            "sigma3_MPa": round(sigma3, 2),
            "Pp_MPa": round(Pp, 2),
            "SHmax_azimuth_deg": round(SHmax_az, 1),
            "n_faults_analyzed": len(fault_analyses),
            "n_high_risk": n_high,
            "n_moderate_risk": n_moderate,
            "n_low_risk": n_low,
            "overall_risk": overall_risk,
            "fault_analyses": fault_analyses,
            "recommendations": recommendations,
            "plot": plot_b64,
            "stakeholder_brief": {
                "headline": f"Fault reactivation: {overall_risk} risk ({n_high} high, {n_moderate} moderate)",
                "risk_level": "RED" if overall_risk == "HIGH" else ("AMBER" if overall_risk == "MODERATE" else "GREEN"),
                "what_this_means": f"Analyzed {len(fault_analyses)} fault orientations at {depth:.0f}m for reactivation potential.",
                "for_non_experts": "This checks if underground faults could slip due to current stress conditions. High risk means fluid injection could trigger seismicity.",
            },
        }

    result = await asyncio.to_thread(_compute)
    elapsed = round(time.time() - t0, 2)
    result["elapsed_s"] = elapsed
    _fault_reactivation_cache[cache_key] = result
    return _sanitize_for_json(result)
