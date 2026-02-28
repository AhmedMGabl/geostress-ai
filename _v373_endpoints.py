

# ═══════════════════════════════════════════════════════════════════════════════
# [300] SAND FAILURE PREDICTION  (v3.73.0)
# ═══════════════════════════════════════════════════════════════════════════════
_sand_failure_cache: dict = {}

@app.post("/api/analysis/sand-failure-prediction")
async def analysis_sand_failure_prediction(request: Request):
    """Predict sand failure onset using Mohr-Coulomb and TWC (thick-wall cylinder) criteria."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_from = body.get("depth_from", 500)
    depth_to = body.get("depth_to", 5000)
    n_points = body.get("n_points", 20)
    UCS_MPa = body.get("UCS_MPa", 30)
    TWC_factor = body.get("TWC_factor", 3.0)

    ck = f"{source}_{well}_{depth_from}_{depth_to}_{n_points}_{UCS_MPa}_{TWC_factor}"
    if ck in _sand_failure_cache:
        c = _sand_failure_cache[ck]
        c["elapsed_s"] = round(time.time() - t0, 3)
        return JSONResponse(content=c)

    df_all = get_df(source)
    if df_all is None:
        return JSONResponse(content={"error": "data not loaded"}, status_code=400)
    df = df_all[df_all["well"] == well].copy()
    if df.empty:
        return JSONResponse(content={"error": f"well {well} not found"}, status_code=404)

    def _compute():
        import numpy as np
        depths = np.linspace(max(depth_from, 100), depth_to, n_points)
        profile = []
        for d in depths:
            Sv = 0.025 * d
            Pp = 0.0098 * d
            Shmin = 0.017 * d
            SHmax = 0.022 * d
            sigma_r_eff = Shmin - Pp
            sigma_theta_eff = 3 * SHmax - Shmin - Pp - Pp
            # TWC collapse pressure
            twc_strength = UCS_MPa * (1 + TWC_factor * (sigma_r_eff / (UCS_MPa + 0.001)))
            drawdown_limit = twc_strength - (sigma_theta_eff - sigma_r_eff)
            sand_risk = max(0, min(1, 1 - drawdown_limit / (UCS_MPa + 0.001)))
            profile.append({
                "depth_m": round(float(d), 1),
                "Sv_MPa": round(float(Sv), 2),
                "Pp_MPa": round(float(Pp), 2),
                "TWC_strength_MPa": round(float(twc_strength), 2),
                "drawdown_limit_MPa": round(float(drawdown_limit), 2),
                "sand_risk": round(float(sand_risk), 4),
            })
        risks = [p["sand_risk"] for p in profile]
        mean_risk = float(np.mean(risks))
        max_risk = float(np.max(risks))
        pct_critical = float(np.mean([1 for r in risks if r > 0.7]) / len(risks) * 100) if risks else 0
        if max_risk > 0.8:
            sf_class = "CRITICAL"
        elif max_risk > 0.5:
            sf_class = "HIGH_RISK"
        elif max_risk > 0.3:
            sf_class = "MODERATE"
        else:
            sf_class = "STABLE"

        recs = []
        if sf_class in ("CRITICAL", "HIGH_RISK"):
            recs.append("Install sand screens or gravel pack for sand control")
            recs.append("Reduce drawdown below TWC collapse pressure limit")
        if sf_class == "MODERATE":
            recs.append("Monitor sand production rates during flowback")
            recs.append("Consider oriented perforations to minimize hoop stress")
        if sf_class == "STABLE":
            recs.append("Sand failure risk is low — standard completion acceptable")
        recs.append(f"UCS = {UCS_MPa} MPa assumed — confirm with core testing")

        fig, ax = None, None
        plot_b64 = ""
        try:
            with plot_lock:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot([p["depth_m"] for p in profile], risks, "r-o", ms=4, label="Sand Risk Index")
                ax.axhline(0.7, color="red", ls="--", alpha=0.6, label="Critical threshold")
                ax.axhline(0.3, color="orange", ls="--", alpha=0.6, label="Moderate threshold")
                ax.set_xlabel("Depth (m)")
                ax.set_ylabel("Sand Risk Index")
                ax.set_title(f"Sand Failure Prediction — {well}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_b64 = _fig_to_base64(fig)
                plt.close(fig)
        except Exception:
            if fig:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass

        brief = {
            "headline": f"Sand failure risk is {sf_class} for {well}",
            "risk_level": sf_class,
            "what_this_means": f"Peak sand risk index is {max_risk:.2f} with {pct_critical:.0f}% of depth intervals in critical zone.",
            "for_non_experts": "Sand production occurs when the rock around the borehole fails under drawdown pressure. "
                              "High risk means sand screens or controlled drawdown rates are needed to prevent equipment damage."
        }

        return {
            "well": well,
            "depth_from_m": depth_from,
            "depth_to_m": depth_to,
            "UCS_MPa": UCS_MPa,
            "TWC_factor": TWC_factor,
            "mean_sand_risk": round(mean_risk, 4),
            "max_sand_risk": round(max_risk, 4),
            "pct_critical": round(pct_critical, 1),
            "sf_class": sf_class,
            "profile": profile,
            "recommendations": recs,
            "plot": plot_b64,
            "stakeholder_brief": brief,
        }

    result = await asyncio.to_thread(_compute)
    result["elapsed_s"] = round(time.time() - t0, 3)
    result = _sanitize_for_json(result)
    _sand_failure_cache[ck] = result
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════════════════════════════
# [301] WELLBORE BREATHING  (v3.73.0)
# ═══════════════════════════════════════════════════════════════════════════════
_wellbore_breathing_cache: dict = {}

@app.post("/api/analysis/wellbore-breathing")
async def analysis_wellbore_breathing(request: Request):
    """Analyze wellbore breathing (ballooning) — fracture open/close cycles during drilling."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_m = body.get("depth_m", 3000)
    mud_weight_ppg = body.get("mud_weight_ppg", 11)
    pump_on_ecd_ppg = body.get("pump_on_ecd_ppg", 12)

    ck = f"{source}_{well}_{depth_m}_{mud_weight_ppg}_{pump_on_ecd_ppg}"
    if ck in _wellbore_breathing_cache:
        c = _wellbore_breathing_cache[ck]
        c["elapsed_s"] = round(time.time() - t0, 3)
        return JSONResponse(content=c)

    df_all = get_df(source)
    if df_all is None:
        return JSONResponse(content={"error": "data not loaded"}, status_code=400)
    df = df_all[df_all["well"] == well].copy()
    if df.empty:
        return JSONResponse(content={"error": f"well {well} not found"}, status_code=404)

    def _compute():
        import numpy as np
        d = depth_m
        Sv = 0.025 * d
        Pp_MPa = 0.0098 * d
        Shmin = 0.017 * d
        SHmax = 0.022 * d
        frac_grad_ppg = Shmin / (0.00981 * d / 14.696 * 8.33) if d > 0 else 15
        frac_grad_ppg = Shmin / (0.0519 * d) * 8.33 if d > 0 else 15

        # MW and ECD in MPa
        mw_MPa = mud_weight_ppg * 0.0519 * d / 8.33
        ecd_MPa = pump_on_ecd_ppg * 0.0519 * d / 8.33

        # Breathing analysis
        frac_open_pressure = Shmin  # fractures open when BHP > Shmin
        frac_close_pressure = Pp_MPa + 0.5 * (Shmin - Pp_MPa)  # close at intermediate

        pumps_on_margin = frac_open_pressure - ecd_MPa
        pumps_off_margin = mw_MPa - frac_close_pressure

        # Volume exchange estimation (simplified)
        if ecd_MPa > frac_open_pressure:
            volume_loss_bbl = (ecd_MPa - frac_open_pressure) * 10  # simplified proxy
        else:
            volume_loss_bbl = 0
        if mw_MPa < frac_close_pressure:
            volume_return_bbl = (frac_close_pressure - mw_MPa) * 8
        else:
            volume_return_bbl = 0

        breathing_index = 0
        if ecd_MPa > frac_open_pressure and mw_MPa < frac_close_pressure:
            breathing_index = min(1.0, (ecd_MPa - frac_open_pressure + frac_close_pressure - mw_MPa) / (Shmin + 0.001))
        elif ecd_MPa > frac_open_pressure:
            breathing_index = min(1.0, (ecd_MPa - frac_open_pressure) / (Shmin + 0.001)) * 0.6

        if breathing_index > 0.5:
            br_class = "SEVERE"
        elif breathing_index > 0.3:
            br_class = "MODERATE"
        elif breathing_index > 0.1:
            br_class = "MILD"
        else:
            br_class = "NONE"

        # MW sweep
        mw_sweep = []
        for mw in np.arange(8, 17, 0.5):
            mw_m = mw * 0.0519 * d / 8.33
            ecd_m = (mw + (pump_on_ecd_ppg - mud_weight_ppg)) * 0.0519 * d / 8.33
            bi = 0
            if ecd_m > frac_open_pressure and mw_m < frac_close_pressure:
                bi = min(1.0, (ecd_m - frac_open_pressure + frac_close_pressure - mw_m) / (Shmin + 0.001))
            elif ecd_m > frac_open_pressure:
                bi = min(1.0, (ecd_m - frac_open_pressure) / (Shmin + 0.001)) * 0.6
            mw_sweep.append({
                "MW_ppg": round(float(mw), 1),
                "ECD_ppg": round(float(mw + (pump_on_ecd_ppg - mud_weight_ppg)), 1),
                "breathing_index": round(float(bi), 4),
            })

        recs = []
        if br_class == "SEVERE":
            recs.append("Reduce ECD by lowering flow rate or using MPD (managed pressure drilling)")
            recs.append("High risk of lost returns and kick-loss cycles — consider wellbore strengthening")
        elif br_class == "MODERATE":
            recs.append("Monitor pit volumes closely for breathing signatures")
            recs.append("Adjust MW to split the difference between frac-open and frac-close pressures")
        elif br_class == "MILD":
            recs.append("Mild breathing expected — monitor but no immediate action needed")
        else:
            recs.append("No significant breathing expected at current parameters")
        recs.append(f"Frac open pressure ~{frac_open_pressure:.1f} MPa, close ~{frac_close_pressure:.1f} MPa")

        fig, ax = None, None
        plot_b64 = ""
        try:
            with plot_lock:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                mws = [s["MW_ppg"] for s in mw_sweep]
                bis = [s["breathing_index"] for s in mw_sweep]
                ax.plot(mws, bis, "b-o", ms=4, label="Breathing Index")
                ax.axhline(0.5, color="red", ls="--", alpha=0.6, label="Severe threshold")
                ax.axhline(0.3, color="orange", ls="--", alpha=0.6, label="Moderate threshold")
                ax.axvline(mud_weight_ppg, color="green", ls=":", alpha=0.8, label=f"Current MW={mud_weight_ppg} ppg")
                ax.set_xlabel("Mud Weight (ppg)")
                ax.set_ylabel("Breathing Index")
                ax.set_title(f"Wellbore Breathing Analysis — {well} @ {depth_m}m")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_b64 = _fig_to_base64(fig)
                plt.close(fig)
        except Exception:
            if fig:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass

        brief = {
            "headline": f"Wellbore breathing is {br_class} for {well} at {depth_m}m",
            "risk_level": br_class,
            "what_this_means": f"Breathing index = {breathing_index:.3f}. ECD margin to frac open = {pumps_on_margin:.1f} MPa.",
            "for_non_experts": "Wellbore breathing is when drilling fluid is lost into fractures when pumps are on (high pressure) "
                              "and returns when pumps are off. Severe breathing causes kick-loss cycles that complicate well control."
        }

        return {
            "well": well,
            "depth_m": depth_m,
            "mud_weight_ppg": mud_weight_ppg,
            "pump_on_ecd_ppg": pump_on_ecd_ppg,
            "frac_open_MPa": round(float(frac_open_pressure), 2),
            "frac_close_MPa": round(float(frac_close_pressure), 2),
            "pumps_on_margin_MPa": round(float(pumps_on_margin), 2),
            "pumps_off_margin_MPa": round(float(pumps_off_margin), 2),
            "volume_loss_bbl": round(float(volume_loss_bbl), 1),
            "volume_return_bbl": round(float(volume_return_bbl), 1),
            "breathing_index": round(float(breathing_index), 4),
            "br_class": br_class,
            "mw_sweep": mw_sweep,
            "recommendations": recs,
            "plot": plot_b64,
            "stakeholder_brief": brief,
        }

    result = await asyncio.to_thread(_compute)
    result["elapsed_s"] = round(time.time() - t0, 3)
    result = _sanitize_for_json(result)
    _wellbore_breathing_cache[ck] = result
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════════════════════════════
# [302] SURGE-SWAB PRESSURE  (v3.73.0)
# ═══════════════════════════════════════════════════════════════════════════════
_surge_swab_cache: dict = {}

@app.post("/api/analysis/surge-swab-pressure")
async def analysis_surge_swab_pressure(request: Request):
    """Calculate surge and swab pressures during tripping operations."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_m = body.get("depth_m", 3000)
    mud_weight_ppg = body.get("mud_weight_ppg", 10)
    pipe_speed_ft_min = body.get("pipe_speed_ft_min", 90)
    hole_diameter_in = body.get("hole_diameter_in", 8.5)
    pipe_od_in = body.get("pipe_od_in", 5.0)

    ck = f"{source}_{well}_{depth_m}_{mud_weight_ppg}_{pipe_speed_ft_min}_{hole_diameter_in}_{pipe_od_in}"
    if ck in _surge_swab_cache:
        c = _surge_swab_cache[ck]
        c["elapsed_s"] = round(time.time() - t0, 3)
        return JSONResponse(content=c)

    df_all = get_df(source)
    if df_all is None:
        return JSONResponse(content={"error": "data not loaded"}, status_code=400)
    df = df_all[df_all["well"] == well].copy()
    if df.empty:
        return JSONResponse(content={"error": f"well {well} not found"}, status_code=404)

    def _compute():
        import numpy as np
        d = depth_m
        Pp_MPa = 0.0098 * d
        Shmin = 0.017 * d
        frac_grad_ppg = Shmin / (0.0519 * d) * 8.33 if d > 0 else 15
        pore_grad_ppg = Pp_MPa / (0.0519 * d) * 8.33 if d > 0 else 8.5

        # Annular area ratio
        A_hole = 3.14159 * (hole_diameter_in / 2) ** 2
        A_pipe = 3.14159 * (pipe_od_in / 2) ** 2
        A_ann = A_hole - A_pipe
        clinging_factor = A_pipe / A_ann if A_ann > 0 else 0.5

        # Surge/swab pressure (Burkhardt model simplified)
        pipe_speed_m_s = pipe_speed_ft_min * 0.3048 / 60
        PV = 20  # assumed plastic viscosity cP
        YP = 10  # assumed yield point lbf/100ft2
        ann_gap = (hole_diameter_in - pipe_od_in) / 2

        # Pressure in ppg equivalent
        surge_ppg = clinging_factor * pipe_speed_ft_min * PV / (1500 * (ann_gap ** 2) + 0.001)
        surge_ppg = min(surge_ppg, 3.0)
        swab_ppg = surge_ppg * 0.9  # swab slightly less than surge

        surge_eqmw = mud_weight_ppg + surge_ppg
        swab_eqmw = mud_weight_ppg - swab_ppg

        frac_margin_ppg = frac_grad_ppg - surge_eqmw
        kick_margin_ppg = swab_eqmw - pore_grad_ppg

        if frac_margin_ppg < 0 or kick_margin_ppg < 0:
            ss_class = "CRITICAL"
        elif frac_margin_ppg < 0.5 or kick_margin_ppg < 0.3:
            ss_class = "TIGHT"
        elif frac_margin_ppg < 1.0 or kick_margin_ppg < 0.5:
            ss_class = "MODERATE"
        else:
            ss_class = "SAFE"

        # Speed sweep
        speed_sweep = []
        for spd in np.arange(30, 210, 15):
            sp = clinging_factor * spd * PV / (1500 * (ann_gap ** 2) + 0.001)
            sp = min(sp, 3.0)
            sw = sp * 0.9
            speed_sweep.append({
                "speed_ft_min": round(float(spd), 0),
                "surge_ppg": round(float(sp), 3),
                "swab_ppg": round(float(sw), 3),
                "surge_eqmw_ppg": round(float(mud_weight_ppg + sp), 2),
                "swab_eqmw_ppg": round(float(mud_weight_ppg - sw), 2),
            })

        recs = []
        if ss_class == "CRITICAL":
            recs.append("Reduce tripping speed immediately — risk of fracture or kick")
            recs.append("Consider MPD or controlled tripping procedures")
        elif ss_class == "TIGHT":
            recs.append("Limit tripping speed below 60 ft/min in tight margin sections")
            recs.append("Monitor flowback closely during pipe movement")
        elif ss_class == "MODERATE":
            recs.append("Standard tripping speed acceptable with monitoring")
        else:
            recs.append("Adequate surge/swab margins — normal operations safe")
        recs.append(f"Surge: +{surge_ppg:.2f} ppg, Swab: -{swab_ppg:.2f} ppg at {pipe_speed_ft_min} ft/min")

        fig, ax = None, None
        plot_b64 = ""
        try:
            with plot_lock:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                speeds = [s["speed_ft_min"] for s in speed_sweep]
                surges = [s["surge_eqmw_ppg"] for s in speed_sweep]
                swabs = [s["swab_eqmw_ppg"] for s in speed_sweep]
                ax.plot(speeds, surges, "r-o", ms=4, label="Surge (EQMW)")
                ax.plot(speeds, swabs, "b-s", ms=4, label="Swab (EQMW)")
                ax.axhline(frac_grad_ppg, color="red", ls="--", alpha=0.6, label=f"Frac grad = {frac_grad_ppg:.1f} ppg")
                ax.axhline(pore_grad_ppg, color="blue", ls="--", alpha=0.6, label=f"Pore grad = {pore_grad_ppg:.1f} ppg")
                ax.axhline(mud_weight_ppg, color="green", ls=":", alpha=0.8, label=f"MW = {mud_weight_ppg} ppg")
                ax.set_xlabel("Tripping Speed (ft/min)")
                ax.set_ylabel("Equivalent MW (ppg)")
                ax.set_title(f"Surge/Swab Pressure — {well} @ {depth_m}m")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                plot_b64 = _fig_to_base64(fig)
                plt.close(fig)
        except Exception:
            if fig:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass

        brief = {
            "headline": f"Surge/swab margins are {ss_class} for {well} at {depth_m}m",
            "risk_level": ss_class,
            "what_this_means": f"Surge adds {surge_ppg:.2f} ppg, swab subtracts {swab_ppg:.2f} ppg. Frac margin = {frac_margin_ppg:.2f} ppg, kick margin = {kick_margin_ppg:.2f} ppg.",
            "for_non_experts": "When drill pipe moves up or down, it pushes (surge) or pulls (swab) the mud, changing bottom-hole pressure. "
                              "Too much surge can fracture the rock; too much swab can cause a kick (influx of formation fluid)."
        }

        return {
            "well": well,
            "depth_m": depth_m,
            "mud_weight_ppg": mud_weight_ppg,
            "pipe_speed_ft_min": pipe_speed_ft_min,
            "hole_diameter_in": hole_diameter_in,
            "pipe_od_in": pipe_od_in,
            "surge_ppg": round(float(surge_ppg), 3),
            "swab_ppg": round(float(swab_ppg), 3),
            "surge_eqmw_ppg": round(float(surge_eqmw), 2),
            "swab_eqmw_ppg": round(float(swab_eqmw), 2),
            "frac_grad_ppg": round(float(frac_grad_ppg), 2),
            "pore_grad_ppg": round(float(pore_grad_ppg), 2),
            "frac_margin_ppg": round(float(frac_margin_ppg), 3),
            "kick_margin_ppg": round(float(kick_margin_ppg), 3),
            "ss_class": ss_class,
            "speed_sweep": speed_sweep,
            "recommendations": recs,
            "plot": plot_b64,
            "stakeholder_brief": brief,
        }

    result = await asyncio.to_thread(_compute)
    result["elapsed_s"] = round(time.time() - t0, 3)
    result = _sanitize_for_json(result)
    _surge_swab_cache[ck] = result
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════════════════════════════
# [303] LOST CIRCULATION RISK  (v3.73.0)
# ═══════════════════════════════════════════════════════════════════════════════
_lost_circ_risk_cache: dict = {}

@app.post("/api/analysis/lost-circulation-risk")
async def analysis_lost_circulation_risk(request: Request):
    """Assess lost circulation risk based on MW window, fracture density, and stress state."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_from = body.get("depth_from", 500)
    depth_to = body.get("depth_to", 5000)
    n_points = body.get("n_points", 20)
    mud_weight_ppg = body.get("mud_weight_ppg", 11)

    ck = f"{source}_{well}_{depth_from}_{depth_to}_{n_points}_{mud_weight_ppg}"
    if ck in _lost_circ_risk_cache:
        c = _lost_circ_risk_cache[ck]
        c["elapsed_s"] = round(time.time() - t0, 3)
        return JSONResponse(content=c)

    df_all = get_df(source)
    if df_all is None:
        return JSONResponse(content={"error": "data not loaded"}, status_code=400)
    df = df_all[df_all["well"] == well].copy()
    if df.empty:
        return JSONResponse(content={"error": f"well {well} not found"}, status_code=404)

    def _compute():
        import numpy as np
        depths = np.linspace(max(depth_from, 100), depth_to, n_points)
        depth_col = DEPTH_COL if DEPTH_COL in df.columns else df.columns[0]
        well_depths = df[depth_col].dropna().values

        profile = []
        for d in depths:
            Shmin = 0.017 * d
            Pp = 0.0098 * d
            frac_grad_ppg = Shmin / (0.0519 * d) * 8.33 if d > 0 else 15
            pore_grad_ppg = Pp / (0.0519 * d) * 8.33 if d > 0 else 8.5
            mw_window = frac_grad_ppg - pore_grad_ppg

            # Fracture density near this depth (count within ±50m)
            nearby = np.sum(np.abs(well_depths - d) < 50)
            frac_density = float(nearby) / 100.0  # per meter

            # Overbalance
            mw_MPa = mud_weight_ppg * 0.0519 * d / 8.33
            overbalance_ppg = mud_weight_ppg - pore_grad_ppg
            overbalance_frac = frac_grad_ppg - mud_weight_ppg

            # LC risk: high when close to frac grad AND high fracture density
            lc_risk = 0
            if overbalance_frac < 0:
                lc_risk = 1.0
            elif overbalance_frac < 0.5:
                lc_risk = 0.8 + frac_density * 0.2
            elif overbalance_frac < 1.0:
                lc_risk = 0.4 + frac_density * 0.4
            else:
                lc_risk = max(0, frac_density * 0.3)
            lc_risk = min(1.0, max(0, lc_risk))

            profile.append({
                "depth_m": round(float(d), 1),
                "frac_grad_ppg": round(float(frac_grad_ppg), 2),
                "pore_grad_ppg": round(float(pore_grad_ppg), 2),
                "mw_window_ppg": round(float(mw_window), 2),
                "frac_density_per_m": round(float(frac_density), 4),
                "overbalance_to_frac_ppg": round(float(overbalance_frac), 3),
                "lc_risk": round(float(lc_risk), 4),
            })

        risks = [p["lc_risk"] for p in profile]
        mean_risk = float(np.mean(risks))
        max_risk = float(np.max(risks))
        pct_high = float(np.mean([1 for r in risks if r > 0.6]) / len(risks) * 100) if risks else 0

        if max_risk > 0.8:
            lc_class = "SEVERE"
        elif max_risk > 0.5:
            lc_class = "HIGH"
        elif max_risk > 0.3:
            lc_class = "MODERATE"
        else:
            lc_class = "LOW"

        recs = []
        if lc_class in ("SEVERE", "HIGH"):
            recs.append("Pre-treat with LCM (lost circulation material) before drilling into high-risk zones")
            recs.append("Consider wellbore strengthening techniques (stress caging)")
            recs.append("Set casing above high-risk lost circulation zones")
        elif lc_class == "MODERATE":
            recs.append("Keep LCM readily available on rig site")
            recs.append("Monitor mud losses and adjust MW if needed")
        else:
            recs.append("Lost circulation risk is low — standard drilling procedures adequate")
        recs.append(f"MW = {mud_weight_ppg} ppg; {pct_high:.0f}% of intervals at high LC risk")

        fig, ax = None, None
        plot_b64 = ""
        try:
            with plot_lock:
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                dd = [p["depth_m"] for p in profile]
                ax1.plot(dd, [p["frac_grad_ppg"] for p in profile], "r-", label="Frac Grad")
                ax1.plot(dd, [p["pore_grad_ppg"] for p in profile], "b-", label="Pore Grad")
                ax1.axvline(0, visible=False)
                ax1.axhline(mud_weight_ppg, color="green", ls="--", label=f"MW = {mud_weight_ppg} ppg")
                ax1.set_xlabel("Depth (m)")
                ax1.set_ylabel("Pressure (ppg)")
                ax1.set_title("MW Window")
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                ax2.plot(dd, risks, "r-o", ms=3, label="LC Risk")
                ax2.axhline(0.6, color="red", ls="--", alpha=0.6, label="High threshold")
                ax2.fill_between(dd, risks, alpha=0.3, color="red")
                ax2.set_xlabel("Depth (m)")
                ax2.set_ylabel("LC Risk Index")
                ax2.set_title(f"Lost Circulation Risk — {well}")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

                fig.tight_layout()
                plot_b64 = _fig_to_base64(fig)
                plt.close(fig)
        except Exception:
            if fig:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass

        brief = {
            "headline": f"Lost circulation risk is {lc_class} for {well}",
            "risk_level": lc_class,
            "what_this_means": f"Mean LC risk = {mean_risk:.3f}, max = {max_risk:.3f}. {pct_high:.0f}% of depth intervals at high risk.",
            "for_non_experts": "Lost circulation means drilling fluid flows into the rock formation instead of returning to surface. "
                              "This wastes expensive mud, can cause well control issues, and delays drilling operations."
        }

        return {
            "well": well,
            "depth_from_m": depth_from,
            "depth_to_m": depth_to,
            "mud_weight_ppg": mud_weight_ppg,
            "mean_lc_risk": round(mean_risk, 4),
            "max_lc_risk": round(max_risk, 4),
            "pct_high_risk": round(pct_high, 1),
            "lc_class": lc_class,
            "profile": profile,
            "recommendations": recs,
            "plot": plot_b64,
            "stakeholder_brief": brief,
        }

    result = await asyncio.to_thread(_compute)
    result["elapsed_s"] = round(time.time() - t0, 3)
    result = _sanitize_for_json(result)
    _lost_circ_risk_cache[ck] = result
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════════════════════════════
# [304] HOLE CLEANING EFFICIENCY  (v3.73.0)
# ═══════════════════════════════════════════════════════════════════════════════
_hole_cleaning_eff_cache: dict = {}

@app.post("/api/analysis/hole-cleaning-efficiency")
async def analysis_hole_cleaning_efficiency(request: Request):
    """Evaluate hole cleaning efficiency based on annular velocity, inclination, and cuttings transport."""
    t0 = time.time()
    body = await request.json()
    source = body.get("source", "demo")
    well = body.get("well", "3P")
    depth_from = body.get("depth_from", 500)
    depth_to = body.get("depth_to", 5000)
    flow_rate_gpm = body.get("flow_rate_gpm", 500)
    hole_diameter_in = body.get("hole_diameter_in", 8.5)
    pipe_od_in = body.get("pipe_od_in", 5.0)
    mud_weight_ppg = body.get("mud_weight_ppg", 10)
    inclination_deg = body.get("inclination_deg", 0)

    ck = f"{source}_{well}_{depth_from}_{depth_to}_{flow_rate_gpm}_{hole_diameter_in}_{pipe_od_in}_{mud_weight_ppg}_{inclination_deg}"
    if ck in _hole_cleaning_eff_cache:
        c = _hole_cleaning_eff_cache[ck]
        c["elapsed_s"] = round(time.time() - t0, 3)
        return JSONResponse(content=c)

    df_all = get_df(source)
    if df_all is None:
        return JSONResponse(content={"error": "data not loaded"}, status_code=400)
    df = df_all[df_all["well"] == well].copy()
    if df.empty:
        return JSONResponse(content={"error": f"well {well} not found"}, status_code=404)

    def _compute():
        import numpy as np
        n_points = 20
        depths = np.linspace(max(depth_from, 100), depth_to, n_points)

        # Annular area
        A_hole = 3.14159 * (hole_diameter_in / 2) ** 2  # in²
        A_pipe = 3.14159 * (pipe_od_in / 2) ** 2
        A_ann = A_hole - A_pipe  # in²

        # Annular velocity
        ann_velocity_ft_min = flow_rate_gpm / (2.448 * A_ann) if A_ann > 0 else 0
        ann_velocity_ft_s = ann_velocity_ft_min / 60

        # Cuttings transport ratio depends on inclination
        # Vertical: gravity helps, cuttings settle axially
        # 30-60°: worst case (cuttings bed on low side)
        # Horizontal: stable bed, need high AV

        incl_rad = inclination_deg * 3.14159 / 180
        # Larsen model simplified: CTR factor
        if inclination_deg < 30:
            incl_factor = 1.0
        elif inclination_deg < 60:
            incl_factor = 0.6  # worst zone
        else:
            incl_factor = 0.7  # horizontal

        # Min transport velocity (ft/min) — Luo/Bern/Erdemir type correlation
        PV = 15 + mud_weight_ppg * 0.5  # assumed PV
        min_transport_vel = 120 * incl_factor  # ft/min base

        profile = []
        for d in depths:
            # At each depth, vary eccentricity slightly
            ecc = 0.3 + 0.4 * (d - depths[0]) / (depths[-1] - depths[0] + 1)  # 0.3 to 0.7
            eff_av = ann_velocity_ft_min * (1 - 0.3 * ecc)  # eccentricity reduces effective AV

            transport_ratio = eff_av / (min_transport_vel + 0.001)
            cleaning_eff = min(1.0, max(0, transport_ratio * 0.8))  # 80% of ratio as efficiency

            # Cuttings concentration (higher = worse)
            cuttings_pct = max(0, (1 - cleaning_eff) * 15)  # up to 15% by volume

            profile.append({
                "depth_m": round(float(d), 1),
                "eff_ann_velocity_ft_min": round(float(eff_av), 1),
                "transport_ratio": round(float(transport_ratio), 3),
                "cleaning_efficiency": round(float(cleaning_eff), 4),
                "cuttings_vol_pct": round(float(cuttings_pct), 2),
            })

        effs = [p["cleaning_efficiency"] for p in profile]
        mean_eff = float(np.mean(effs))
        min_eff = float(np.min(effs))
        pct_poor = float(np.mean([1 for e in effs if e < 0.6]) / len(effs) * 100) if effs else 0

        if min_eff < 0.4:
            hc_class = "POOR"
        elif min_eff < 0.6:
            hc_class = "MARGINAL"
        elif min_eff < 0.8:
            hc_class = "ADEQUATE"
        else:
            hc_class = "GOOD"

        # Flow rate sweep
        flow_sweep = []
        for fr in np.arange(200, 900, 50):
            av = fr / (2.448 * A_ann) if A_ann > 0 else 0
            eff_av_mid = av * (1 - 0.3 * 0.5)  # mid eccentricity
            tr = eff_av_mid / (min_transport_vel + 0.001)
            ce = min(1.0, max(0, tr * 0.8))
            flow_sweep.append({
                "flow_rate_gpm": round(float(fr), 0),
                "ann_velocity_ft_min": round(float(av), 1),
                "cleaning_efficiency": round(float(ce), 4),
            })

        recs = []
        if hc_class == "POOR":
            recs.append("Increase flow rate or use high-viscosity sweeps for hole cleaning")
            recs.append("Rotate pipe continuously to prevent cuttings beds")
            recs.append("Consider wiper trips before casing runs")
        elif hc_class == "MARGINAL":
            recs.append("Increase flow rate to improve annular velocity")
            recs.append("Schedule regular viscous sweeps in deviated sections")
        elif hc_class == "ADEQUATE":
            recs.append("Hole cleaning is acceptable — monitor cuttings returns")
        else:
            recs.append("Excellent hole cleaning — maintain current flow rate")
        recs.append(f"Annular velocity = {ann_velocity_ft_min:.0f} ft/min (min transport = {min_transport_vel:.0f} ft/min)")

        fig, ax = None, None
        plot_b64 = ""
        try:
            with plot_lock:
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                dd = [p["depth_m"] for p in profile]
                ax1.plot(dd, effs, "g-o", ms=4, label="Cleaning Efficiency")
                ax1.axhline(0.6, color="orange", ls="--", alpha=0.6, label="Marginal threshold")
                ax1.axhline(0.4, color="red", ls="--", alpha=0.6, label="Poor threshold")
                ax1.fill_between(dd, effs, alpha=0.2, color="green")
                ax1.set_xlabel("Depth (m)")
                ax1.set_ylabel("Cleaning Efficiency")
                ax1.set_title(f"Hole Cleaning — {well}")
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                frs = [s["flow_rate_gpm"] for s in flow_sweep]
                ces = [s["cleaning_efficiency"] for s in flow_sweep]
                ax2.plot(frs, ces, "b-s", ms=4, label="Cleaning Efficiency")
                ax2.axvline(flow_rate_gpm, color="green", ls=":", alpha=0.8, label=f"Current = {flow_rate_gpm} gpm")
                ax2.axhline(0.6, color="orange", ls="--", alpha=0.6)
                ax2.set_xlabel("Flow Rate (gpm)")
                ax2.set_ylabel("Cleaning Efficiency")
                ax2.set_title("Flow Rate Sensitivity")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

                fig.tight_layout()
                plot_b64 = _fig_to_base64(fig)
                plt.close(fig)
        except Exception:
            if fig:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass

        brief = {
            "headline": f"Hole cleaning is {hc_class} for {well}",
            "risk_level": hc_class,
            "what_this_means": f"Mean cleaning efficiency = {mean_eff:.1%}, min = {min_eff:.1%}. {pct_poor:.0f}% of intervals have poor cleaning.",
            "for_non_experts": "Hole cleaning measures how well cuttings are removed from the borehole during drilling. "
                              "Poor cleaning leads to stuck pipe, high torque/drag, and pack-off events that delay operations."
        }

        return {
            "well": well,
            "depth_from_m": depth_from,
            "depth_to_m": depth_to,
            "flow_rate_gpm": flow_rate_gpm,
            "hole_diameter_in": hole_diameter_in,
            "pipe_od_in": pipe_od_in,
            "mud_weight_ppg": mud_weight_ppg,
            "inclination_deg": inclination_deg,
            "ann_velocity_ft_min": round(float(ann_velocity_ft_min), 1),
            "mean_cleaning_eff": round(mean_eff, 4),
            "min_cleaning_eff": round(min_eff, 4),
            "pct_poor": round(pct_poor, 1),
            "hc_class": hc_class,
            "profile": profile,
            "flow_sweep": flow_sweep,
            "recommendations": recs,
            "plot": plot_b64,
            "stakeholder_brief": brief,
        }

    result = await asyncio.to_thread(_compute)
    result["elapsed_s"] = round(time.time() - t0, 3)
    result = _sanitize_for_json(result)
    _hole_cleaning_eff_cache[ck] = result
    return JSONResponse(content=result)
