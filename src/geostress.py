"""
Geostress Inversion Engine.

Estimates the full in-situ stress tensor from fracture orientation data
using Mohr-Coulomb frictional failure theory, nonlinear optimization,
and Bayesian MCMC inference.

Based on the methodology in Geostress_DEL_16Feb26.docx:
- Fracture plane normals from azimuth/dip
- Stress tensor parameterization (σ1, σ2, σ3, SHmax azimuth, R ratio)
- Mohr-Coulomb slip criterion as constraint
- Scipy least-squares for initial estimate
- emcee MCMC for Bayesian uncertainty quantification
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation


# ──────────────────────────────────────────────
# Stress Tensor Construction
# ──────────────────────────────────────────────

def build_stress_tensor(
    sigma1: float, sigma3: float, R: float, shmax_azimuth_deg: float, regime: str = "normal"
) -> np.ndarray:
    """Build 3x3 stress tensor in geographic coordinates (E, N, Down).

    Parameters
    ----------
    sigma1 : Maximum principal stress magnitude (MPa)
    sigma3 : Minimum principal stress magnitude (MPa)
    R : Stress ratio = (σ2 - σ3) / (σ1 - σ3), in [0, 1]
    shmax_azimuth_deg : Azimuth of max horizontal stress from North (degrees)
    regime : Faulting regime - 'normal', 'strike_slip', or 'thrust'

    Returns
    -------
    S : (3,3) stress tensor in [East, North, Down] coordinates
    """
    sigma2 = sigma3 + R * (sigma1 - sigma3)

    # Principal stresses in principal coordinate system
    # Regime determines which principal axis is vertical:
    if regime == "normal":
        # σ1 = σv (vertical), σ2 = SHmax, σ3 = Shmin
        s_principal = np.diag([sigma3, sigma2, sigma1])  # [Shmin, SHmax, Sv]
    elif regime == "strike_slip":
        # σ2 = σv (vertical), σ1 = SHmax, σ3 = Shmin
        s_principal = np.diag([sigma3, sigma1, sigma2])  # [Shmin, SHmax, Sv]
    elif regime == "thrust":
        # σ3 = σv (vertical), σ1 = SHmax, σ2 = Shmin
        s_principal = np.diag([sigma2, sigma1, sigma3])  # [Shmin, SHmax, Sv]
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Rotate from principal to geographic: SHmax along azimuth
    az_rad = np.radians(shmax_azimuth_deg)
    # Rotation about vertical axis (Down) by SHmax azimuth
    cos_a, sin_a = np.cos(az_rad), np.sin(az_rad)
    # Maps [Shmin_dir, SHmax_dir, Down] -> [East, North, Down]
    rot = np.array([
        [sin_a, cos_a, 0],   # East component
        [cos_a, -sin_a, 0],  # North component (SHmax=N when az=0)
        [0, 0, 1],           # Down
    ])

    S = rot @ s_principal @ rot.T
    return S


# ──────────────────────────────────────────────
# Traction and Stress Resolution on Planes
# ──────────────────────────────────────────────

def resolve_stress_on_planes(S: np.ndarray, normals: np.ndarray) -> tuple:
    """Resolve stress tensor onto fracture planes.

    Parameters
    ----------
    S : (3,3) stress tensor
    normals : (N, 3) unit normal vectors of fracture planes

    Returns
    -------
    sigma_n : (N,) normal stress on each plane
    tau : (N,) shear stress magnitude on each plane
    """
    # Traction vector: t = S · n
    traction = normals @ S.T  # (N, 3)

    # Normal stress: σn = n · t
    sigma_n = np.sum(normals * traction, axis=1)

    # Shear stress: τ = |t - σn·n|
    normal_component = sigma_n[:, np.newaxis] * normals
    shear_vector = traction - normal_component
    tau = np.linalg.norm(shear_vector, axis=1)

    return sigma_n, tau


# ──────────────────────────────────────────────
# Mohr-Coulomb Criterion
# ──────────────────────────────────────────────

def mohr_coulomb_misfit(sigma_n: np.ndarray, tau: np.ndarray,
                        mu: float = 0.6, cohesion: float = 0.0,
                        pore_pressure: float = 0.0) -> np.ndarray:
    """Compute misfit from Mohr-Coulomb failure envelope.

    Uses effective stress principle (Terzaghi):
        τ = cohesion + μ · (σn - Pp)

    where Pp is pore fluid pressure.

    Misfit is the signed distance below the failure line (positive = below,
    meaning the stress state couldn't have caused slip on this plane).

    Parameters
    ----------
    sigma_n : Normal stresses
    tau : Shear stresses
    mu : Friction coefficient (typically 0.6 for Byerlee's law)
    cohesion : Cohesive strength (MPa)
    pore_pressure : Pore fluid pressure in MPa (shifts failure envelope)

    Returns
    -------
    misfit : Array of misfits (positive = fracture is below failure line)
    """
    sigma_n_eff = sigma_n - pore_pressure
    tau_critical = cohesion + mu * sigma_n_eff
    return tau_critical - tau  # positive means tau < tau_critical (not failing)


# ──────────────────────────────────────────────
# Slip and Dilation Tendency
# ──────────────────────────────────────────────

def slip_tendency(sigma_n: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Slip tendency Ts = τ / σn (Morris et al., 1996)."""
    return np.where(sigma_n > 0, tau / sigma_n, 0.0)


def dilation_tendency(sigma_n: np.ndarray, sigma1: float, sigma3: float) -> np.ndarray:
    """Dilation tendency Td = (σ1 - σn) / (σ1 - σ3)."""
    denom = sigma1 - sigma3
    if denom < 1e-10:
        return np.zeros_like(sigma_n)
    return (sigma1 - sigma_n) / denom


# ──────────────────────────────────────────────
# Inversion: Objective Function
# ──────────────────────────────────────────────

def _pack_params(sigma1, sigma3, R, shmax_az, mu):
    return np.array([sigma1, sigma3, R, shmax_az, mu])


def _unpack_params(params):
    return params[0], params[1], params[2], params[3], params[4]


def inversion_objective(params: np.ndarray, normals: np.ndarray,
                        regime: str = "strike_slip", cohesion: float = 0.0,
                        pore_pressure: float = 0.0) -> float:
    """Objective function for geostress inversion.

    Minimizes the sum of squared Mohr-Coulomb misfits across all fractures.
    The idea: the correct stress tensor should place all fractures at or
    above the Mohr-Coulomb failure envelope (misfit ≤ 0).
    Uses effective stress principle when pore_pressure > 0.
    """
    sigma1, sigma3, R, shmax_az, mu = _unpack_params(params)

    # Physical constraints
    if sigma1 <= sigma3 or R < 0 or R > 1 or mu < 0.1 or mu > 1.2:
        return 1e10

    S = build_stress_tensor(sigma1, sigma3, R, shmax_az, regime)
    sigma_n, tau = resolve_stress_on_planes(S, normals)

    misfit = mohr_coulomb_misfit(sigma_n, tau, mu, cohesion, pore_pressure)

    # Penalize fractures that are far below the failure line
    # (they should all be near or on the line)
    # Also slightly penalize fractures too far above (prefer tight fit)
    penalty = np.sum(misfit**2) + 0.1 * np.sum(np.maximum(-misfit, 0)**2)

    return penalty


# ──────────────────────────────────────────────
# Inversion: Optimization
# ──────────────────────────────────────────────

def invert_stress(
    normals: np.ndarray,
    regime: str = "strike_slip",
    depth_m: float = 3000.0,
    cohesion: float = 0.0,
    pore_pressure: float = None,
) -> dict:
    """Run geostress inversion to estimate the stress tensor.

    Uses differential evolution (global optimization) followed by
    local refinement.  Supports pore pressure correction for
    effective stress analysis.

    Parameters
    ----------
    normals : (N, 3) fracture plane normal vectors
    regime : Faulting regime assumption
    depth_m : Average depth (m) for initial stress magnitude bounds
    cohesion : Rock cohesion (MPa)
    pore_pressure : Pore fluid pressure in MPa.
        If None, estimates hydrostatic: ρ_water * g * depth.

    Returns
    -------
    result : dict with keys:
        sigma1, sigma3, R, shmax_azimuth_deg, mu, stress_tensor,
        sigma_n, tau, slip_tend, dilation_tend, misfit,
        pore_pressure, effective_sigma_n
    """
    # Estimate pore pressure if not provided (hydrostatic assumption)
    if pore_pressure is None:
        pore_pressure = 1020.0 * 9.81 * depth_m / 1e6  # ρw*g*h in MPa

    # Estimate reasonable stress bounds from depth
    # Typical: σv ≈ ρgh ≈ 2500 * 9.81 * depth / 1e6 MPa
    sv_est = 2500 * 9.81 * depth_m / 1e6  # ~73 MPa at 3000m

    bounds = [
        (sv_est * 0.5, sv_est * 2.5),   # sigma1
        (sv_est * 0.2, sv_est * 1.5),   # sigma3
        (0.01, 0.99),                     # R
        (0.0, 360.0),                     # SHmax azimuth
        (0.3, 1.0),                       # mu (friction)
    ]

    # Global optimization
    result = differential_evolution(
        inversion_objective,
        bounds,
        args=(normals, regime, cohesion, pore_pressure),
        maxiter=500,
        seed=42,
        tol=1e-8,
        polish=True,
    )

    sigma1, sigma3, R, shmax_az, mu = _unpack_params(result.x)

    # Build final stress tensor and compute all outputs
    S = build_stress_tensor(sigma1, sigma3, R, shmax_az, regime)
    sigma_n, tau = resolve_stress_on_planes(S, normals)
    misfit = mohr_coulomb_misfit(sigma_n, tau, mu, cohesion, pore_pressure)

    sigma2 = sigma3 + R * (sigma1 - sigma3)

    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
        "R": R,
        "shmax_azimuth_deg": shmax_az,
        "mu": mu,
        "regime": regime,
        "stress_tensor": S,
        "sigma_n": sigma_n,
        "tau": tau,
        "slip_tend": slip_tendency(sigma_n, tau),
        "dilation_tend": dilation_tendency(sigma_n, sigma1, sigma3),
        "misfit": misfit,
        "pore_pressure": pore_pressure,
        "effective_sigma_n": sigma_n - pore_pressure,
        "optimization_result": result,
    }


# ──────────────────────────────────────────────
# Bayesian MCMC Inference
# ──────────────────────────────────────────────

def log_prior(params: np.ndarray, bounds: list) -> float:
    """Uniform prior within bounds."""
    for val, (lo, hi) in zip(params, bounds):
        if val < lo or val > hi:
            return -np.inf
    return 0.0


def log_likelihood(params: np.ndarray, normals: np.ndarray,
                   regime: str, cohesion: float, pore_pressure: float = 0.0,
                   sigma_obs: float = 5.0) -> float:
    """Log-likelihood assuming Gaussian errors on Mohr-Coulomb misfit."""
    sigma1, sigma3, R, shmax_az, mu = _unpack_params(params)
    if sigma1 <= sigma3:
        return -np.inf

    S = build_stress_tensor(sigma1, sigma3, R, shmax_az, regime)
    sigma_n, tau = resolve_stress_on_planes(S, normals)
    misfit = mohr_coulomb_misfit(sigma_n, tau, mu, cohesion, pore_pressure)

    # Gaussian log-likelihood
    return -0.5 * np.sum((misfit / sigma_obs) ** 2)


def log_posterior(params, normals, regime, cohesion, pore_pressure, bounds, sigma_obs):
    lp = log_prior(params, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, normals, regime, cohesion, pore_pressure, sigma_obs)


def bayesian_inversion(
    normals: np.ndarray,
    initial_result: dict,
    regime: str = "strike_slip",
    cohesion: float = 0.0,
    pore_pressure: float = 0.0,
    depth_m: float = 3000.0,
    nwalkers: int = 32,
    nsteps: int = 2000,
    burnin: int = 500,
    fast: bool = False,
) -> dict:
    """Run Bayesian MCMC inversion using emcee.

    Parameters
    ----------
    normals : (N, 3) fracture plane normals
    initial_result : dict from invert_stress() for initialization
    regime, cohesion : Same as invert_stress
    pore_pressure : Pore fluid pressure in MPa
    depth_m : Average depth for bound estimation
    nwalkers : Number of MCMC walkers
    nsteps : Total steps per walker
    burnin : Steps to discard as burn-in
    fast : If True, use reduced steps (500/150) for quicker results

    Returns
    -------
    dict with posterior statistics (no raw samples for API serialization)
    """
    try:
        import emcee
    except ImportError:
        return {
            "available": False,
            "error": "emcee not installed. Run: pip install emcee",
        }

    if fast:
        nsteps = 500
        burnin = 150
        nwalkers = 16

    sv_est = 2500 * 9.81 * depth_m / 1e6

    bounds = [
        (sv_est * 0.3, sv_est * 3.0),
        (sv_est * 0.1, sv_est * 2.0),
        (0.01, 0.99),
        (0.0, 360.0),
        (0.2, 1.2),
    ]

    # Initialize walkers near the optimization result
    p0_center = _pack_params(
        initial_result["sigma1"],
        initial_result["sigma3"],
        initial_result["R"],
        initial_result["shmax_azimuth_deg"],
        initial_result["mu"],
    )

    ndim = 5
    # Small perturbations around the best-fit
    p0 = p0_center + 1e-3 * np.abs(p0_center) * np.random.randn(nwalkers, ndim)
    # Clip to bounds
    for i, (lo, hi) in enumerate(bounds):
        p0[:, i] = np.clip(p0[:, i], lo + 1e-6, hi - 1e-6)

    sigma_obs = 5.0

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=(normals, regime, cohesion, pore_pressure, bounds, sigma_obs),
    )

    sampler.run_mcmc(p0, nsteps, progress=False)

    # Extract posterior samples (after burn-in)
    samples = sampler.get_chain(discard=burnin, flat=True)
    param_names = ["sigma1", "sigma3", "R", "SHmax_azimuth", "mu"]

    # Compute sigma2 for each sample
    sigma2_samples = samples[:, 1] + samples[:, 2] * (samples[:, 0] - samples[:, 1])

    # Summary statistics
    medians = np.median(samples, axis=0)
    stds = np.std(samples, axis=0)
    q5 = np.percentile(samples, 5, axis=0)
    q16 = np.percentile(samples, 16, axis=0)
    q84 = np.percentile(samples, 84, axis=0)
    q95 = np.percentile(samples, 95, axis=0)

    # Per-parameter results for frontend display
    parameters = {}
    for i, name in enumerate(param_names):
        parameters[name] = {
            "median": round(float(medians[i]), 3),
            "std": round(float(stds[i]), 3),
            "ci_68": [round(float(q16[i]), 3), round(float(q84[i]), 3)],
            "ci_90": [round(float(q5[i]), 3), round(float(q95[i]), 3)],
            "best_fit": round(float(p0_center[i]), 3),
        }

    # Sigma2 derived statistics
    parameters["sigma2"] = {
        "median": round(float(np.median(sigma2_samples)), 3),
        "std": round(float(np.std(sigma2_samples)), 3),
        "ci_68": [round(float(np.percentile(sigma2_samples, 16)), 3),
                   round(float(np.percentile(sigma2_samples, 84)), 3)],
        "ci_90": [round(float(np.percentile(sigma2_samples, 5)), 3),
                   round(float(np.percentile(sigma2_samples, 95)), 3)],
    }

    # Convergence diagnostics
    try:
        autocorr = sampler.get_autocorr_time(quiet=True)
        converged = all(nsteps > 50 * tau for tau in autocorr)
    except Exception:
        autocorr = None
        converged = nsteps >= 1000  # rough heuristic

    # Acceptance fraction
    acc_frac = float(np.mean(sampler.acceptance_fraction))

    return {
        "available": True,
        "parameters": parameters,
        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "burnin": burnin,
        "n_samples": len(samples),
        "acceptance_fraction": round(acc_frac, 3),
        "converged": converged,
        "convergence_note": (
            "Chain appears converged." if converged
            else "Chain may not be fully converged. Consider running with more steps."
        ),
        "pore_pressure": round(float(pore_pressure), 2),
        "regime": regime,
        "stakeholder_summary": _bayesian_stakeholder_summary(parameters, converged),
    }


def _bayesian_stakeholder_summary(parameters: dict, converged: bool) -> str:
    """Generate plain-language summary of Bayesian uncertainty for stakeholders."""
    s1 = parameters["sigma1"]
    shmax = parameters["SHmax_azimuth"]
    mu = parameters["mu"]

    parts = [
        f"Maximum stress (sigma1): {s1['median']:.1f} MPa "
        f"(90% CI: {s1['ci_90'][0]:.1f}-{s1['ci_90'][1]:.1f} MPa). ",
        f"Maximum horizontal stress direction: {shmax['median']:.0f} degrees "
        f"(90% CI: {shmax['ci_90'][0]:.0f}-{shmax['ci_90'][1]:.0f} degrees). ",
        f"Friction coefficient: {mu['median']:.2f} "
        f"(90% CI: {mu['ci_90'][0]:.2f}-{mu['ci_90'][1]:.2f}). ",
    ]

    if shmax['ci_90'][1] - shmax['ci_90'][0] > 60:
        parts.append(
            "WARNING: SHmax direction is poorly constrained (>60 degree range). "
            "Drilling azimuth recommendations have high uncertainty."
        )

    if not converged:
        parts.append(
            "NOTE: MCMC chain may not have fully converged. "
            "Results should be treated as preliminary."
        )

    return "".join(parts)


if __name__ == "__main__":
    from data_loader import load_all_fractures, fracture_plane_normal, AZIMUTH_COL, DIP_COL, WELL_COL

    df = load_all_fractures()
    # Use well 3P data for inversion
    df_3p = df[df[WELL_COL] == "3P"]
    normals = fracture_plane_normal(df_3p[AZIMUTH_COL].values, df_3p[DIP_COL].values)

    print(f"Running inversion on {len(normals)} fractures from Well 3P...")
    result = invert_stress(normals, regime="strike_slip", depth_m=3100.0)

    print(f"\n--- Inversion Results (Well 3P) ---")
    print(f"  σ1 = {result['sigma1']:.1f} MPa")
    print(f"  σ2 = {result['sigma2']:.1f} MPa")
    print(f"  σ3 = {result['sigma3']:.1f} MPa")
    print(f"  R  = {result['R']:.3f}")
    print(f"  SHmax azimuth = {result['shmax_azimuth_deg']:.1f}°")
    print(f"  μ  = {result['mu']:.3f}")
    print(f"  Regime: {result['regime']}")
    print(f"\n  Stress tensor (MPa):")
    print(f"  {result['stress_tensor']}")
