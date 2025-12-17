from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .io_points import load_points_txt, downsample_uniform

# Import flessibile:
# - se hai vendor/icp.py (copiato dal repo richardos/icp) aggiungi vendor/ a PYTHONPATH
# - oppure metti icp.py dentro src/laser_tilt/ e importi relativo
try:
    from .vendor.icp import icp  # vendor/icp.py o module su sys.path
except Exception as e:
    raise ImportError(
        "Non trovo il modulo 'icp'. Copia 'icp.py' da richardos/icp in vendor/ "
        "e aggiungi vendor/ al PYTHONPATH, oppure inseriscilo nel package."
    ) from e


@dataclass(frozen=True)
class ICPResult:
    T: np.ndarray          # 3x3 omogenea 2D
    theta_deg: float
    tx: float
    ty: float
    rmse: float
    iterations: int


@dataclass(frozen=True)
class ICPResult:
    T: np.ndarray
    theta_deg: float
    tx: float
    ty: float
    rmse: float
    iterations: int

def _theta_from_T(T: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))

def run_icp(measured: np.ndarray, target: np.ndarray, *, max_iterations=100, distance_threshold=10.0,
            point_pairs_threshold=5, verbose=False) -> ICPResult:
    # NB: ordine giusto per il TUO icp: (reference, points)
    history, aligned = icp(
        reference_points=target,
        points=measured,
        max_iterations=max_iterations,
        distance_threshold=distance_threshold,
        point_pairs_threshold=point_pairs_threshold,
        verbose=verbose,
    )

    T = compose_history(history)
    t = T[:2, 2]
    theta_deg = _theta_from_T(T)

    # RMSE semplice: distanza media tra aligned e nearest neighbor su target
    # (non perfetta ma utile da dashboard)
    rmse = float("nan")
    if aligned is not None and aligned.shape[0] > 0 and target.shape[0] > 0:
        # nearest neighbor brute force (ok per demo; se vuoi, riusa sklearn già presente)
        diff = aligned[:, None, :] - target[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        min_d = np.sqrt(np.min(d2, axis=1))
        rmse = float(np.sqrt(np.mean(min_d**2)))

    return ICPResult(
        T=T,
        theta_deg=theta_deg,
        tx=float(t[0]),
        ty=float(t[1]),
        rmse=rmse,
        iterations=len(history),
    )


def correction_transform(res: ICPResult) -> np.ndarray:
    """Trasformazione da applicare per correggere (inversa)."""
    return np.linalg.inv(res.T)


def compute_from_files(
    target_path: str,
    measured_path: str,
    *,
    downsample_step: int = 1,
    max_iterations: int = 50,
) -> tuple[ICPResult, np.ndarray]:
    target = downsample_uniform(load_points_txt(target_path), downsample_step)
    meas = downsample_uniform(load_points_txt(measured_path), downsample_step)

    if target.shape[0] < 10 or meas.shape[0] < 10:
        raise ValueError("Troppi pochi punti validi: controlla file / filtro (0,0) / downsample.")

    res = run_icp(meas, target, max_iterations=max_iterations)
    Tcorr = correction_transform(res)
    return res, Tcorr

def _to_T33(rt_2x3: np.ndarray) -> np.ndarray:
    """Converte [R|t] 2x3 in omogenea 3x3."""
    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = rt_2x3[:, :2]
    T[:2,  2] = rt_2x3[:, 2]
    return T

def compose_history(history) -> np.ndarray:
    """
    history: lista di matrici 2x3 (una per iterazione)
    Ritorna T totale 3x3.
    """
    T_tot = np.eye(3, dtype=np.float64)
    for rt in history:
        Ti = _to_T33(np.asarray(rt, dtype=np.float64))
        T_tot = Ti @ T_tot
    return T_tot