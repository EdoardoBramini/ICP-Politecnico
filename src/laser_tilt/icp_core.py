from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .io_points import load_points_txt, downsample_uniform

# Import flessibile:
# - se hai vendor/icp.py (copiato dal repo richardos/icp) aggiungi vendor/ a PYTHONPATH
# - oppure metti icp.py dentro src/laser_tilt/ e importi relativo
try:
    from icp import icp  # vendor/icp.py o module su sys.path
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


def _theta_from_T(T: np.ndarray) -> float:
    R = T[:2, :2]
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    return float(np.degrees(theta))


def run_icp(measured: np.ndarray, target: np.ndarray, *, max_iterations=50, tolerance=1e-8) -> ICPResult:
    """
    Allinea measured -> target con richardos/icp.
    Ritorna trasformazione T (3x3), angolo e traslazione.
    """
    # icp() lavora su Nx2
    T, distances, iters = icp(
        measured,
        target,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    t = T[:2, 2]
    rmse = float(np.sqrt(np.mean(np.asarray(distances) ** 2)))
    return ICPResult(
        T=np.asarray(T, dtype=np.float64),
        theta_deg=_theta_from_T(T),
        tx=float(t[0]),
        ty=float(t[1]),
        rmse=rmse,
        iterations=int(iters),
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
    tolerance: float = 1e-8,
) -> tuple[ICPResult, np.ndarray]:
    target = downsample_uniform(load_points_txt(target_path), downsample_step)
    meas = downsample_uniform(load_points_txt(measured_path), downsample_step)

    if target.shape[0] < 10 or meas.shape[0] < 10:
        raise ValueError("Troppi pochi punti validi: controlla file / filtro (0,0) / downsample.")

    res = run_icp(meas, target, max_iterations=max_iterations, tolerance=tolerance)
    Tcorr = correction_transform(res)
    return res, Tcorr
