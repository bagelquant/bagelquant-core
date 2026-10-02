"""Application-neutral capped-simplex portfolio optimization."""
from __future__ import annotations
import numpy as np

def solve_regularized_weights(
    scores: np.ndarray,
    reference: np.ndarray,
    *,
    concentration_penalty: float,
    turnover_penalty: float,
    max_weight: float,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    """Solve the separable capped-simplex objective through its scalar dual."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    anchors = np.asarray(reference, dtype=float).reshape(-1)
    if values.shape != anchors.shape or values.size == 0:
        raise ValueError("scores and reference must be non-empty matching vectors")
    if not np.isfinite(values).all() or not np.isfinite(anchors).all():
        raise ValueError("scores and reference must be finite")
    if not np.isfinite([concentration_penalty, turnover_penalty, max_weight, tolerance]).all():
        raise ValueError("optimizer parameters must be finite")
    if concentration_penalty <= 0 or turnover_penalty < 0 or not 0 < max_weight <= 1 or tolerance <= 0:
        raise ValueError("invalid optimizer parameters")
    if values.size * max_weight < 1 - tolerance:
        raise ValueError("insufficient capped-simplex capacity")
    shrink = turnover_penalty / (2.0 * concentration_penalty)

    def weights_for_dual(dual: float) -> np.ndarray:
        centered = (values - dual) / (2.0 * concentration_penalty) - anchors
        proximal = anchors + np.sign(centered) * np.maximum(
            np.abs(centered) - shrink,
            0.0,
        )
        return np.clip(proximal, 0.0, max_weight)

    lower = float(
        np.min(values - 2.0 * concentration_penalty * (max_weight + shrink))
    )
    upper = float(np.max(values + 2.0 * concentration_penalty * shrink))
    solution = weights_for_dual((lower + upper) / 2.0)
    iterations = 0
    for iteration in range(1, 101):
        iterations = iteration
        dual = (lower + upper) / 2.0
        solution = weights_for_dual(dual)
        total = float(solution.sum())
        # A merely feasible 1e-7 residual can create tradable one-lot positions
        # when later projection spreads it across assets at the zero bound.
        if abs(total - 1.0) <= min(tolerance, 1e-13):
            break
        if total > 1.0:
            lower = dual
        else:
            upper = dual
    return solution, iterations


def project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    """Remove solver noise without opening assets at the exact zero bound."""

    active = values > 0
    result = np.zeros_like(values, dtype=float)
    values = values[active]
    if not values.size:
        raise ValueError("optimizer returned an empty active set")
    if abs(values.size * cap - 1.0) <= 1e-12:
        result[active] = 1.0 / values.size
        return result
    lower = float(values.min() - cap)
    upper = float(values.max())
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        projected = np.clip(values - midpoint, 0.0, cap)
        if projected.sum() > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(values - (lower + upper) / 2.0, 0.0, cap)
    projected /= projected.sum()
    result[active] = projected
    return result


def solve_exposure_weights(scores, reference, forced_exit, valid, bounds, evaluation_date,
        *, max_weight, max_turnover, concentration_penalty, turnover_penalty,
        constraint_tolerance):
    """Solve supplied absolute exposure bounds without application dependencies."""
    try:
        import cvxpy as cp
    except ImportError as error:
        raise ValueError(
            "exposure optimization requires bagelquant-core[optimizer] "
            "(CVXPY/CLARABEL)"
        ) from error
    weights = cp.Variable(len(scores), nonneg=True)
    turnover = cp.norm1(weights - reference) + forced_exit
    constraints = [cp.sum(weights) == 1.0, weights <= max_weight]
    if max_turnover is not None:
        constraints.append(turnover <= max_turnover)
    for key, bound in bounds.items():
        exposure = valid[key].to_numpy() @ weights
        if bound.lower is not None:
            constraints.append(exposure >= bound.lower)
        if bound.upper is not None:
            constraints.append(exposure <= bound.upper)
    problem = cp.Problem(
        cp.Maximize(
            scores @ weights
            - concentration_penalty * cp.sum_squares(weights)
            - turnover_penalty * turnover
        ),
        constraints,
    )
    try:
        problem.solve(
            solver="CLARABEL",
            tol_gap_abs=min(1e-9, constraint_tolerance / 10),
            tol_gap_rel=min(1e-9, constraint_tolerance / 10),
            tol_feas=min(1e-9, constraint_tolerance / 10),
            max_iter=200,
        )
    except cp.error.SolverError as error:
        raise ValueError(
            f"exposure optimizer solver failed at {evaluation_date}: {error}"
        ) from error
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise ValueError(
            f"exposure optimizer failed at {evaluation_date}: {problem.status}"
        )
    solution = np.asarray(weights.value, dtype=float)
    if not np.isfinite(solution).all():
        raise ValueError(
            f"exposure optimizer returned nonfinite weights at {evaluation_date}"
        )
    # CVXPY's nonnegative variable may contain negative numerical noise.
    # Never project/renormalize: that could violate an exposure constraint.
    return np.maximum(solution, 0.0), problem.status, problem.solver_stats.num_iters
