from typing import Union

import jax
import jax.numpy as jnp

from mpax.iteration_stats_utils import evaluate_unscaled_iteration_stats
from mpax.utils import (
    CachedQuadraticProgramInfo,
    ConvergenceInformation,
    OptimalityNorm,
    QuadraticProgrammingProblem,
    TerminationCriteria,
    TerminationStatus,
    PdhgSolverState,
    ScaledQpProblem,
)


def validate_termination_criteria(criteria: TerminationCriteria) -> None:
    """
    Validates the termination criteria to ensure all parameters are within acceptable ranges.

    Parameters
    ----------
    criteria : TerminationCriteria
        The criteria to validate.

    Raises
    ------
    ValueError
        If any of the criteria parameters are not valid.
    """
    if criteria.eps_primal_infeasible < 0:
        raise ValueError("eps_primal_infeasible must be nonnegative.")
    if criteria.eps_dual_infeasible < 0:
        raise ValueError("eps_dual_infeasible must be nonnegative.")
    # if criteria.time_sec_limit <= 0:
    #     raise ValueError("time_sec_limit must be positive.")
    if criteria.iteration_limit <= 0:
        raise ValueError("iteration_limit must be positive.")


def cached_quadratic_program_info(
    qp: QuadraticProgrammingProblem,
) -> CachedQuadraticProgramInfo:
    """
    Computes information about the quadratic program used in termination criteria.

    Parameters
    ----------
    qp : QuadraticProgrammingProblem
        The quadratic programming problem instance.

    Returns
    -------
    CachedQuadraticProgramInfo
        Cached information about the quadratic program.
    """
    return CachedQuadraticProgramInfo(
        jnp.linalg.norm(qp.objective_vector, jnp.inf),
        jnp.linalg.norm(qp.right_hand_side, jnp.inf),
        jnp.linalg.norm(qp.objective_vector, 2),
        jnp.linalg.norm(qp.right_hand_side, 2),
    )


def optimality_criteria_met(
    optimality_norm: OptimalityNorm,
    rel_tol: float,
    convergence_information: ConvergenceInformation,
) -> bool:
    """
    Checks if the algorithm should terminate declaring the optimal solution is found.

    Parameters
    ----------
    optimality_norm : OptimalityNorm
        The norm to measure the optimality criteria.
    rel_tol : float
        Relative tolerance.
    convergence_information : ConvergenceInformation
        Convergence information of the current iteration.

    Returns
    -------
    bool
        True if optimality criteria are met, False otherwise.
    """
    ci = convergence_information
    gap = ci.relative_optimality_gap
    dual_err = jax.lax.cond(
        optimality_norm == OptimalityNorm.L_INF,
        lambda: ci.relative_l_inf_dual_residual,
        lambda: ci.relative_l2_dual_residual,
    )
    primal_err = jax.lax.cond(
        optimality_norm == OptimalityNorm.L_INF,
        lambda: ci.relative_l_inf_primal_residual,
        lambda: ci.relative_l2_primal_residual,
    )
    return (dual_err < rel_tol) & (primal_err < rel_tol) & (gap < rel_tol)


def primal_infeasibility_criteria_met(
    eps_primal_infeasible: float, infeasibility_information
) -> bool:
    """
    Checks if the algorithm should terminate declaring the primal is infeasible.

    Parameters
    ----------
    eps_primal_infeasible : float
        The tolerance for primal infeasibility.
    infeasibility_information : InfeasibilityInformation
        Information regarding infeasibility.

    Returns
    -------
    bool
        True if primal infeasibility criteria are met, False otherwise.
    """
    ii = infeasibility_information
    return jax.lax.cond(
        ii.dual_ray_objective <= 0.0,
        lambda _: False,
        lambda _: ii.max_dual_ray_infeasibility / ii.dual_ray_objective
        <= eps_primal_infeasible,
        operand=None,
    )


def dual_infeasibility_criteria_met(
    eps_dual_infeasible: float, infeasibility_information
) -> bool:
    """
    Checks if the algorithm should terminate declaring the dual is infeasible.

    Parameters
    ----------
    eps_dual_infeasible : float
        The tolerance for dual infeasibility.
    infeasibility_information : InfeasibilityInformation
        Information regarding infeasibility.

    Returns
    -------
    bool
        True if dual infeasibility criteria are met, False otherwise.
    """
    ii = infeasibility_information
    return jax.lax.cond(
        ii.primal_ray_linear_objective >= 0.0,
        lambda _: False,
        lambda _: ii.max_primal_ray_infeasibility / (-ii.primal_ray_linear_objective)
        <= eps_dual_infeasible,
        operand=None,
    )


def check_termination_criteria(
    scaled_problem,
    solver_state,
    criteria: TerminationCriteria,
    qp_cache: CachedQuadraticProgramInfo,
    numerical_error: bool,
    elapsed_time: float,
    average: bool = True,
    infeasibility_detection: bool = True,
) -> Union[str, bool]:
    """
    Checks if the given iteration_stats satisfy the termination criteria.

    Parameters
    ----------
    criteria : TerminationCriteria
        Termination criteria to check against.
    qp_cache : CachedQuadraticProgramInfo
        Cached information about the quadratic program.
    iteration_stats : IterationStats
        Statistics of the current iteration.

    Returns
    -------
    Union[str, bool]
        Termination reason if criteria are met, False otherwise.
    """
    eps_ratio = criteria.eps_abs / criteria.eps_rel
    current_iteration_stats = evaluate_unscaled_iteration_stats(
        scaled_problem,
        qp_cache,
        solver_state,
        elapsed_time,
        eps_ratio,
        average,
        infeasibility_detection,
    )
    should_terminate = False
    termination_status = TerminationStatus.UNSPECIFIED
    should_terminate, termination_status = jax.lax.cond(
        optimality_criteria_met(
            criteria.optimality_norm,
            criteria.eps_rel,
            current_iteration_stats.convergence_information,
        ),
        lambda _: (True, TerminationStatus.OPTIMAL),
        lambda _: (should_terminate, termination_status),
        operand=None,
    )

    should_terminate, termination_status = jax.lax.cond(
        (should_terminate == False)
        & primal_infeasibility_criteria_met(
            criteria.eps_primal_infeasible,
            current_iteration_stats.infeasibility_information,
        ),
        lambda _: (True, TerminationStatus.PRIMAL_INFEASIBLE),
        lambda _: (should_terminate, termination_status),
        operand=None,
    )

    should_terminate, termination_status = jax.lax.cond(
        (should_terminate == False)
        & dual_infeasibility_criteria_met(
            criteria.eps_dual_infeasible,
            current_iteration_stats.infeasibility_information,
        ),
        lambda _: (True, TerminationStatus.DUAL_INFEASIBLE),
        lambda _: (should_terminate, termination_status),
        operand=None,
    )

    should_terminate, termination_status = jax.lax.cond(
        (should_terminate == False)
        & (current_iteration_stats.iteration_number >= criteria.iteration_limit),
        lambda _: (True, TerminationStatus.ITERATION_LIMIT),
        lambda _: (should_terminate, termination_status),
        operand=None,
    )

    # should_terminate, termination_status = jax.lax.cond(
    #     (should_terminate == False) & (elapsed_time >= criteria.time_sec_limit),
    #     lambda _: (True, TerminationStatus.TIME_LIMIT),
    #     lambda _: (should_terminate, termination_status),
    #     operand=None,
    # )

    should_terminate, termination_status = jax.lax.cond(
        (should_terminate == False) & numerical_error,
        lambda _: (True, TerminationStatus.NUMERICAL_ERROR),
        lambda _: (should_terminate, termination_status),
        operand=None,
    )
    # should_terminate, termination_status = jax.lax.cond(
    #     (should_terminate == True)
    #     & ((termination_status == TerminationStatus.PRIMAL_INFEASIBLE) | (termination_status == TerminationStatus.DUAL_INFEASIBLE)),
    #     lambda _: (False, TerminationStatus.UNSPECIFIED),
    #     lambda _: (should_terminate, termination_status),
    #     operand=None,
    # )
    return should_terminate, termination_status


def check_primal_feasibility(
    scaled_problem: ScaledQpProblem,
    solver_state: PdhgSolverState,
    criteria: TerminationCriteria,
    qp_cache: CachedQuadraticProgramInfo,
    elapsed_time: float,
    average: bool = True,
) -> bool:
    """
    Checks if the given iteration_stats satisfy the termination criteria.

    Parameters
    ----------
    scaled_problem : ScaledQpProblem
        The scaled quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    criteria : TerminationCriteria
        Termination criteria to check against.
    qp_cache : CachedQuadraticProgramInfo
        Cached information about the quadratic program.
    elapsed_time : float
        Elapsed time since the start of the algorithm.
    average : bool, optional
        Whether is raPDHG, by default True.

    Returns
    -------
    bool
        True if primal feasibility criteria are met, False otherwise.
    """
    eps_ratio = criteria.eps_abs / criteria.eps_rel
    current_iteration_stats = evaluate_unscaled_iteration_stats(
        scaled_problem,
        qp_cache,
        solver_state,
        elapsed_time,
        eps_ratio,
        average,
        infeasibility_detection=False,
    )
    ci = current_iteration_stats.convergence_information
    primal_err = jax.lax.cond(
        criteria.optimality_norm == OptimalityNorm.L_INF,
        lambda: ci.l_inf_primal_residual,
        lambda: ci.l2_primal_residual,
    )
    primal_err_baseline = jax.lax.cond(
        criteria.optimality_norm == OptimalityNorm.L_INF,
        lambda: qp_cache.l_inf_norm_primal_right_hand_side,
        lambda: qp_cache.l2_norm_primal_right_hand_side,
    )
    return primal_err < criteria.eps_abs + criteria.eps_rel * primal_err_baseline


def check_dual_feasibility(
    scaled_problem: ScaledQpProblem,
    solver_state: PdhgSolverState,
    criteria: TerminationCriteria,
    qp_cache: CachedQuadraticProgramInfo,
    elapsed_time: float,
    average: bool = True,
) -> Union[str, bool]:
    """
    Checks if the given iteration_stats satisfy the termination criteria.

    Parameters
    ----------
    scaled_problem : ScaledQpProblem
        The scaled quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    criteria : TerminationCriteria
        Termination criteria to check against.
    qp_cache : CachedQuadraticProgramInfo
        Cached information about the quadratic program.
    elapsed_time : float
        Elapsed time since the start of the algorithm.
    average : bool, optional
        Whether is raPDHG, by default True.

    Returns
    -------
    bool
        True if dual feasibility criteria are met, False otherwise.
    """
    eps_ratio = criteria.eps_abs / criteria.eps_rel
    current_iteration_stats = evaluate_unscaled_iteration_stats(
        scaled_problem,
        qp_cache,
        solver_state,
        elapsed_time,
        eps_ratio,
        average,
        infeasibility_detection=False,
    )
    ci = current_iteration_stats.convergence_information
    dual_err = jax.lax.cond(
        criteria.optimality_norm == OptimalityNorm.L_INF,
        lambda: ci.l_inf_dual_residual,
        lambda: ci.l2_dual_residual,
    )
    dual_err_baseline = jax.lax.cond(
        criteria.optimality_norm == OptimalityNorm.L_INF,
        lambda: qp_cache.l_inf_norm_primal_linear_objective,
        lambda: qp_cache.l2_norm_primal_linear_objective,
    )
    return dual_err < criteria.eps_abs + criteria.eps_rel * dual_err_baseline
