import abc
import logging
import timeit
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCSR

from mpax.loop_utils import while_loop
from mpax.preprocess import rescale_problem
from mpax.restart import (
    run_restart_scheme,
    select_initial_primal_weight,
    unscaled_saddle_point_output,
)
from mpax.solver_log import (
    display_iteration_stats_heading,
    pdhg_final_log,
    setup_logger,
)
from mpax.termination import cached_quadratic_program_info, check_termination_criteria
from mpax.utils import (
    OptimalityNorm,
    PdhgSolverState,
    QuadraticProgrammingProblem,
    RestartInfo,
    RestartParameters,
    RestartScheme,
    RestartToCurrentMetric,
    SaddlePointOutput,
    TerminationCriteria,
    TerminationStatus,
)

logger = logging.getLogger(__name__)


def estimate_maximum_singular_value(
    matrix: BCSR,
    probability_of_failure: float = 0.01,
    desired_relative_error: float = 0.1,
    seed: int = 1,
) -> tuple:
    """
    Estimate the maximum singular value of a sparse matrix using the power method.

    Parameters
    ----------
    matrix : BCSR
        The sparse matrix in BCSR format.
    probability_of_failure : float, optional
        The acceptable probability of failure.
    desired_relative_error : float, optional
        The desired relative error for the estimation.
    seed : int, optional
        The random seed for reproducibility.

    Returns
    -------
    tuple
        The estimated maximum singular value and the number of power iterations.
    """
    epsilon = 1.0 - (1.0 - desired_relative_error) ** 2
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (matrix.shape[1],))
    matrix_transpose = BCSR.from_bcoo(matrix.to_bcoo().T)
    number_of_power_iterations = 0

    def cond_fun(state):
        # Corresponds to the power_method_failure_probability in CuPDLP.jl
        x, number_of_power_iterations = state
        power_method_failure_probability = jax.lax.cond(
            # We have to use bitwise operators | instead of or in JAX.
            # Ref: https://github.com/jax-ml/jax/issues/3761#issuecomment-658456938
            (number_of_power_iterations < 2) | (epsilon <= 0.0),
            lambda _: 1.0,
            lambda _: (
                jax.lax.min(
                    0.824, 0.354 / jnp.sqrt(epsilon * (number_of_power_iterations - 1))
                )
                * jnp.sqrt(matrix.shape[1])
                * (1.0 - epsilon) ** (number_of_power_iterations - 0.5)
            ),
            operand=None,
        )
        return power_method_failure_probability > probability_of_failure

    def body_fun(state):
        x, number_of_power_iterations = state
        x = x / jnp.linalg.norm(x, 2)
        x = matrix_transpose @ (matrix @ x)
        return x, number_of_power_iterations + 1

    # while_loop() compiles cond_fun and body_fun, so while it can be combined with jit(), it’s usually unnecessary.
    x, number_of_power_iterations = while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=(x, 0),
        maxiter=1000,
        unroll=True,
        jit=True,
    )
    return (
        jnp.sqrt(
            jnp.dot(x, matrix_transpose @ (matrix @ x)) / jnp.linalg.norm(x, 2) ** 2
        ),
        number_of_power_iterations,
    )


def compute_next_solution(
    problem: QuadraticProgrammingProblem,
    solver_state: PdhgSolverState,
    step_size: float,
    extrapolation_coefficient: float,
):
    """Compute the next primal and dual solutions.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    step_size : float
        The step size used in the PDHG algorithm.
    extrapolation_coefficient : float
        The extrapolation coefficient.

    Returns
    -------
    tuple
        The delta primal, delta primal product, and delta dual.
    """
    # Compute the next primal solution.
    next_primal_solution = solver_state.current_primal_solution - (
        step_size / solver_state.primal_weight
    ) * (problem.objective_vector - solver_state.current_dual_product)
    # Projection.
    next_primal_solution = jnp.minimum(
        problem.variable_upper_bound,
        jnp.maximum(problem.variable_lower_bound, next_primal_solution),
    )
    delta_primal = next_primal_solution - solver_state.current_primal_solution
    delta_primal_product = problem.constraint_matrix @ delta_primal

    # Compute the next dual solution.
    next_dual_solution = solver_state.current_dual_solution + (
        solver_state.primal_weight * step_size
    ) * (
        problem.right_hand_side
        - (1 + extrapolation_coefficient) * delta_primal_product
        - extrapolation_coefficient * solver_state.current_primal_product
    )
    next_dual_solution = jnp.where(
        problem.inequalities_mask,
        jnp.maximum(next_dual_solution, 0.0),
        next_dual_solution,
    )
    delta_dual_solution = next_dual_solution - solver_state.current_dual_solution
    return delta_primal, delta_primal_product, delta_dual_solution


def line_search(
    problem, solver_state, reduction_exponent, growth_exponent, step_size_limit_coef
):
    """Perform a line search to find a good step size.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    reduction_exponent : float
        The reduction exponent for adaptive step size.
    growth_exponent : float
        The growth exponent for adaptive step size.
    step_size_limit_coef: float
        The step size limit coefficient for adaptive step size.

    Returns
    -------
    tuple
        The delta_primal, delta_dual, delta_primal_product, step_size, and line_search_iter.
    """

    def cond_fun(line_search_state):
        step_size_limit = line_search_state[4]
        old_step_size = line_search_state[6]
        return jax.lax.cond(
            old_step_size <= step_size_limit,
            lambda _: False,
            lambda _: True,
            operand=None,
        )

    def body_fun(line_search_state):
        line_search_iter = line_search_state[0]
        step_size_limit = line_search_state[4]
        step_size = line_search_state[5]
        line_search_iter += 1
        delta_primal, delta_primal_product, delta_dual = compute_next_solution(
            problem, solver_state, step_size, 1.0
        )
        interaction = jnp.abs(jnp.dot(delta_primal_product, delta_dual))
        movement = 0.5 * solver_state.primal_weight * jnp.sum(
            jnp.square(delta_primal)
        ) + (0.5 / solver_state.primal_weight) * jnp.sum(jnp.square(delta_dual))

        step_size_limit = jax.lax.cond(
            interaction > 0,
            lambda _: movement / interaction * step_size_limit_coef,
            lambda _: jnp.inf,
            operand=None,
        )
        old_step_size = step_size
        first_term = (
            1
            - 1
            / (solver_state.num_steps_tried + line_search_iter + 1)
            ** reduction_exponent
        ) * step_size_limit
        second_term = (
            1
            + 1
            / (solver_state.num_steps_tried + line_search_iter + 1) ** growth_exponent
        ) * step_size
        step_size = jnp.minimum(first_term, second_term)

        return (
            line_search_iter,
            delta_primal,
            delta_dual,
            delta_primal_product,
            step_size_limit,
            step_size,
            old_step_size,
        )

    (
        line_search_iter,
        delta_primal,
        delta_dual,
        delta_primal_product,
        step_size_limit,
        step_size,
        old_step_size,
    ) = while_loop(
        cond_fun,
        body_fun,
        init_val=(
            0,
            jnp.zeros_like(solver_state.current_primal_solution),
            jnp.zeros_like(solver_state.current_dual_solution),
            jnp.zeros_like(solver_state.current_dual_solution),
            -jnp.inf,
            solver_state.step_size,
            solver_state.step_size,
        ),
        maxiter=10,
        unroll=False,
        jit=True,
    )

    return delta_primal, delta_dual, delta_primal_product, step_size, line_search_iter


@dataclass(eq=False)
class raPDHG(abc.ABC):
    """
    The raPDHG solver class.
    """

    verbose: bool = False
    debug: bool = False
    display_frequency: int = 10
    jit: bool = True
    unroll: bool = False
    termination_evaluation_frequency: int = 64
    optimality_norm: int = OptimalityNorm.L2
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    eps_primal_infeasible: float = 1e-8
    eps_dual_infeasible: float = 1e-8
    # time_sec_limit: float = float("inf")
    iteration_limit: int = jnp.iinfo(jnp.int32).max
    l_inf_ruiz_iterations: int = 10
    l2_norm_rescaling: bool = False
    pock_chambolle_alpha: float = 1.0
    primal_importance: float = 1.0
    scale_invariant_initial_primal_weight: bool = True
    restart_scheme: int = RestartScheme.ADAPTIVE_KKT
    restart_to_current_metric: int = RestartToCurrentMetric.KKT_GREEDY
    restart_frequency_if_fixed: int = 1000
    artificial_restart_threshold: float = 0.36
    sufficient_reduction_for_restart: float = 0.2
    necessary_reduction_for_restart: float = 0.8
    primal_weight_update_smoothing: float = 0.5
    adaptive_step_size: bool = True
    adaptive_step_size_reduction_exponent: float = 0.3
    adaptive_step_size_growth_exponent: float = 0.6
    adaptive_step_size_limit_coef: float = 1.0

    def check_config(self):
        self._termination_criteria = TerminationCriteria(
            optimality_norm=self.optimality_norm,
            eps_abs=self.eps_abs,
            eps_rel=self.eps_rel,
            eps_primal_infeasible=self.eps_primal_infeasible,
            eps_dual_infeasible=self.eps_dual_infeasible,
            # time_sec_limit=self.time_sec_limit,
            iteration_limit=self.iteration_limit,
        )
        self._restart_params = RestartParameters(
            restart_scheme=self.restart_scheme,
            restart_to_current_metric=self.restart_to_current_metric,
            restart_frequency_if_fixed=self.restart_frequency_if_fixed,
            artificial_restart_threshold=self.artificial_restart_threshold,
            sufficient_reduction_for_restart=self.sufficient_reduction_for_restart,
            necessary_reduction_for_restart=self.necessary_reduction_for_restart,
            primal_weight_update_smoothing=self.primal_weight_update_smoothing,
        )

    def initialize_solver_status(
        self, scaled_qp: QuadraticProgrammingProblem
    ) -> PdhgSolverState:
        """Initialize the solver status for PDHG.

        Parameters
        ----------
        scaled_qp : QuadraticProgrammingProblem
            The scaled quadratic programming problem.
        params : PdhgParameters
            The parameters for the PDHG algorithm.

        Returns
        -------
        PdhgSolverState
            The initial solver status.
        """
        primal_size = len(scaled_qp.variable_lower_bound)
        dual_size = len(scaled_qp.right_hand_side)

        # Step size computation
        if self.adaptive_step_size:
            step_size = 1.0 / jnp.max(jnp.abs(scaled_qp.constraint_matrix.data))
        else:
            desired_relative_error = 0.2
            maximum_singular_value, number_of_power_iterations = (
                estimate_maximum_singular_value(
                    scaled_qp.constraint_matrix,
                    probability_of_failure=0.001,
                    desired_relative_error=desired_relative_error,
                )
            )
            step_size = (1 - desired_relative_error) / maximum_singular_value

        # Primal weight initialization
        if self.scale_invariant_initial_primal_weight:
            primal_weight = select_initial_primal_weight(
                scaled_qp, 1.0, 1.0, self.primal_importance
            )
        else:
            primal_weight = self.primal_importance

        solver_state = PdhgSolverState(
            current_primal_solution=jnp.zeros(primal_size),
            current_dual_solution=jnp.zeros(dual_size),
            current_primal_product=jnp.zeros(dual_size),
            current_dual_product=jnp.zeros(primal_size),
            solutions_count=0,
            weights_sum=0.0,
            step_size=step_size,
            primal_weight=primal_weight,
            numerical_error=False,
            # total_number_iterations=0,
            avg_primal_solution=jnp.zeros(primal_size),
            avg_dual_solution=jnp.zeros(dual_size),
            avg_primal_product=jnp.zeros(dual_size),
            avg_dual_product=jnp.zeros(primal_size),
            initial_primal_solution=jnp.zeros(primal_size),
            initial_dual_solution=jnp.zeros(dual_size),
            initial_primal_product=jnp.zeros(dual_size),
            initial_dual_product=jnp.zeros(primal_size),
            num_steps_tried=0,
            num_iterations=0,
            termination_status=TerminationStatus.UNSPECIFIED,
            delta_primal=jnp.zeros(primal_size),
            delta_dual=jnp.zeros(dual_size),
            delta_primal_product=jnp.zeros(dual_size),
        )

        last_restart_info = RestartInfo(
            primal_solution=jnp.zeros(primal_size),
            dual_solution=jnp.zeros(dual_size),
            primal_diff=jnp.zeros(primal_size),
            dual_diff=jnp.zeros(dual_size),
            primal_diff_product=jnp.zeros(dual_size),
            primal_product=jnp.zeros(dual_size),
            dual_product=jnp.zeros(primal_size),
        )
        return solver_state, last_restart_info

    def should_check_termination(self, solver_state: PdhgSolverState) -> bool:
        return (
            (solver_state.num_iterations % self.termination_evaluation_frequency == 0)
            | (solver_state.num_iterations == self.iteration_limit)
            | (solver_state.num_iterations <= 10)
            | (solver_state.numerical_error)
        )

    def take_step(
        self, solver_state: PdhgSolverState, problem: QuadraticProgrammingProblem
    ) -> PdhgSolverState:
        """
        Take a PDHG step with adaptive step size.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        problem : QuadraticProgrammingProblem
            The problem being solved.
        """

        if self.adaptive_step_size:
            (
                delta_primal,
                delta_dual,
                delta_primal_product,
                step_size,
                line_search_iter,
            ) = line_search(
                problem,
                solver_state,
                self.adaptive_step_size_reduction_exponent,
                self.adaptive_step_size_growth_exponent,
                self.adaptive_step_size_limit_coef,
            )
        else:
            delta_primal, delta_primal_product, delta_dual = compute_next_solution(
                problem, solver_state, solver_state.step_size, 1.0
            )
            step_size = solver_state.step_size
            line_search_iter = 1

        next_primal_solution = solver_state.current_primal_solution + delta_primal
        next_primal_product = solver_state.current_primal_product + delta_primal_product
        next_dual_solution = solver_state.current_dual_solution + delta_dual
        next_dual_product = problem.constraint_matrix_t @ next_dual_solution

        ratio = step_size / (solver_state.weights_sum + step_size)
        next_avg_primal_solution = solver_state.avg_primal_solution + ratio * (
            next_primal_solution - solver_state.avg_primal_solution
        )
        next_avg_dual_solution = solver_state.avg_dual_solution + ratio * (
            next_dual_solution - solver_state.avg_dual_solution
        )
        next_avg_primal_product = solver_state.avg_primal_product + ratio * (
            next_primal_product - solver_state.avg_primal_product
        )
        next_avg_dual_product = solver_state.avg_dual_product + ratio * (
            next_dual_product - solver_state.avg_dual_product
        )
        new_solutions_count = solver_state.solutions_count + 1
        new_weights_sum = solver_state.weights_sum + step_size

        return PdhgSolverState(
            current_primal_solution=next_primal_solution,
            current_dual_solution=next_dual_solution,
            current_primal_product=next_primal_product,
            current_dual_product=next_dual_product,
            avg_primal_solution=next_avg_primal_solution,
            avg_dual_solution=next_avg_dual_solution,
            avg_primal_product=next_avg_primal_product,
            avg_dual_product=next_avg_dual_product,
            initial_primal_solution=solver_state.initial_primal_solution,
            initial_dual_solution=solver_state.initial_dual_solution,
            initial_primal_product=solver_state.initial_primal_product,
            initial_dual_product=solver_state.initial_dual_product,
            delta_primal=delta_primal,
            delta_dual=delta_dual,
            delta_primal_product=delta_primal_product,
            solutions_count=new_solutions_count,
            weights_sum=new_weights_sum,
            step_size=step_size,
            primal_weight=solver_state.primal_weight,
            numerical_error=False,
            num_steps_tried=solver_state.num_steps_tried + line_search_iter,
            num_iterations=solver_state.num_iterations + 1,
            termination_status=TerminationStatus.UNSPECIFIED,
        )

    def iteration_update(
        self,
        solver_state,
        last_restart_info,
        should_terminate,
        scaled_problem,
        qp_cache,
    ):
        """The inner loop of PDLP algorithm.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        last_restart_info : RestartInfo
            The information of the last restart.
        should_terminate : bool
            Whether the algorithm should terminate.
        scaled_problem : QuadraticProgrammingProblem
            The scaled quadratic programming problem.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.

        Returns
        -------
        tuple
            The updated solver state, the updated last restart info, whether to terminate, the scaled problem, and the cached quadratic programming information.
        """
        # Check for termination
        should_terminate, termination_status = jax.lax.cond(
            self.should_check_termination(solver_state),
            lambda: check_termination_criteria(
                scaled_problem,
                solver_state,
                self._termination_criteria,
                qp_cache,
                solver_state.numerical_error,
                1.0,
                self.termination_evaluation_frequency * self.display_frequency,
            ),
            lambda: (False, TerminationStatus.UNSPECIFIED),
        )

        restarted_solver_state, new_last_restart_info = jax.lax.cond(
            self.should_check_termination(solver_state),
            lambda: run_restart_scheme(
                scaled_problem.scaled_qp,
                solver_state,
                last_restart_info,
                self._restart_params,
            ),
            lambda: (solver_state, last_restart_info),
        )

        new_solver_state = self.take_step(
            restarted_solver_state, scaled_problem.scaled_qp
        )
        new_solver_state.termination_status = termination_status
        return (
            new_solver_state,
            new_last_restart_info,
            should_terminate,
            scaled_problem,
            qp_cache,
        )

    def optimize(
        self, original_problem: QuadraticProgrammingProblem
    ) -> SaddlePointOutput:
        """
        Main algorithm: given parameters and LP problem, return solutions.

        Parameters
        ----------
        original_problem : QuadraticProgrammingProblem
            The quadratic programming problem to be solved.

        Returns
        -------
        SaddlePointOutput
            The solution to the optimization problem.
        """
        setup_logger(self.verbose, self.debug)
        # validate(original_problem)
        # config_check(params)
        self.check_config()
        qp_cache = cached_quadratic_program_info(original_problem)

        precondition_start_time = timeit.default_timer()
        scaled_problem = rescale_problem(
            self.l_inf_ruiz_iterations,
            self.l2_norm_rescaling,
            self.pock_chambolle_alpha,
            original_problem,
        )
        precondition_time = timeit.default_timer() - precondition_start_time
        logger.info("Preconditioning Time (seconds): %.2e", precondition_time)

        solver_state, last_restart_info = self.initialize_solver_status(
            scaled_problem.scaled_qp
        )

        # Iteration loop
        display_iteration_stats_heading()

        iteration_start_time = timeit.default_timer()
        (solver_state, last_restart_info, should_terminate, _, _) = while_loop(
            cond_fun=lambda state: state[2] == False,
            body_fun=lambda state: self.iteration_update(*state),
            init_val=(solver_state, last_restart_info, False, scaled_problem, qp_cache),
            maxiter=self.iteration_limit,
            unroll=self.unroll,
            jit=self.jit,
        )
        iteration_time = timeit.default_timer() - iteration_start_time
        timing = {
            "Preconditioning": precondition_time,
            "Iteration loop": iteration_time,
        }

        # Log the stats of the final iteration.
        pdhg_final_log(
            scaled_problem.scaled_qp,
            solver_state.avg_primal_solution,
            solver_state.avg_dual_solution,
            solver_state.num_iterations,
            solver_state.termination_status,
            timing,
        )
        return unscaled_saddle_point_output(
            scaled_problem,
            solver_state.avg_primal_solution,
            solver_state.avg_dual_solution,
            solver_state.termination_status,
            solver_state.num_iterations - 1,
        )
