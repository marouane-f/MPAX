import timeit

import gurobipy as gp
import jax
import jax.numpy as jnp
import pytest
from jax import config, jit, vmap
from jax.experimental.sparse import BCOO
from jax.sharding import PartitionSpec as P
from pulp import LpProblem
from pulp2mat import convert_all

from mpax.mp_io import create_lp, create_lp_from_gurobi
from mpax.rapdhg import raPDHG

config.update("jax_enable_x64", True)


def test_rapdhg():
    """Test the raPDHG solver on a sample LP problem."""
    gurobi_model = gp.read("tests/lp_instances/gen-ip054.mps")
    lp = create_lp_from_gurobi(gurobi_model)
    solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6)
    result = solver.optimize(lp)
    objective_value = (
        jnp.dot(lp.objective_vector, result.primal_solution) + lp.objective_constant
    )

    # Expected optimal objective value
    expected_obj = 6.765209043e03
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj


def test_rapdhg_with_jit():
    """Test the raPDHG solver on a sample LP problem."""
    gurobi_model = gp.read("tests/lp_instances/gen-ip054.mps")
    lp = create_lp_from_gurobi(gurobi_model)
    solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6)
    jit_optimize = jit(solver.optimize)
    result = jit_optimize(lp)
    objective_value = (
        jnp.dot(lp.objective_vector, result.primal_solution) + lp.objective_constant
    )

    # Expected optimal objective value
    expected_obj = 6.765209043e03
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj


def test_rapdhg_with_vmap():
    """Test the raPDHG solver on a batch of LP problems."""
    var, prob = LpProblem.fromMPS("tests/lp_instances/gen-ip054.mps")
    c, integrality, constraints, bounds = convert_all(prob)
    c = jnp.array(c)
    A = BCOO.fromdense(constraints.A)
    l = jnp.array(constraints.lb)
    u = jnp.array(constraints.ub)
    var_lb = jnp.array(bounds.lb)
    var_ub = jnp.array(bounds.ub)

    solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6)
    jit_optimize = jit(solver.optimize)

    def solve_single(c):
        boxed_lp = create_lp(c, A, l, u, var_lb, var_ub)
        # result = solver.optimize(boxed_lp)
        result = jit_optimize(boxed_lp)
        return (
            jnp.dot(boxed_lp.objective_vector, result.primal_solution)
            + boxed_lp.objective_constant
        )

    start_time = timeit.default_timer()
    objective_value = solve_single(c)
    print("1st single run time = ", timeit.default_timer() - start_time)

    # # Expected optimal objective value
    expected_obj = 6.765209043e03
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj

    start_time = timeit.default_timer()
    objective_value = solve_single(c)
    print("2nd single run time = ", timeit.default_timer() - start_time)
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj

    batch_size = 100
    batch_c = jnp.tile(c, (batch_size, 1))
    batch_optimize = vmap(solve_single)

    start_time = timeit.default_timer()
    batch_result = batch_optimize(batch_c)
    print("1st batch run time = ", timeit.default_timer() - start_time)

    for obj in batch_result:
        assert pytest.approx(obj, rel=1e-2) == expected_obj

    start_time = timeit.default_timer()
    batch_result = batch_optimize(batch_c)
    print("2nd batch run time = ", timeit.default_timer() - start_time)

    for obj in batch_result:
        assert pytest.approx(obj, rel=1e-2) == expected_obj


@pytest.mark.skipif(
    jax.device_count() <= 1,
    reason="Only one device available, skipping test_rapdhg_with_sharding",
)
def test_rapdhg_with_sharding():
    """Test the raPDHG solver on a batch of LP problems."""
    mesh = jax.make_mesh((2,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, P('x'))

    gurobi_model = gp.read("tests/lp_instances/flugpl.mps")
    lp_sharded = create_lp_from_gurobi(gurobi_model, sharding=sharding)
    jax.debug.visualize_array_sharding(lp_sharded.constraint_matrix)

    solver = raPDHG(eps_abs=1e-8, eps_rel=1e-8)
    jit_optimize = jax.jit(solver.optimize)
    result = jit_optimize(lp_sharded)

    objective_value = (
        jnp.dot(lp_sharded.objective_vector, result.primal_solution)
        + lp_sharded.objective_constant
    )

    # Expected optimal objective value
    expected_obj = 1.167185726e06
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj
