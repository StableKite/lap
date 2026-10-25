"""Concurrent independent LAPJV/LAPMOD calls, including normal FT import."""
from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
import lap


def test_normal_free_threaded_import():
    import sysconfig
    if sysconfig.get_config_var('Py_GIL_DISABLED'):
        assert not sys._is_gil_enabled()


def _solve(seed):
    rng=np.random.default_rng(seed)
    costs=rng.uniform(.1,2.,size=(19,23))
    costs[np.arange(19),np.arange(19)]=.001
    before=costs.copy()
    result=lap.lapjv(costs,extend_cost=True,cost_limit=.1)
    np.testing.assert_array_equal(result[1],np.arange(19))
    np.testing.assert_array_equal(result[2],np.r_[np.arange(19),[-1]*4])
    np.testing.assert_array_equal(costs,before)
    dense=costs[:19,:19].copy()
    cc=dense.ravel();ii=np.arange(0,19*19+1,19,dtype=np.uint32)
    kk=np.tile(np.arange(19,dtype=np.uint32),19)
    sparse=lap.lapmod(19,cc,ii,kk)
    np.testing.assert_array_equal(sparse[1],np.arange(19))


def test_parallel_independent_calls():
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_solve,range(256)))
