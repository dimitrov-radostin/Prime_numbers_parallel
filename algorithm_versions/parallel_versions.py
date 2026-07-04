import numpy as np
from bitarray import bitarray
from numba import njit, prange
import math

@njit(parallel=True)
def mark_multiples_parallel(sieve, p, N):
    """
    Mark multiples of p as False in sieve.
    prange iterates over k (0..nsteps-1) with constant step 1.
    j = start + k * p is computed inside the loop.
    """
    start = p * p
    if start >= N:
        return
    # number of multiples to mark
    nsteps = (N - start + p - 1) // p
    for k in prange(nsteps):            # step is 1 (constant) -> OK
        j = start + k * p
        sieve[j] = False
        
        
def numba_parallel_inner_loop_bool_sieve(N_max): 
    """
    Sieve of Eratosthenes using bitarray (1 bit per entry).
    Parallel over the inner loop, i.e. the work per prime with numba
    Returns a list of primes < N_max.
    """
    
    if N_max <= 2:
        return np.empty(0, dtype=np.uint64)

    sieve = np.ones(N_max, dtype=np.bool_)   # Numba-friendly
    sieve[0:2] = False
    N_sqrt = int(math.isqrt(N_max))

    # sequential loop over primes up to sqrt(N); inner marking is parallel
    for p in range(2, N_sqrt + 1):
        if sieve[p]:
            mark_multiples_parallel(sieve, p, N_max)

    return
    # primes = np.flatnonzero(sieve)         

    # return primes


#primes = numba_parallel_inner_loop_bool_sieve(1_000_000_000)
# print(primes)