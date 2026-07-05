import numpy as np
from numba import njit, prange
import math

def serial_sieve(N_max):

    if N_max <= 2:
        return []

    sieve = np.ones(N_max, dtype=np.bool_)
    sieve[0:2] = False

    N_sqrt = int(np.floor(np.sqrt(N_max)))

    for i in range(2, N_sqrt + 1):
        if sieve[i] != 0:
            sieve[i * i :: i] = False

    primes = np.flatnonzero(sieve)
    return primes


def segment_sieve(primes_seed, N_start, d):
    """
    Create a sieve for the interval [N_start, N_start + d);
    Uses primes_seed (the primes primes <= sqrt(N_start + d - 1)) to mark composites in that segment;
    Extracts and returns the primes contained in that segment.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    if N_start < 0:
        raise ValueError("N_start must be non-negative")

    N_end = N_start + d  # exclusive upper bound

    # if primes_seed[-1] * primes_seed[-1] < N_end - 1:
    #     raise ValueError("Not enough primes provided")        conservative, but actually turns out difficult to always provide one prime extra

    sieve = np.ones(d, dtype=bool)

    for p in primes_seed:
        if p * p >= N_end:
            break  # this and all larger seed primes can't affect this segment

        start = p * p
        if start < N_start:
            # first multiple of p that is >= N_start, via integer ceil-division
            start = -(-N_start // p) * p

        index = start - N_start
        sieve[index::p] = False

    primes = N_start + np.flatnonzero(sieve)
    return primes.astype(np.uint64)


# seed = [2, 3, 5, 7, 11]
# print(segment_sieve(seed, 19, 200))


def serial_segmented(N_max, d):
    """
    Compute all primes < N_max using a segmented Sieve of Eratosthenes.
    Builds a seed of primes <= sqrt(N_max - 1) with a plain sieve, then
    sieves the rest of the range in fixed-size segments of length d,
    reusing the same seed for every segment.
    """
    
    if N_max <= 2:
        return np.empty(0, dtype=np.uint64)
 
    N_sqrt = math.isqrt(N_max - 1)      # exact integer sqrt, no float precision risk
    seed = serial_sieve(N_sqrt + 1)     # primes < N_sqrt + 1, i.e. primes <= N_sqrt
 
    chunks = [seed.astype(np.uint64)]
 
    start = N_sqrt + 1
    while start < N_max:
        this_d = min(d, N_max - start)  # clip the final segment so we never exceed N_max
        chunks.append(segment_sieve(seed, start, this_d))
        start += this_d
 
    return np.concatenate(chunks)


# p = serial_segmented(109, 20)
# print(p)




# def numpy_vectorized_inner_loop(N_max):
#     """
#     Same as the boolean serial implementation,
#     but the inner loop is replaced with a strided assignment
#     """
#     if N_max <= 2:
#         return []

#     sieve = np.ones(N_max, dtype=np.bool_)
#     sieve[0:2] = False

#     N_sqrt = int(np.floor(np.sqrt(N_max)))

#     for i in range(2, N_sqrt + 1):
#         if sieve[i] != 0:
#             sieve[i * i :: i] = False

#     primes = np.flatnonzero(sieve)
#     return primes


# @njit(parallel=True)
# def mark_multiples_parallel(sieve, p, N):
#     """
#     Mark multiples of p as False in sieve.
#     prange iterates over k (0..nsteps-1) with constant step 1.
#     j = start + k * p is computed inside the loop.
#     """
#     start = p * p
#     if start >= N:
#         return
#     # number of multiples to mark
#     nsteps = (N - start + p - 1) // p
#     for k in prange(nsteps):  # step is 1 (constant) -> OK
#         j = start + k * p
#         sieve[j] = False


# def numba_parallel_inner_loop_bool_sieve(N_max):
#     """
#     Sieve of Eratosthenes using bitarray (1 bit per entry).
#     Parallel over the inner loop, i.e. the work per prime with numba
#     Returns a list of primes < N_max.
#     """

#     if N_max <= 2:
#         return np.empty(0, dtype=np.uint64)

#     sieve = np.ones(N_max, dtype=np.bool_)  # Numba-friendly
#     sieve[0:2] = False
#     N_sqrt = int(math.isqrt(N_max))

#     # sequential loop over primes up to sqrt(N); inner marking is parallel
#     for p in range(2, N_sqrt + 1):
#         if sieve[p]:
#             mark_multiples_parallel(sieve, p, N_max)

#     primes = np.flatnonzero(sieve)

#     return primes


# # primes = numba_parallel_inner_loop_bool_sieve(1_000_000_000)
# # print(primes)
