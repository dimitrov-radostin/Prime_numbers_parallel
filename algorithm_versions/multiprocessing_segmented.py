import math
import numpy as np
from multiprocessing import Pool


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


def _init_worker(seed):
    """Runs once per worker process: stash the seed in that process's globals
    so it isn't re-pickled/re-sent on every single task."""
    global _seed
    _seed = seed
 
 
def _process_segment(args):
    N_start, d = args
    return segment_sieve(_seed, N_start, d)
 

 
def parallel_segmented(N_max, d, n_workers=8):
    """
    Multiprocessing-parallel segmented Sieve of Eratosthenes.
    Segments are independent given the shared read-only seed, so they are
    dispatched across a process pool. Order is preserved (pool.map), so
    results can be concatenated directly without re-sorting.
    """
    if N_max <= 2:
        return np.empty(0, dtype=np.uint64)
 
    N_sqrt = math.isqrt(N_max - 1)
    seed = serial_sieve(N_sqrt + 1)
 
    # build the list of (N_start, this_d) segment descriptors up front
    segments = []
    start = N_sqrt + 1
    while start < N_max:
        this_d = min(d, N_max - start)
        segments.append((start, this_d))
        start += this_d
 
    with Pool(processes=n_workers, initializer=_init_worker, initargs=(seed,)) as pool:
        results = pool.map(_process_segment, segments)
 
    chunks = [seed.astype(np.uint64)] + results
    return np.concatenate(chunks)


def parallel_segmented_with_pool(N_max, d, pool):
    """
    Same as parallel_segmented, but reuses an externally-supplied, already-running
    Pool instead of creating a fresh one -- use this to measure "warm" per-call
    cost (dispatch + compute + gather only), separate from one-time Pool startup cost.
 
    IMPORTANT: the pool's workers had a seed baked in ONCE, at Pool-creation time
    (via _init_worker), for whatever N_max was used then. This function does NOT
    re-send the seed on each call. Only reuse a given pool across calls that share
    the SAME N_max (varying d is fine -- the seed only depends on N_max).
    Reusing a pool across DIFFERENT N_max values will silently produce wrong
    results (stale seed), with no error raised.
    """
    if N_max <= 2:
        return np.empty(0, dtype=np.uint64)
 
    N_sqrt = math.isqrt(N_max - 1)
    seed = serial_sieve(N_sqrt + 1)  # only used locally to prepend to the final result
 
    segments = []
    start = N_sqrt + 1
    while start < N_max:
        this_d = min(d, N_max - start)
        segments.append((start, this_d))
        start += this_d
 
    results = pool.map(_process_segment, segments)
    chunks = [seed.astype(np.uint64)] + results
    return np.concatenate(chunks)