import numpy as np

def check_correctness(f):
    out1 = f(111)
    out2 = f(110)
    out3 = f(100_000_000)
    expected1_2 = np.array(
        [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
            101,
            103,
            107,
            109,
        ]
    )
    return (
        (out1 == expected1_2).all()
        and (out2 == expected1_2).all()
        and out3.size == 5_761_455
        and out3[-1] == 99_999_989
    )


from algorithm_versions.serial_versions import (
    serial_int_array_sieve,
    serial_bitarray_sieve,
    serial_bool_sieve,
)

# print(check_correctness(serial_bitarray_sieve))

# tests/test_serial_segmented_vs_numba.py
"""
Compare the serial segmented sieve implementation against the numba full-sieve.
- serial_segmented(N_max, d) should return all primes < N_max (it builds its own seed).
- numba_parallel_inner_loop_bool_sieve(N_max) returns all primes < N_max.

This test runs for N_max in [3..1000] and for several d values and asserts equality.
"""

import math
import numpy as np

from algorithm_versions.segmented_versions import serial_segmented
from algorithm_versions.inner_loop_opt_versions import numpy_vectorized_inner_loop

def test_serial_segmented_matches_vectorized():
    test_cases = [
        (1000, 100),
        (1000, 137),   # d does not evenly divide N_max
        (26, 5),       # N_max = 5^2 + 1, the perfect-square edge case
        (10000, 7),    # tiny, awkward segment size
        (2, 10),       # d larger than N_max
    ]
 
    for N_max, d in test_cases:
        expected = numpy_vectorized_inner_loop(N_max)
        got = serial_segmented(N_max, d)
        print(got)
        assert np.array_equal(got, expected), f"Mismatch for N_max={N_max}, d={d}"
 
 

test_serial_segmented_matches_vectorized()
print("OK: serial_segmented matches numpy_vectorized_inner_loop on all test cases")
 
 