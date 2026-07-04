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


from serial_versions import (
    serial_int_array_sieve,
    serial_bitarray_sieve,
    serial_bool_sieve,
)

print(check_correctness(serial_bitarray_sieve))
