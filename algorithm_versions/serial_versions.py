import numpy as np
from bitarray import bitarray

def serial_int_array_sieve(N_max): 
    """
    Straightforward implementation of the sieve of Eratosthenes.
    Returns all primes < N_max.
    
    Internally the sieve is represented by an np.array of int!  
    """
    if N_max <= 2:
        return []
    
    sieve = np.arange(0, N_max, 1, np.uint64)
    sieve[1] = 0
    
    N_sqrt = int(np.floor(np.sqrt(N_max)))  

    for i in range(2, N_sqrt+1):
        if  sieve[i] != 0:
            for j in range(i**2, N_max, i):
                sieve[j] = 0

    primes = sieve[sieve != 0]
    return primes

# print(serial_sieve_int_array(19))

def serial_bool_sieve(N_max): 
    """
    Straightforward implementation of the sieve of Eratosthenes.
    Returns all primes < N_max.
    
    Internally the sieve is represented by an np.array of bools.
    An additional array   
    """
    if N_max <= 2:
        return []
        
    sieve = np.ones(N_max, dtype=np.bool_)
    sieve[0:2] = False  

    N_sqrt = int(np.floor(np.sqrt(N_max)))  

    for i in range(2, N_sqrt+1):
        if  sieve[i] != 0:
            for j in range(i**2, N_max, i):
                sieve[j] = False
      
    primes = np.flatnonzero(sieve)         
    return primes

# print(serial_bool_sieve(26))

def serial_bitarray_sieve(N_max): 
    """
    Sieve of Eratosthenes using bitarray (1 bit per entry).
    Returns a list of primes < N_max.
    """
    if N_max <= 2:
        return []

    sieve = bitarray(N_max)
    sieve.setall(True)
    sieve[0:2] = False
    
    N_sqrt = int(np.floor(np.sqrt(N_max)))  
    
    for i in range(2, N_sqrt+1):
        if  sieve[i] != 0:
            for j in range(i**2, N_max, i):
                sieve[j] = False
      
    count = sieve.count() 
    primes = np.zeros(count, dtype=np.uint64)
    
    idx = 0
    for i in sieve.search(1):        # iterator over set-bit indices
        primes[idx] = i
        idx += 1

    return primes    
    
# big = serial_int_array_sieve(1_000_000_000)    
# print(big[-1])    
    
    
###############    
## Not used  ##
###############    
    
# def serial_bitarray_sieve_w_dtype(N_max, np_dtype=np.uint64): 
#     """
#     Sieve of Eratosthenes using bitarray (1 bit per entry).
#     Returns a list of primes < N_max.
#     """
#     if N_max <= 2:
#         return []

#     sieve = bitarray(N_max)
#     sieve.setall(True)
#     sieve[0:2] = False
    
#     N_sqrt = int(np.floor(np.sqrt(N_max)))  
    
#     i = np_dtype(2)
#     j = np_dtype(0)
    
#     while i < N_sqrt:
#         if  sieve[i] != 0:
#             j = i**2
#             while j < N_max:
#                 sieve[j] = False
#                 j += i
#         i += 1
      
#     count = sieve.count() 
#     primes = np.zeros(count, dtype=np_dtype)
    
#     idx = 0
#     for i in sieve.search(1):     
#         primes[idx] = i
#         idx += 1

#     return primes  



def serial_int_array_sieve_no_return(N_max): 
    """
    Straightforward implementation of the sieve of Eratosthenes.
    Returns all primes < N_max.
    
    Internally the sieve is represented by an np.array of int!  
    """
    if N_max <= 2:
        return []
    
    sieve = np.arange(0, N_max, 1, np.uint64)
    sieve[1] = 0
    
    N_sqrt = int(np.floor(np.sqrt(N_max)))  

    for i in range(2, N_sqrt+1):
        if  sieve[i] != 0:
            for j in range(i**2, N_max, i):
                sieve[j] = 0

    return 


def serial_bool_sieve_no_return(N_max): 
    """
    Straightforward implementation of the sieve of Eratosthenes.
    Returns all primes < N_max.
    
    Internally the sieve is represented by an np.array of bools.
    An additional array   
    """
    if N_max <= 2:
        return []
        
    sieve = np.ones(N_max, dtype=np.bool_)
    sieve[0:2] = False  

    N_sqrt = int(np.floor(np.sqrt(N_max)))  

    for i in range(2, N_sqrt+1):
        if  sieve[i] != 0:
            for j in range(i**2, N_max, i):
                sieve[j] = False
            
    return 

def serial_bitarray_sieve_no_return(N_max): 
    """
    Sieve of Eratosthenes using bitarray (1 bit per entry).
    Returns a list of primes < N_max.
    """
    if N_max <= 2:
        return []

    sieve = bitarray(N_max)
    sieve.setall(True)
    sieve[0:2] = False
    
    N_sqrt = int(np.floor(np.sqrt(N_max)))  
    
    for i in range(2, N_sqrt+1):
        if  sieve[i] != 0:
            for j in range(i**2, N_max, i):
                sieve[j] = False
      
    return     
    