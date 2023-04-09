from itertools import permutations
import numpy as np
def complete_perm(arr):
    perm_list = list(permutations(arr))
    perm_sign = list(permutations([-1,-1,1,1]))
    perm = []
    
    for i in range(0,len(perm_list)):
        for j in range(0,len(perm_sign)):
            perm.append(list(np.multiply(perm_list[i],perm_sign[j])))
    return perm
print(complete_perm(np.array([1,2,3,4])))
