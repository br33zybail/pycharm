import numpy as np
import math

def l1_norm(v):
    return sum(abs(x) for x in v)

def l2_norm(v):
    return math.sqrt(sum(x * x for x in v))

def linf_norm(v):
    return max(abs(x) for x in v)

#Test Vector
v = [3, -4, 2]

#Compute
l1 = l1_norm(v)
l2 = l2_norm(v)
linf = linf_norm(v)

#Verification
v_np = np.array(v)
l1_np = np.sum(np.abs(v_np))
l2_np = np.sqrt(np.sum(v_np ** 2))
linf_np = np.max(np.abs(v_np))

l1_test = np.linalg.norm(v_np, ord=1)
print(l1_test)
l2_test = np.linalg.norm(v_np)
print(l2_test)
l3_test = np.linalg.norm(v_np, ord=np.inf)
print(l3_test)

print(f"L1 Norm: Custom = {l1}, NumPy = {l1_np}, Equal? {l1 == l1_np}")
print(f"L2 Norm: Custom = {l2:.2f}, NumPy = {l2_np:.2f}, Equal? {abs(l2 - l2_np) < 1e-10}")
print(f"L∞ Norm: Custom = {linf}, NumPy = {linf_np}, Equal? {linf == linf_np}")