#!/usr/bin/env python3
import argparse
import numpy as np
from igd.utils.transform import Transform

ap = argparse.ArgumentParser()
ap.add_argument("extrinsic", help="path to a saved task->camera extrinsic .npy file")
args = ap.parse_args()

extr_path = args.extrinsic
T = np.load(extr_path).astype(np.float64)

t_obj = Transform.from_matrix(T)
T_roundtrip = t_obj.as_matrix()

print("Original matrix:")
print(T)
print("\nTransform.from_matrix(...).as_matrix() roundtrip:")
print(T_roundtrip)
print("\nMax abs difference:", np.max(np.abs(T - T_roundtrip)))

# also check: does applying t_obj to a known task-frame point give the same
# camera-frame result as directly doing R @ p + t ?
p_task = np.array([0.1, 0.1, 0.05])
via_transform = t_obj.apply(p_task)
via_manual = T[:3,:3] @ p_task + T[:3,3]
print("\nt_obj.apply(p):        ", via_transform)
print("manual R@p + t:        ", via_manual)
print("difference:            ", via_transform - via_manual)