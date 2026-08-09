import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Define original translation from your command
p_old = np.array([0.020935997196566872, -0.049639717297300295, 0.06164087685337155])

# 2. Define original rotation quaternion [qx, qy, qz, qw]
q = R.from_quat([0.6470003043057254, -0.2811976051028193, 0.6493197260688655, 0.2840816897489022])

# 3. Define the local shift (e.g., move 3 centimeters to the camera's right)
# In standard ROS camera_link: X is Right, Y is Down, Z is Forward
local_shift = np.array([0.03, 0.0, 0.0]) 

# 4. Rotate the local shift into the parent frame space
parent_shift = q.apply(local_shift)

# 5. Calculate new position
p_new = p_old + parent_shift

print(f"New x: {p_new[0]:.6f}")
print(f"New y: {p_new[1]:.6f}")
print(f"New z: {p_new[2]:.6f}")
