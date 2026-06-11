# How to run:
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python scripts/sim_grasp_multiple.py \
  --num-view 1 --object-set pile/test --scene pile \
  --num-rounds 5 \
  --sideview \
  --add-noise dex \
  --force --best \
  --model ./data/models/IGD_pile.pt \
  --type igd \
  --result-path results/debug_visual \
  --vis \
  --sim-gui
  
(remove --vis and --sim-gui for headless mode)


# Additional downloads:
wget -O src/igd/ConvONets/utils/libmcubes/marchingcubes.cpp \
  https://raw.githubusercontent.com/autonomousvision/convolutional_occupancy_networks/master/src/utils/libmcubes/marchingcubes.cpp

wget -O src/igd/ConvONets/utils/libmcubes/marchingcubes.h \
  https://raw.githubusercontent.com/autonomousvision/convolutional_occupancy_networks/master/src/utils/libmcubes/marchingcubes.h

wget -O src/igd/ConvONets/utils/libmcubes/pywrapper.cpp \
  https://raw.githubusercontent.com/autonomousvision/convolutional_occupancy_networks/master/src/utils/libmcubes/pywrapper.cpp


# Environment info
python scripts/check_env.py 
==================================================
     IGD / GIGA PROJECT ENVIRONMENT INFRASTRUCTURE
==================================================
OS / Platform:            Linux-6.8.1-1048-realtime-x86_64-with-glibc2.39
Linux Kernel:             6.8.1-1048-realtime
Python Version:           3.11.15
--------------------------------------------------
PyTorch:                  2.11.0+cu128
CUDA Available:           True
CUDA Device Name:         NVIDIA GeForce RTX 5090
Numpy:                    1.26.4
Scipy:                    1.17.1
Trimesh:                  4.12.2
Open3d:                   0.19.0
--------------------------------------------------
pybullet build time: Jan 29 2025 23:17:20
PyBullet:                 3.2.7
URDFpy:                   0.0.4 (Patched)
ROS 1 (rospy):            Available
==================================================
