# Smart grasping for 2F grippers
20.05.2026 - Successfully replicated https://github.com/mousecpn/Implicit-Grasp-Diffusion with newer libraries running on RTX 5090 

## Prerequisites
```text
OS / Platform:            Linux-6.8.1-1048-realtime-x86_64-with-glibc2.39 (Ubuntu 24.04 and ROS2 Jazzy (not used yet))
Linux Kernel:             6.8.1-1048-realtime
Python Version:           3.11.15

PyTorch:                  2.11.0+cu128
CUDA Available:           True
CUDA Device Name:         NVIDIA GeForce RTX 5090
Numpy:                    1.26.4
Scipy:                    1.17.1
Trimesh:                  4.12.2
Open3d:                   0.19.0

pybullet build time: Jan 29 2025 23:17:20
PyBullet:                 3.2.7
URDFpy:                   0.0.4 (Patched)
```

1. `conda create -n [ENV_NAME] python=3.11 -y`
2. Install pytorch and CUDA.
3. Install packages list in `requirements.txt`. Then install `torch-scatter` following here, based on pytorch version and cuda version. (PS: if there is an error about sklearn when installing open3d, you can `export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True`)
4. Go to the root directory and install the project locally using pip `pip install -e .`
5. Build ConvONets dependents by running `python scripts/convonet_setup.py build_ext --inplace`.
6. `cd [PROJECT ROOT]`
7. `mkdir data`
8. download data: https://utexas.app.box.com/s/h3ferwjhuzy6ja8bzcm3nu9xq1wkn94s and https://github.com/UT-Austin-RPL/GIGA#pre-trained-models-and-pre-generated-data
9. Check additional documents in `README-Gizem.md`. Make sure that your data folder has
```text
  .
  ├── experiments
  ├── models
  ├── packed
  ├── pile
  └── urdfs
```
10. Export the zip files in pile/ and packed/

## How to run
```bash
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
```
(remove --vis and --sim-gui for headless mode)

## Real FR3 connection
NOTE: Never shutdown the robot from the button! Especially if it is FCI active mode! Use the dashboard always!

1. Browse through ´https://192.170.10.101/desk/´
2. Release the breaks, activate FCI.
3. Run the bringup with Moveit_ ´ros2 launch franka_fr3_moveit_config moveit.launch.py   robot_ip:=192.170.10.101   robot_type:=fr3   use_fake_hardware:=false´

The terminal with ROS should have these sourced:
´´´bash
source /opt/ros/jazzy/setup.bash
source ~/calibration_ws/install/setup.bash
´´´

The terminal with IGD should have ´conda activate igd_blackwell´

