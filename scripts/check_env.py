import sys
import platform
import subprocess

def get_pip_version(package_name):
    """Fallback helper to check versions via pip if the module doesn't expose a __version__."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        return "Not found via pip"
    return "Unknown"

print("=" * 50)
print("     IGD / GIGA PROJECT ENVIRONMENT INFRASTRUCTURE")
print("=" * 50)

# --- Operating System & Kernel ---
print(f"{'OS / Platform:':<25} {platform.platform()}")
print(f"{'Linux Kernel:':<25} {platform.release()}")
print(f"{'Python Version:':<25} {platform.python_version()}")

print("-" * 50)

# --- Core Machine Learning ---
try:
    import torch
    print(f"{'PyTorch:':<25} {torch.__version__}")
    print(f"{'CUDA Available:':<25} {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"{'CUDA Device Name:':<25} {torch.cuda.get_device_name(0)}")
except ImportError:
    print(f"{'PyTorch:':<25} NOT INSTALLED")

# --- Core Matrix Math & Meshes ---
for lib in ["numpy", "scipy", "trimesh", "open3d"]:
    try:
        mod = __import__(lib)
        print(f"{lib.capitalize() + ':':<25} {mod.__version__}")
    except ImportError:
        print(f"{lib.capitalize() + ':':<25} NOT INSTALLED")

print("-" * 50)

# --- Simulation & Robotics Libraries ---
try:
    import pybullet
    # pybullet doesn't consistently expose a __version__ string, check via pip hook
    pb_ver = get_pip_version("pybullet")
    print(f"{'PyBullet:':<25} {pb_ver}")
except ImportError:
    print(f"{'PyBullet:':<25} NOT INSTALLED")

try:
    import urdfpy
    print(f"{'URDFpy:':<25} {get_pip_version('urdfpy')} (Patched)")
except ImportError:
    print(f"{'URDFpy:':<25} NOT INSTALLED")

try:
    import rospy
    print(f"{'ROS 1 (rospy):':<25} Available")
except ImportError:
    print(f"{'ROS 1 (rospy):':<25} Not found in this active Conda path")

print("=" * 50)