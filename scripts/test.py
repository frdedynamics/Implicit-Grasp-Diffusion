"""import torch
import numpy as np
from igd.networks import load_network

device = torch.device('cuda')
print('Loading network...')
net = load_network('data/models/IGD_pile.pt', device, model_type='igd')
net.eval()
print('Network loaded OK')

resolution = 40
x, y, z = torch.meshgrid(
    torch.linspace(-0.5, 0.5 - 1.0/resolution, resolution),
    torch.linspace(-0.5, 0.5 - 1.0/resolution, resolution),
    torch.linspace(-0.5, 0.5 - 1.0/resolution, resolution)
)
pos = torch.stack((x,y,z), dim=-1).float().unsqueeze(0).to(device)
pos = pos.view(1, resolution**3, 3)

tsdf_vol = torch.zeros((1, 1, 40, 40, 40), dtype=torch.float32).to(device)

print('Firing net(tsdf_vol, pos)...')
with torch.no_grad():
    out = net(tsdf_vol, pos)
print('SUCCESS:', [o.shape for o in out])"""

"""
2
import torch
import numpy as np
from igd.networks import load_network

device = torch.device('cuda')
print('Loading network...')
net = load_network('data/models/IGD_pile.pt', device, model_type='igd')
net.eval()
print('Network loaded OK')

resolution = 40
x, y, z = torch.meshgrid(
    torch.linspace(-0.5, 0.5 - 1.0/resolution, resolution),
    torch.linspace(-0.5, 0.5 - 1.0/resolution, resolution),
    torch.linspace(-0.5, 0.5 - 1.0/resolution, resolution)
)
pos = torch.stack((x,y,z), dim=-1).float().unsqueeze(0).to(device)
pos = pos.view(1, resolution**3, 3)

# FIXED: 5D tensor [batch=1, channels=1, D=40, H=40, W=40]
tsdf_vol = torch.zeros((1, 1, 40, 40, 40), dtype=torch.float32).to(device)

print('Firing net(tsdf_vol, pos)...')
with torch.no_grad():
    out = net(tsdf_vol, pos)
print('SUCCESS:', [o.shape for o in out])"""


"""
3
from src.igd.ConvONets.utils.libmcubes import mcubes
from src.igd.ConvONets.utils.libmise import mise
from src.igd.ConvONets.utils.libmesh import triangle_hash
from src.igd.ConvONets.utils.libvoxelize import voxelize
from src.igd.ConvONets.utils.libsimplify import simplify_mesh
from pykdtree.kdtree import KDTree
print('All extensions import OK')
"""

import open3d as o3d
import numpy as np

vol = o3d.pipelines.integration.UniformTSDFVolume(
    length=0.3,
    resolution=40,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
)

# Synthesize a flat depth image 0.15m away (centre of the volume)
width, height = 640, 480
depth = np.full((height, width), 0.15, dtype=np.float32)
color = np.zeros((height, width, 3), dtype=np.uint8)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(color),
    o3d.geometry.Image(depth),
    depth_scale=1.0,
    depth_trunc=2.0,
    convert_rgb_to_intensity=False,
)

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=width, height=height,
    fx=540.0, fy=540.0,
    cx=320.0, cy=240.0,
)

# Identity extrinsic — camera at origin looking down +Z
extrinsic = np.eye(4)
vol.integrate(rgbd, intrinsic, extrinsic)

vpc = vol.extract_voxel_point_cloud()
pts = np.asarray(vpc.points)
nrm = np.asarray(vpc.normals)
col = np.asarray(vpc.colors)

print('points shape:', pts.shape)
print('normals shape:', nrm.shape)
print('colors shape:', col.shape)
if len(pts) > 0:
    print('points range: min', pts.min(axis=0), 'max', pts.max(axis=0))
    print('normals sample:', nrm[:5])
    print('colors sample:', col[:5])
    print('normals range:', nrm.min(), nrm.max())
    print('colors range:', col.min(), col.max())