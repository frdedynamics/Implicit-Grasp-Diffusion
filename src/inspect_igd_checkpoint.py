import torch
ckpt = torch.load("data/models/IGD_packed.pt", map_location="cpu")
print(ckpt['encoder.conv_in.weight'].shape)
# Expected: [out_channels, 1, kH, kW, kD]  ← in_channels=1 confirms single TSDF channel