# import torch
# model = torch.load('/home/rashare/zhong/Humanoid-Terrain-Bench/legged_gym/logs/parkour_new/Sep27_18-11-45--continue_rfl_1_0.5/2/model_41500.pt')
# print(model.keys())

# python
import torch
pth = '/home/rashare/zhong/Humanoid-Terrain-Bench/legged_gym/logs/parkour_new/Sep27_18-11-45--continue_rfl_1_0.5/2/model_41500.pt'
data = torch.load(pth, map_location="cpu")
print(type(data))
if isinstance(data, dict):
    for k, v in data.items():
        print(k, ":", type(v))
        if hasattr(v, "keys"):
            try:
                print("  keys:", list(v.keys())[:20])
            except:
                pass
        elif torch.is_tensor(v):
            print("  shape:", v.shape, "dtype:", v.dtype)
else:
    print(repr(data))