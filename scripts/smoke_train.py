import sys, time, os
sys.path.insert(0, os.path.expanduser("~/FVR_Project"))
import torch
from torch.utils.data import DataLoader
from utils.dataset import FVRDataset
from models.fvr_net import FVRNet

dev = "cuda"
torch.manual_seed(0)

root = os.path.expanduser("~/FVR_Project/data/processed/train")
d = FVRDataset(os.path.join(root, "blurry"),
               os.path.join(root, "sharp"), patch_size=128)
loader = DataLoader(d, batch_size=4, shuffle=True, num_workers=4, drop_last=True)

model = FVRNet(depth=2, use_checkpoint=True).train().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

losses = []
t0 = time.time()
step = 0
for blurry, sharp in loader:
    blurry, sharp = blurry.to(dev), sharp.to(dev)
    opt.zero_grad()
    out = model(blurry)
    loss = torch.nn.functional.l1_loss(out, sharp)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    step += 1
    if step % 10 == 0:
        avg = sum(losses[-10:]) / 10
        print(f"step {step:3d}  loss {loss.item():.4f}  (avg10 {avg:.4f})  {(time.time()-t0)/step:.2f}s/step", flush=True)
    if step >= 100:
        break

print()
print(f"first-10 avg loss: {sum(losses[:10])/10:.4f}")
print(f"last-10  avg loss: {sum(losses[-10:])/10:.4f}")
sps = (time.time()-t0)/step
print(f"sec/step: {sps:.2f}   -> full epoch (~2550 steps): {sps*2550/60:.1f} min")
print("LEARNING" if sum(losses[-10:]) < sum(losses[:10]) else "NOT DECREASING - investigate")
