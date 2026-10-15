import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import pyiqa

class FVRQualityMetrics:
   
    def __init__(self, device="cpu"):
        self.device = device
        
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        
        self.niqe = pyiqa.create_metric('niqe', device=self.device, as_loss=False)

        self._niqe_sum = 0.0
        self._niqe_count = 0

    def evaluate_batch(self, restored_imgs, sharp_imgs):

        # PSNR/SSIM are defined for data_range=1.0, so anything the network
        # pushes outside [0, 1] has to be clipped before scoring.
        restored_imgs = restored_imgs.to(self.device).clamp(0.0, 1.0)
        sharp_imgs = sharp_imgs.to(self.device).clamp(0.0, 1.0)

        psnr_val = self.psnr(restored_imgs, sharp_imgs)
        ssim_val = self.ssim(restored_imgs, sharp_imgs)

        niqe_val = self.niqe(restored_imgs)
        self._niqe_sum += niqe_val.mean().item() * restored_imgs.shape[0]
        self._niqe_count += restored_imgs.shape[0]

        return {
            "PSNR": psnr_val.item(),
            "SSIM": ssim_val.item(),
            "NIQE": niqe_val.mean().item()
        }

    def compute(self):

        # torchmetrics accumulates across evaluate_batch calls; this returns the
        # dataset-level value, which is not the same as averaging per-batch PSNR.
        return {
            "PSNR": self.psnr.compute().item(),
            "SSIM": self.ssim.compute().item(),
            "NIQE": self._niqe_sum / max(self._niqe_count, 1),
        }

    def reset(self):

        self.psnr.reset()
        self.ssim.reset()
        self._niqe_sum = 0.0
        self._niqe_count = 0

if __name__ == "__main__":
    dummy_restored = torch.rand(2, 3, 256, 256)
    dummy_sharp = torch.rand(2, 3, 256, 256)
    
    print("Loading metric models...")
    evaluator = FVRQualityMetrics()
    
    print("Calculating metrics...")
    results = evaluator.evaluate_batch(dummy_restored, dummy_sharp)
    
    print("\n--- FVR Batch Evaluation Results ---")
    print(f"PSNR (Higher is better): {results['PSNR']:.4f} dB")
    print(f"SSIM (Higher is better): {results['SSIM']:.4f}")
    print(f"NIQE (Lower is better):  {results['NIQE']:.4f}")
    print("Metrics Evaluation Suite is fully operational!")