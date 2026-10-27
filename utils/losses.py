import torch
import torch.nn as nn

class RestorationLoss(nn.Module):
    """L1 + gradient (edge) + frequency (FFT) loss.
       L = L1 + w_grad * L1(gradients) + w_freq * L1(FFT real/imag).
       This is the configuration that produced the reported GoPro result."""
    def __init__(self, w_grad=0.05, w_freq=0.1):
        super().__init__()
        self.w_grad, self.w_freq = w_grad, w_freq
        self.l1 = nn.L1Loss()

    @staticmethod
    def _grad(x):
        return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]

    def forward(self, pred, target):
        loss = self.l1(pred, target)
        if self.w_grad > 0:
            px, py = self._grad(pred); tx, ty = self._grad(target)
            loss = loss + self.w_grad * (self.l1(px, tx) + self.l1(py, ty))
        if self.w_freq > 0:
            with torch.autocast(device_type=pred.device.type, enabled=False):
                pf = torch.fft.rfft2(pred.float(), norm="ortho")
                tf = torch.fft.rfft2(target.float(), norm="ortho")
                loss = loss + self.w_freq * (self.l1(pf.real, tf.real) + self.l1(pf.imag, tf.imag))
        return loss
