import torch
import torch.nn.functional as F
import torchxrayvision as xrv


class LungMasker:
    def __init__(self, device, img_size=512, threshold=0.5):
        self.device = device
        self.img_size = img_size
        self.threshold = threshold
        self.model = xrv.baseline_models.chestx_det.PSPNet().to(device)
        self.model.eval()
        self.targets = self.model.targets
        self.left_idx = self.targets.index("Left Lung")
        self.right_idx = self.targets.index("Right Lung")
        self.center_crop = xrv.datasets.XRayCenterCrop()
        self.resizer = xrv.datasets.XRayResizer(img_size)

    def __call__(self, img_np):
        img = xrv.datasets.normalize(img_np, 255)
        img = img.mean(2)[None, ...]
        img = self.center_crop(img)
        img = self.resizer(img)
        img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(img_t)

        left = out[:, self.left_idx]
        right = out[:, self.right_idx]
        lung = torch.clamp(left + right, 0, 1)
        lung = (lung > self.threshold).float()
        return lung.squeeze(0).detach()


def resize_mask(mask_t, size):
    if mask_t.dim() == 2:
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)
    elif mask_t.dim() == 3:
        mask_t = mask_t.unsqueeze(1)

    mask_t = F.interpolate(mask_t, size=size, mode="nearest")
    return mask_t.squeeze().cpu().numpy()
