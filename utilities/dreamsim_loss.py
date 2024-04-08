# pip install dreamsim

import torch
import torchvision.transforms as transforms
from dreamsim import dreamsim

class DreamsimFeatureLoss(torch.nn.Module):
    def __init__(self, device=None):
        super(DreamsimFeatureLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, preprocess = dreamsim(pretrained=True)
        self.model = self.model.to(device, dtype=torch.float16)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ])
    def forward(self, pred, target):
        pred_norm = (pred + 1.0)/2
        traget_norm = (target + 1.0)/2
        pred_process = self.transform(pred_norm)
        traget_process = self.transform(traget_norm)
        loss = self.model(pred_process.to(torch.float32), traget_process.to(torch.float32))
        return loss