import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms as T
import warnings

warnings.filterwarnings('ignore')



class DINOFetureLoss(torch.nn.Module):
    def __init__(self, device=None):
        super(DINOFetureLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dinov2_vitg14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device, dtype=torch.float16)
        self.dinov2_vitg14.eval()
            
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        
        self.augment_trans = transforms.Compose(augemntations)

        self.trans = T.Compose([
                        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                        T.CenterCrop(224),
                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
    
    def forward(self, pred, target):
        # normalize the input from [-1,1] to [0,1]
        pred = (pred/2 + 0.5).clamp(0,1)
        target = (target/2 + 0.5).clamp(0,1)
        pred = pred[:,:3]

        pred = self.trans(pred)
        target = self.trans(target)

        pred_feature = self.dinov2_vitg14.forward_features(pred)['x_prenorm']
        target_feature = self.dinov2_vitg14.forward_features(target)['x_prenorm']

        difference = torch.nn.MSELoss()(pred_feature,target_feature)
        return difference

# dino_model = DINOFetureLoss()