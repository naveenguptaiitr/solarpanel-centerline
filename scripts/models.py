import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import deeplabv3_resnet50


class ConvNeXtModel(nn.Module):
    def __init__(self, backbone_name="convnext_tiny", num_classes=1, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,    
            pretrained=pretrained,
            in_chans=3
        )


        self.enc_channels = self.backbone.feature_info.channels()  
        self.num_layers = len(self.enc_channels)

        self.dec_channels = [256, 128, 64, 32]  
        self.up_convs = nn.ModuleList()

        for i in range(len(self.dec_channels)):
            if i == 0:
                in_ch = self.enc_channels[-1] + self.enc_channels[-2]
            elif i < len(self.dec_channels) - 1:
                in_ch = self.dec_channels[i-1] + self.enc_channels[-(i+2)]
            else:
                in_ch = self.dec_channels[i-1]
            out_ch = self.dec_channels[i]

            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )

        self.final_conv = nn.Conv2d(self.dec_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        x_input = x 
        features = self.backbone(x)

        x = features[-1]

        for i, up_conv in enumerate(self.up_convs):
            if i < len(features) - 1:
                skip = features[-(i+2)]
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            else:
                # last upconv: double spatial size
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            x = up_conv(x)

        out = self.final_conv(x)

        out = F.interpolate(out, size=x_input.shape[2:], mode='bilinear', align_corners=False)
        return out


class DeepLabV3Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet50(weights="DEFAULT")
        # Replace classifier for binary output
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        self.model.aux_classifier = None

    def forward(self, x):
        # Return only the tensor
        return self.model(x)['out']

def get_segmentation_model(model_name, model_encoder):

    if model_name == "unet":
        model = smp.Unet(encoder_name=model_encoder,
                 encoder_weights="imagenet",
                 in_channels=3,
                 classes=1)
    elif model_name == "deeplabv3" and model_encoder == "resnet50":
        model = DeepLabV3Binary()
    else:
        raise ValueError(f"Unsupported model/encoder combination: {model_name}/{model_encoder}")
    
    return model

