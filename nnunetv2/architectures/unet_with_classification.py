from typing import Union, Type, List, Tuple
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from torch import nn
from torch.nn.modules.conv import _ConvNd


class PlainConvUNetWithClassification(PlainConvUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 num_classification_classes: int,
                 bottleneck_dim: int = 512,
                 **kwargs):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op,
                        kernel_sizes, strides, n_conv_per_stage, num_classes,
                        n_conv_per_stage_decoder, **kwargs)

        # Get dimensionality from conv_op
        dim = 2 if conv_op == nn.Conv2d else 3
        
        # Get number of features at bottleneck
        if isinstance(features_per_stage, (list, tuple)):
            bottleneck_features = features_per_stage[-1]
        else:
            bottleneck_features = features_per_stage
            
        # Classification head with conv layers
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        norm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
        
        self.classification_head = nn.Sequential(
            # Reduce spatial dimensions gradually
            conv(bottleneck_features, bottleneck_features//2, kernel_size=3, padding=1),
            norm(bottleneck_features//2),
            nn.GELU(),
            
            conv(bottleneck_features//2, bottleneck_features//4, kernel_size=3, padding=1),
            norm(bottleneck_features//4),
            nn.GELU(),
            
            # Final spatial reduction
            conv(bottleneck_features//4, bottleneck_features//4, kernel_size=1),
            norm(bottleneck_features//4),
            nn.GELU(),
            
            # Now pool
            nn.AdaptiveAvgPool3d(1) if dim == 3 else nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Final classification layers
            nn.Linear(bottleneck_features//4, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(bottleneck_dim, num_classification_classes)
        )

    def forward(self, x):
        # Get features through encoder
        skips = self.encoder(x)
        
        # Get bottleneck features (last skip connection)
        bottleneck = skips[-1]
        
        # Get classification output
        classification_output = self.classification_head(bottleneck)
        
        # Get segmentation output
        segmentation_output = self.decoder(skips)
        
        return segmentation_output, classification_output
    

class ResidualEncoderUNetWithClassification(ResidualEncoderUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 num_classification_classes: int,
                 bottleneck_dim: int = 512,
                 **kwargs):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op,
                        kernel_sizes, strides, n_blocks_per_stage, num_classes,
                        n_conv_per_stage_decoder, **kwargs)

        # Get dimensionality from conv_op
        dim = 2 if conv_op == nn.Conv2d else 3
        
        # Get number of features at bottleneck
        if isinstance(features_per_stage, (list, tuple)):
            bottleneck_features = features_per_stage[-1]
        else:
            bottleneck_features = features_per_stage
            
        # Classification head with conv layers
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        norm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
        
        self.classification_head = nn.Sequential(
            # Reduce spatial dimensions gradually
            conv(bottleneck_features, bottleneck_features//2, kernel_size=3, padding=1),
            norm(bottleneck_features//2),
            nn.GELU(),
            
            conv(bottleneck_features//2, bottleneck_features//4, kernel_size=3, padding=1),
            norm(bottleneck_features//4),
            nn.GELU(),
            
            # Final spatial reduction
            conv(bottleneck_features//4, bottleneck_features//4, kernel_size=1),
            norm(bottleneck_features//4),
            nn.GELU(),
            
            # Now pool
            nn.AdaptiveAvgPool3d(1) if dim == 3 else nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Final classification layers
            nn.Linear(bottleneck_features//4, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(bottleneck_dim, num_classification_classes)
        )

    def forward(self, x):
        # Get features through encoder
        skips = self.encoder(x)
        
        # Get bottleneck features (last skip connection)
        bottleneck = skips[-1]
        
        # Get classification output
        classification_output = self.classification_head(bottleneck)
        
        # Get segmentation output
        segmentation_output = self.decoder(skips)
        
        return segmentation_output, classification_output