"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction
from torchvision.ops import nms



class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32
        - level p6: (out_channels, H / 64, W / 64)      stride = 64
        - level p7: (out_channels, H / 128, W / 128)    stride = 128

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        #_cnn = models.regnet_x_400mf(pretrained=True)
        _cnn=models.resnet50(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "layer2.2.conv3": "c3",
                "layer3.2.conv3": "c4",
                "layer4.2.conv3": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        #out_channels=256 #The output channels are 256 in the FCOS paper
        # Replace "pass" statement with your code
        self.fpn_params['p5'] = nn.Conv2d(dummy_out['c5'].shape[1], out_channels, 1)
        self.fpn_params['p4'] = nn.Conv2d(dummy_out['c4'].shape[1], out_channels, 1)
        self.fpn_params['p3'] = nn.Conv2d(dummy_out['c3'].shape[1], out_channels, 1)

        self.fpn_params['p6']=nn.Conv2d(out_channels, out_channels, 3,stride=2,padding=1)
        self.fpn_params['p7']=nn.Conv2d(out_channels, out_channels, 3,stride=2,padding=1)
        
        self.fpn_params['p7_out']=nn.Conv2d(out_channels,out_channels,3,stride=1,padding=1)
        self.fpn_params['p6_out']=nn.Conv2d(out_channels,out_channels,3,stride=1,padding=1)
        self.fpn_params['p5_out']=nn.Conv2d(out_channels,out_channels,3,stride=1,padding=1)
        self.fpn_params['p4_out']=nn.Conv2d(out_channels,out_channels,3,stride=1,padding=1)
        self.fpn_params['p3_out']=nn.Conv2d(out_channels,out_channels,3,stride=1,padding=1)

        torch.nn.init.normal_(self.fpn_params['p7_out'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p7_out'].bias,0.0)

        torch.nn.init.normal_(self.fpn_params['p6_out'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p6_out'].bias,0.0)
        
        torch.nn.init.normal_(self.fpn_params['p5_out'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p5_out'].bias,0.0)

        torch.nn.init.normal_(self.fpn_params['p4_out'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p4_out'].bias,0.0)

        torch.nn.init.normal_(self.fpn_params['p3_out'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p3_out'].bias,0.0)

        torch.nn.init.normal_(self.fpn_params['p7'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p7'].bias,0.0)

        torch.nn.init.normal_(self.fpn_params['p6'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p6'].bias,0.0)


        torch.nn.init.normal_(self.fpn_params['p5'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p5'].bias,0.0)
        
        torch.nn.init.normal_(self.fpn_params['p4'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p4'].bias,0.0)
        
        torch.nn.init.normal_(self.fpn_params['p3'].weight,mean=0.0,std=0.01)
        torch.nn.init.constant_(self.fpn_params['p3'].bias,0.0)
        
        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32, "p6":64, "p7":128}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None, "p6": None, "p7": None}
        ######################################################################
        # Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.              #
        ######################################################################

        # Replace "pass" statement with your code
        fpn_feats['p5']=self.fpn_params['p5'](backbone_feats['c5'])

        fpn_feats['p4']=self.fpn_params['p4'](backbone_feats['c4']) + F.interpolate(fpn_feats['p5'],scale_factor=[2,2])
        
        fpn_feats['p3']=self.fpn_params['p3'](backbone_feats['c3'])+ F.interpolate(fpn_feats['p4'],scale_factor=[2,2])

        fpn_feats['p6']=self.fpn_params['p6'](fpn_feats['p5'])

        fpn_feats['p7']=self.fpn_params['p7'](fpn_feats['p6'])


        # Apply the anti-aliasing conv layers
        fpn_feats['p7']=self.fpn_params['p7_out'](fpn_feats['p7'])
        fpn_feats['p6']=self.fpn_params['p6_out'](fpn_feats['p6'])
        fpn_feats['p5']=self.fpn_params['p5_out'](fpn_feats['p5'])
        fpn_feats['p4']=self.fpn_params['p4_out'](fpn_feats['p4'])
        fpn_feats['p3']=self.fpn_params['p3_out'](fpn_feats['p3'])
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        location_coords[level_name]=torch.zeros((feat_shape[2]*feat_shape[3],2),dtype=dtype,device=device)
        tmp=torch.zeros((feat_shape[2],feat_shape[3]),dtype=dtype,device=device)
        '''
        for j in range (feat_shape[2]):
          for i in range(feat_shape[3]):
            tmp[j,i,0]=level_stride*(i+0.5)
            tmp[j,i,1]=level_stride*(j+0.5)
        location_coords[level_name]=tmp.reshape(-1,2)
        '''
        tmp_indices=torch.nonzero(tmp==0)
        tmp_indices =(tmp_indices+0.5)*level_stride
        location_coords[level_name]=torch.fliplr(tmp_indices)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords

def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
