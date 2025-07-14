# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:43:29 2023

@author: Tahasanul Abraham
"""

# %% Initialization of Libraries and Directory

# Standard library imports

# Third-party imports
import torchvision
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

#%% model development

def new_model_head(model, num_classes):
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    
    return model  

#%% create_model

def create_model(num_classes, pretrained, model_name = 'resnet18'):
    
    model = None
    
#%% vgg--GPU1-done
    if 'vgg'in model_name.lower():
        
        if pretrained is True:
            backbone_model = torchvision.models.vgg16( weights = torchvision.models.VGG16_Weights.DEFAULT )
        else:
            backbone_model = torchvision.models.vgg16( )
        
        out_channels = backbone_model.features[-3].out_channels
            
        print ( "Loading Faster RCNN..." )
        print ( "Backbone Model: \"VGG_16\"" )            
        print ( "Pretained: {}".format( pretrained ) )
        backbone = backbone_model.features
        
        backbone.out_channels = out_channels
        
        
        anchorgenerator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),))
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator= anchorgenerator,
            box_roi_pool=roi_pooler
        )
        
#%% resnet--GPU1-done
    elif 'resnet' in model_name.lower():
        if pretrained == True:
            backbone_model =  torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
        else:
            backbone_model = torchvision.models.resnet18()
        
        print ( "Loading Faster RCNN..." )
        print ( "Backbone Model: \"Resnet_18\"" )            
        print ( "Pretained: {}".format( pretrained ) )
        
        conv1 = backbone_model.conv1
        bn1 = backbone_model.bn1
        relu = backbone_model.relu
        maxpool = backbone_model.maxpool
        layer1 = backbone_model.layer1
        layer2 = backbone_model.layer2
        layer3 = backbone_model.layer3
        layer4 = backbone_model.layer4
        out_channels = backbone_model.fc.in_features
        
        backbone = nn.Sequential(
                                    conv1, bn1, relu, maxpool, 
                                    layer1, layer2, layer3, layer4
                                )
        
        backbone.out_channels = out_channels
        
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
                                                                                sizes=((32, 64, 128, 256, 512),),
                                                                                aspect_ratios=((0.5, 1.0, 2.0),)
                                                                            )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                                                            featmap_names=['0'],
                                                            output_size=7,
                                                            sampling_ratio=2
                                                        )
        
        model = torchvision.models.detection.FasterRCNN(
                                                            backbone=backbone,
                                                            num_classes=num_classes,
                                                            rpn_anchor_generator=anchor_generator,
                                                            box_roi_pool=roi_pooler
                                                        )
        
#%% convnext--GPU2-done    
    elif "convnext" in model_name.lower( ):
        
                   
        if pretrained is True:
            backbone_model = torchvision.models.convnext_base( weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT )
        else:
            backbone_model = torchvision.models.convnext_base( )
                
        print ( "Loading Faster RCNN..." )
        print ( "Backbone Model: \"convnext_base\"" )            
        print ( "Pretained: {}".format( pretrained ) )
            
             
        
        backbone = backbone_model.features
        
        out_channels = backbone_model.classifier[2].in_features
        
        backbone.out_channels = out_channels
        
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator( 
                                                                                sizes=( ( 32, 64, 128, 256, 512 ), ),
                                                                                aspect_ratios=( ( 0.5, 1.0, 2.0 ), )
                                                                            )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign( 
                                                            featmap_names=["0"],
                                                            output_size=7,
                                                            sampling_ratio=2
                                                        )
        
        
        model = torchvision.models.detection.FasterRCNN( 
                                                            backbone=backbone,
                                                            num_classes=num_classes,
                                                            rpn_anchor_generator=anchor_generator,
                                                            box_roi_pool=roi_pooler
                                                        )
        
#%% squeezenet--LINUX-done
    elif "squeezenet" in model_name.lower( ):
        
                   
        if pretrained is True:
            backbone_model = torchvision.models.squeezenet1_0( weights = torchvision.models.SqueezeNet1_0_Weights.DEFAULT )
        else:
            backbone_model = torchvision.models.squeezenet1_0( )
                
        print ( "Loading Faster RCNN..." )
        print ( "Backbone Model: \"squeezenet1_0\"" )            
        print ( "Pretained: {}".format( pretrained ) )
            
             
        
        backbone = backbone_model.features
        
        out_channels = backbone_model.classifier[1].in_channels
        
        backbone.out_channels = out_channels
        
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator( 
                                                                                sizes=( ( 32, 64, 128, 256, 512 ), ),
                                                                                aspect_ratios=( ( 0.5, 1.0, 2.0 ), )
                                                                            )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign( 
                                                            featmap_names=["0"],
                                                            output_size=7,
                                                            sampling_ratio=2
                                                        )
        
        
        model = torchvision.models.detection.FasterRCNN( 
                                                            backbone=backbone,
                                                            num_classes=num_classes,
                                                            rpn_anchor_generator=anchor_generator,
                                                            box_roi_pool=roi_pooler
                                                        )
        
#%% efficientnet--LINUX-done 
    elif "efficientnet" in model_name.lower( ):
        
                   
        if pretrained is True:
            backbone_model = torchvision.models.efficientnet_b0( weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT )
        else:
            backbone_model = torchvision.models.efficientnet_b0( )
                
        print ( "Loading Faster RCNN..." )
        print ( "Backbone Model: \"efficientnet_b0\"" )            
        print ( "Pretained: {}".format( pretrained ) )
            
             
        
        backbone = backbone_model.features
        
        out_channels = backbone_model.classifier[1].in_features
        
        backbone.out_channels = out_channels
        
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator( 
                                                                                sizes=( ( 32, 64, 128, 256, 512 ), ),
                                                                                aspect_ratios=( ( 0.5, 1.0, 2.0 ), )
                                                                            )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign( 
                                                            featmap_names=["0"],
                                                            output_size=7,
                                                            sampling_ratio=2
                                                        )
        
        
        model = torchvision.models.detection.FasterRCNN( 
                                                            backbone=backbone,
                                                            num_classes=num_classes,
                                                            rpn_anchor_generator=anchor_generator,
                                                            box_roi_pool=roi_pooler
                                                        )
        
#%% shufflenet--LINUX-done
    elif "shufflenet" in model_name.lower( ):
        
                   
        if pretrained is True:
            backbone_model = torchvision.models.shufflenet_v2_x1_0( weights = torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT )
        else:
            backbone_model = torchvision.models.shufflenet_v2_x1_0( )
                
        print ( "Loading Faster RCNN..." )
        print ( "Backbone Model: \"shufflenet_v2_x1_0\"" )            
        print ( "Pretained: {}".format( pretrained ) )
            
        conv1 = backbone_model.conv1
        maxpool = backbone_model.maxpool
        stage2 = backbone_model.stage2
        stage3 = backbone_model.stage3
        stage4 = backbone_model.stage4
        conv5 = backbone_model.conv5        
        out_channels = backbone_model.fc.in_features
        
        backbone = nn.Sequential( 
                                    conv1, maxpool, stage2,
                                    stage3, stage4, conv5
                                )
        
        backbone.out_channels = out_channels
        
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator( 
                                                                                sizes=( ( 32, 64, 128, 256, 512 ), ),
                                                                                aspect_ratios=( ( 0.5, 1.0, 2.0 ), )
                                                                            )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign( 
                                                            featmap_names=["0"],
                                                            output_size=7,
                                                            sampling_ratio=2
                                                        )
        
        
        model = torchvision.models.detection.FasterRCNN( 
                                                            backbone=backbone,
                                                            num_classes=num_classes,
                                                            rpn_anchor_generator=anchor_generator,
                                                            box_roi_pool=roi_pooler
                                                        )
        
#%% mobilenet--GPU2-done 
    elif "mobilenet" in model_name.lower( ):
                                     
        if pretrained is True:
            backbone_model = torchvision.models.mobilenet_v2( weights = torchvision.models.MobileNet_V2_Weights.DEFAULT )                    
        else:
            backbone_model = torchvision.models.mobilenet_v2( )
            
        print ( "Loading Faster RCNN..." )
        print ( "Backbone Model: \"mobilenet_v2\"" )            
        print ( "Pretained: {}".format( pretrained ) )                     
        
        out_channels = backbone_model.classifier[1].in_features
        
        
        backbone = backbone_model.features
        
        backbone.out_channels = out_channels
        
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator( 
                                                                                sizes=( ( 32, 64, 128, 256, 512 ), ),
                                                                                aspect_ratios=( ( 0.5, 1.0, 2.0 ), )
                                                                            )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign( 
                                                            featmap_names=["0"],
                                                            output_size=7,
                                                            sampling_ratio=2
                                                        )
        
              
        model = torchvision.models.detection.FasterRCNN( 
                                                            backbone=backbone,
                                                            num_classes=num_classes,
                                                            rpn_anchor_generator=anchor_generator,
                                                            box_roi_pool=roi_pooler
                                                        )


    
    return new_model_head(model, num_classes)
    
    
if __name__ == "__main__":
    
 
   pass
     