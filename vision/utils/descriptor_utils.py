import torch
import numpy as np

def get_descriptors(features, bboxes,shrinkage=16):
    """
    Given features of shape [B, C, H/shrinkage, W/shrinkage]
    and bboxes [B,4] in percent coordinates [0,1], extract descriptors
    at center of bounding boxes.

    Returns [B, C]
    """
    _,_,Hc,Wc = features.shape
    # compute centers
    x = bboxes[:,2] - bboxes[:,0]
    y = bboxes[:,3] - bboxes[:,1]
    x = (x * 2.) - 1.
    y = (y * 2.) - 1.  # [-1,1]
    pts = torch.cat([x.unsqueeze(1), y.unsqueeze(1)],1)  # [B,2]
    pts = pts.view(-1,1,1,2)
    sampled_features = torch.nn.functional.grid_sample(features,pts).squeeze()  # [B,C]
    sampled_features = sampled_features / torch.norm(sampled_features, 1, keepdim=True)
    return sampled_features
    
