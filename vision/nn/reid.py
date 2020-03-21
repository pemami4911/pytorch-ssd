import torch.nn as nn
import torch.nn.functional as F
import torch


from ..utils import descriptor_utils 


class NTXEntLoss(nn.Module):
    def __init__(self, temperature):
        super(NTXEntLoss, self).__init__()
        self.temperature = temperature
        self.cosine = torch.nn.CosineSimilarity(dim=2)

    def forward(self, features1, features2, bbox1, bbox2):
        """

        Args:
            features1/2 (batch_size, D, Hc, Wc)
            boxes1/2 (batch_size, 4): real boxes w/ values in [0,1]
        """
        # (B,D)
        batch_size = features1.shape[0]
        desc1 = descriptor_utils.get_descriptors(features1, bbox1.squeeze(1).float())
        desc2 = descriptor_utils.get_descriptors(features2, bbox2.squeeze(1).float())
        
        desc1 = desc1.unsqueeze(1).repeat_interleave(batch_size,1)  # (B,B,D)
        desc2 = desc2.unsqueeze(0).repeat_interleave(batch_size,0)  # (B,B,D)

        cosine_sim = self.cosine(desc1,desc2)  # (B,B)
        pos_pairs = torch.diag(cosine_sim)  # (B)
        # set diag element to -inf
        neg_pairs = cosine_sim.clone() # (B,B)
        diag_elems = list(range(batch_size))
        neg_pairs[diag_elems,diag_elems] = float('-inf')
        
        pos_pairs = pos_pairs / self.temperature
        neg_pairs = neg_pairs / self.temperature

        row_max_vals = torch.max(pos_pairs, torch.max(neg_pairs, dim=0, keepdim=False)[0])
        col_max_vals = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=False)[0])
        
        # rowwise
        row_numerator = torch.exp(pos_pairs - row_max_vals) # (B)
        row_denominator = torch.sum(torch.exp(neg_pairs - row_max_vals), dim=1) + row_numerator
        row_log_exp = torch.log((row_numerator / row_denominator) + 1e-20)  # (B)
        
        col_numerator = torch.exp(pos_pairs - col_max_vals) # (B)
        col_denominator = torch.sum(torch.exp(neg_pairs - col_max_vals), dim=0) + col_numerator
        col_log_exp = torch.log((col_numerator / col_denominator) + 1e-20)  # (B)

        loss = -row_log_exp
        loss += -col_log_exp

        loss = torch.sum(loss) / (2. * batch_size)
        return loss
