import numpy as np
import torch


def generating_l2_norm(all_fmaps, samples_num):
    l2_norm = None
    '''
    Since, activations of a sample from different layers are in the different element of the all_fmaps list
    we select an images (outer loop), and go through all the list (inner loop) to collect the activations
    of the given sample and call the normalization function.
    
    This function implements the l2 normalization step coming from the original VEBI paper:
    https://arxiv.org/abs/1712.06302
    '''

    for sample_indx in range(samples_num):
        l2_norm_temp_sample = None
        for i in range(len(all_fmaps)):
            featuremaps = torch.unsqueeze(all_fmaps[i][sample_indx], dim=0)
            fmap_squared = torch.pow(featuremaps, 2)
            fmap_squared_sum_perfiltr = torch.sum(fmap_squared, (2, 3))
            fmap_sum_allfilter = torch.sum(fmap_squared)
            fmap_normalize = torch.div(fmap_squared_sum_perfiltr, fmap_sum_allfilter).detach().cpu().numpy()

            if l2_norm_temp_sample is None:
                l2_norm_temp_sample = fmap_normalize
            else:
                l2_norm_temp_sample = np.concatenate((l2_norm_temp_sample, fmap_normalize), axis=1)
        if l2_norm is None:
            l2_norm = l2_norm_temp_sample
        else:
            l2_norm = np.concatenate((l2_norm, l2_norm_temp_sample), axis=0)
    return l2_norm
