import numpy as np
import torch
import matplotlib.pyplot as plt


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


def generate_training_plots(train_acc, val_acc, train_loss, val_loss, path):
    assert len(train_acc) == len(val_acc) == len(train_loss) == len(val_loss)
    epochs = len(train_acc)

    x_axis = [x for x in range(epochs)]

    plt.plot(x_axis, train_acc, 'tab:blue', label='train accuracy')
    plt.plot(x_axis, val_acc, 'tab:orange', label='validation accuracy')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(path + 'accuracy.png')
    plt.clf()

    plt.plot(x_axis, train_loss, 'tab:blue', label='train loss')
    plt.plot(x_axis, val_loss, 'tab:orange', label='validation loss')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path + 'loss.png')
    plt.clf()

def normalize_numpy(batch):
    normalized_batch = np.empty_like(batch)

    for i in range(batch.shape[0]):
        A = batch[i]
        if np.max(A) == 0.0:
            normalized_batch[i] = A
        else:
            normalized_batch[i] = (A - np.min(A)) / (np.max(A) - np.min(A))

    return normalized_batch
