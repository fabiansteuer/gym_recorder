'''Perturb tensors to see how this impacts a model's predictions.'''

# TODO Move to highlighter.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def get_mask(center, shape, radius=1, sigma=3):
    '''Get a 2D Gaussian mask around center in a tensor of size shape.'''
    y,x = np.ogrid[-center[0]:shape[0]-center[0], -center[1]:shape[1]-center[1]]
    keep = x*x + y*y <= radius
    mask = np.zeros(shape)
    mask[keep] = 1  # keep a circle with radius radius around center
    mask = gaussian_filter(mask, sigma)  # blur the circle to create a smooth mask
    return mask / mask.max()

# Pertubation functions
perturber_mapping = {}

def register_perturber(name):
    def registerer(func):
        perturber_mapping[name] = func
    return registerer

@register_perturber('blurlight')
def blurlight(image, mask, sigma=4):
    '''Blur a small region of image.'''
    return image*(1-mask) + gaussian_filter(image, sigma)*mask


@register_perturber('searchlight')
def searchlight(image, mask, sigma=4):
    '''Blur everything but a small region of image.'''
    return image*mask + gaussian_filter(image, sigma)*(1-mask)


def perturb(tensor, perturber='blurlight', step_size=1):
    '''Perturb tensor using perturber function with a step size of step_size.'''

    perturber = perturber_mapping[perturber]
    
    shape = tensor.shape
    assert len(shape) == 4, (f'Tensor to perturb should be of shape [batch_size, x, y, n_channels] '
                             f'but has shape {shape}.')
    
    centers = []
    perturbed = []
    for i in range(0, shape[1], step_size):
        for j in range(0, shape[2], step_size):
            mask = get_mask(center=[i,j], shape=shape[1:3])[:,:,np.newaxis]
            perturbed.append(perturber(tensor, mask))
            centers.append((i,j))

    assert perturbed[0].shape == shape, (f'Perturbed tensor (shape={perturbed[0].shape} has a '
                                         f'different shape than the original tensor (shape={shape}).')

    return perturbed, centers