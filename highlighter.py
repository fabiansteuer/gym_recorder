'''Get saliency maps.'''

import math
import numpy as np
from skimage.transform import rescale
from scipy.ndimage.filters import gaussian_filter

from .perturb import perturb
from .utils import get_activations

# TODO Refactor functions below

def get_saliency(observation, session, operation_name, feed_operations, step_size=1):
    
    def euclidian_distance(a ,b):
        return np.sqrt(np.sum((a-b)**2, axis=1))
    
    def get_q_values(observation):
        feed_dict = {op: observation for op in feed_operations}
        q_values = get_activations(
            session=session, 
            operation_name=operation_name,
            feed_dict=feed_dict)
        return q_values
    
    q_values = get_q_values(observation)
    
    perturbations, centers = perturb(observation, step_size=step_size)
    perturbations = np.squeeze(np.asarray(perturbations))
    
    new_q_values = get_q_values(perturbations)
    
    distances = euclidian_distance(new_q_values, q_values)
    distances /+ distances.max()  # TODO Division by 0
    
    saliency = np.zeros([84,84], dtype=np.float64)  # TODO np.float64 likely an overkill
    for dist, center in zip(distances,centers):
        saliency[center] = dist

    if step_size > 0:  # blur saliency
        saliency = gaussian_filter(saliency, sigma=math.sqrt(step_size))
        
    # Scale to interval [0,1]  # TODO Should scaling rather take the whole batch of saliencies into account?
    scaled_saliency = saliency - saliency.min()
    scaled_saliency = scaled_saliency / scaled_saliency.max()
    
    return scaled_saliency

def upscale(image, shape):
    '''Rescale a 2D image to shape.'''
    scale = (shape[0]/image.shape[0], shape[1]/image.shape[1])
    rescaled = rescale(image, scale=scale,
                       mode='reflect', multichannel=False, anti_aliasing=True)
    return rescaled

def overlay(image, saliency, channels=[2], opacity=3.0, mode='clipping'):
    
    # TODO Add assert that channels included in image
    
    # Saliency should have the same [x,y] dimensions as image
    if saliency.shape[0:2] != image.shape[0:2]:
        saliency = upscale(saliency, image.shape)
        
    # Scale saliency to interval [0,1]
    saliency -= saliency.min()
    saliency /= saliency.max()
    
    # Convert image integers in intervall [0,255] to floats in [0,1]
    image = image.astype(np.float64)  # TODO np.float64 likely an overkill
    image /= 255

    # Overlay TODO Doesn't work for white background
    if mode == 'weighted_mean':
        image[:,:,channels] = (1-opacity)*image[:,:,channels] + opacity*saliency[:,:,np.newaxis]
    elif mode == 'clipping':  
        image[:,:,channels] += opacity * saliency[:,:,np.newaxis]
        image = image.clip(0,1)

    return image
