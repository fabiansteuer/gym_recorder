'''Get saliency maps.'''

# TODO Refactor functions below

def get_saliency(observation,  # TODO un-hardcode
                 session=session,
                 operation_name='deepq/q_func/action_value/fully_connected_1/MatMul',
                 step_size=1
                ):
    
    def euclidian_distance(a ,b):
        return np.sqrt(np.sum((a-b)**2, axis=1))
    
    def get_q_values(observation):  # TODO un-hardcode
        feed_dict = {
            input1: observation,
            input2: observation}
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
    distances /+ distances.max()
    
    saliency = np.zeros([84,84], dtype=np.float64)  # TODO np.float64 likely an overkill
    for dist, center in zip(distances,centers):
        saliency[center] = dist
        
    # Scale to interval [0,1]  # TODO Should scaling rather take the whole batch of saliencies into account?
    scaled_saliency = saliency - saliency.min()
    scaled_saliency = scaled_saliency / scaled_saliency.max()
    
    return scaled_saliency

def upscale(image, shape):
    '''Rescale a 2D image to shape.'''
    scale = (shape[0]/image.shape[0], shape[1]/image.shape[1])
    rescaled = rescale(saliency, scale=scale,
                       mode='reflect', multichannel=False, anti_aliasing=True)
    return rescaled

def overlay(image, saliency, channels=[2], opacity=1.0, mode='clipping'):
    
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

    # Get weighted mean between image and saliency for channels
    for i in range(image.shape[-1]):
        if mode == 'weighted_mean':
            image[:,:,channels] = (1-opacity)*image[:,:,channel] + opacity*saliency[:,:,np.newaxis]
        elif mode == 'clipping':
            image[:,:,channels] += opacity * saliency[:,:,np.newaxis]
            image = image.clip(0,1)

    print(image.max())
    return image

def upscale(image, shape):
    '''Rescale a 2D image to shape.'''
    scale = (shape[0]/image.shape[0], shape[1]/image.shape[1])
    rescaled = rescale(saliency, scale=scale,
                       mode='reflect', multichannel=False, anti_aliasing=True)
    return rescaled

def overlay(image, saliency, channels=[2], opacity=1.0, mode='clipping'):
    
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

    # Get weighted mean between image and saliency for channels
    for i in range(image.shape[-1]):
        if mode == 'weighted_mean':
            image[:,:,channels] = (1-opacity)*image[:,:,channel] + opacity*saliency[:,:,np.newaxis]
        elif mode == 'clipping':
            image[:,:,channels] += opacity * saliency[:,:,np.newaxis]
            image = image.clip(0,1)

    return image