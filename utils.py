''' Utils for Gym Recorder'''

def shortlist_operations(operations, require=[], exclude=[], shape_includes=None):
    '''Shortlist Tensorflow operations based on strings that are required in or 
    excluded from their names and the shape of their first output tensor.'''
    
    for req in require:
        req_shortlist = [op for op in operations if req in op.name]
    else:
        req_shortlist = operations
    for exc in exclude:
        exc_shortlist = [op for op in operations if exc not in op.name]
    else:
        exc_shortlist = operations
    name_shortlist = list(set(req_shortlist) & set(exc_shortlist))
    
    if shape_includes:
        total_shortlist = []
        shapes = []
        for op in name_shortlist:
            try:
                shape = op.outputs[0].shape.as_list()  # shape of the first output tensor
                if shape_includes in shape:
                    total_shortlist.append(op)
                    shapes.append(shape)
            except ValueError:  # because as_list() is not defined on an unknown TensorShape
                pass
            except IndexError:  # because list index of outputs[0] out of range
                pass
        return total_shortlist, shapes
    
    return name_shortlist

def get_activations(session, operation_name, feed_dict):
    '''Evaluate an operation in a Tensorflow session to get its output tensor.'''
    # Use the first output tensor of the operation
    tensor = session.graph.get_operation_by_name(operation_name).outputs[0]
    activations = tensor.eval(session=session, 
                              feed_dict=feed_dict)
    return activations