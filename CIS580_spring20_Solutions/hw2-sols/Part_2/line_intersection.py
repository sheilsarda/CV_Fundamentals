import numpy as np

def line_intersection(l1, l2):
    """ 
    Compute the intersection of two line

    Input:
        l1: array of size (3,) representing a line in homogenous coordinate 
        l2: array of size (3,) representing another line in homogenous coordinate 
        
    Returns:
        pt: array of size (3,) representing the intersection of two lines in 
        homogenous coordinate. Remeber to normalize it so its last coordinate
        is either 1 or 0.

    """

    ##### STUDENT CODE START #####
    pt = np.cross(l1, l2)
    if np.abs(pt[2]) > 1e-6:
        pt = pt / pt[2]

    ##### STUDENT CODE END #####
        
    return pt