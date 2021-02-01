import numpy as np

def line_from_pts(p1, p2):
    """ 
    Compute a line from two points

    Input:
        p1: array of size (3,) representing point one
        p2: array of size (3,) representing point two
        
    Returns:
        l: array of size (3,) representing a line in homogenous 
        coordinate that go through point one and point two

    """

    ##### STUDENT CODE START #####
    l =  np.cross(p1, p2)
    l = l / np.linalg.norm(l[:2], 2)

    ##### STUDENT CODE END #####
    
    return l