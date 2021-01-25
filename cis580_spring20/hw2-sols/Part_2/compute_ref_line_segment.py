import numpy as np

from line_from_pts import line_from_pts
from line_intersection import line_intersection

def compute_ref_line_segment(ref, ll, lr, ur, ul):
    """ 
    This function finds the end points of the line segment
    where the ref is located on the field. The results will
    be used to plot the virtual line onto the field.

    Input:
        ref: array of size (3,) representing a point of ref on the field
        ll:  array of size (3,) representing lower left point of rectangle in frame
        lr:  array of size (3,) representing lower right point of rectangle in frame
        ur:  array of size (3,) representing upper right point of rectangle in frame
        ul:  array of size (3,) representing upper left point of rectangle in frame
        
    Returns:
        vanishing_pt: array of size (3,) representing scene vanishing point 
        top_pt:       array of size (3,) representing top of ref's line segment
        bottom_pt:    array of size (3,) representing bottom of ref's line segment

    """

    ##### STUDENT CODE START #####
    top_line = line_from_pts(ul, ur)
    bottom_line = line_from_pts(ll, lr)
    
    left_line = line_from_pts(ll, ul)
    right_line = line_from_pts(lr, ur)
    
    vanishing_pt = line_intersection(left_line, right_line)
    ref_line = line_from_pts(ref, vanishing_pt)
    
    top_pt = line_intersection(ref_line, top_line)
    bottom_pt = line_intersection(ref_line, bottom_line)

    ##### STUDENT CODE END #####
    
    return vanishing_pt, top_pt, bottom_pt