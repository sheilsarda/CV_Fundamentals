import numpy as np
from est_homography import est_homography

def warp_pts(X, Y, interior_pts):
    """ 
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts. 
        These coordinate describe where a point inside the goal will be warped 
        to inside the penn logo. For this assignment, you can keep these new 
        coordinates as float numbers.
        
    """
    
    # You should Complete est_homography first!
    H = est_homography(X, Y)
    
    ##### STUDENT CODE START #####
    interior_pts_ = np.hstack([interior_pts, np.ones([interior_pts.shape[0],1])])

    warped_pts = interior_pts_.dot(H.T)
    warped_pts = warped_pts[:,:2] / warped_pts[:,[2]]
    
    ##### STUDENT CODE END #####
    
    return warped_pts