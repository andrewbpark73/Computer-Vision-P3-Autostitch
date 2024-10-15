import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN
        
        
        # Corresponding points in homogeneous coordinates
        A[2*i] = [a_x, a_y, 1, 0, 0, 0, -b_x*a_x, -b_x*a_y, -b_x]
        A[2*i + 1] = [0, 0, 0, a_x, a_y, 1, -b_y*a_x, -b_y*a_y, -b_y]

        
        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN
    
    
    # The homography h is the eigenvector corresponding to the smallest singular value,
    # which is the last row of Vt
    h = Vt[-1]

    # Reshape the homography vector h into the 3x3 matrix H
    H = np.reshape(h, (3, 3))

    # Normalize H so that the last element is 1
    H = H / H[-1, -1]
    
    
    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslation) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN
    
    
    # Initialize variables
    best_inliers = []
    
    # nRANSAC iterations
    for i in range(nRANSAC):
        # Randomly select a minimal set of feature matches
        if m == eTranslate:
            # For translation, we only need 1 match
            sample = random.sample(matches, 1)
        elif m == eHomography:
            # For homography, we need 4 matches
            sample = random.sample(matches, 4)
        
        # Compute the transformation implied by these matches
        if m == eTranslate:
            # Calculate the tanslation between the points
            pt1 = f1[sample[0].queryIdx].pt
            pt2 = f2[sample[0].trainIdx].pt
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        elif m == eHomography:
            # Calculate the homography using the selected sample matches
            M = computeHomography(f1, f2, sample)
        
        # Count the number of inliers
        inliers = getInliers(f1, f2, matches, M, RANSACthresh)
        
        # Updata the best transformation if the current transformation is better
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    
    # Re-compute the best transformation using all inliers
    if len(best_inliers) > 0:
        if m == eTranslate:
            # Average the translation vectors of all inliers
            dx = np.mean([f2[matches[i].trainIdx].pt[0] - f1[matches[i].queryIdx].pt[0] for i in best_inliers])
            dy = np.mean([f2[matches[i].trainIdx].pt[1] - f1[matches[i].queryIdx].pt[1] for i in best_inliers])
            M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        elif m == eHomography:
            # Compute the homography using all inliers
            M = leastSquaresFit(f1, f2, matches, m, best_inliers)
    
    
    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN
        
        
        # Transform the feature in f1 using the transformation matrix M
        pt1 = np.array([f1[matches[i].queryIdx].pt[0], f1[matches[i].queryIdx].pt[1], 1])
        pt2 = np.array([f2[matches[i].trainIdx].pt[0], f2[matches[i].trainIdx].pt[1]])
        
        pt1_transformed = np.dot(M, pt1)
        # Convert from homogeneous coordinates to Euclidean coordinates
        pt1_transformed = pt1_transformed / pt1_transformed[2]
        
        # Calculate the Euclidean distance between the transformed point and the point in f2
        distance = np.linalg.norm(pt1_transformed[:2] - pt2)
        
        # Check if the distance is within the threshold
        if distance < RANSACthresh:
            inlier_indices.append(i)
        
        
        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN
            
            # Get the match from the inlier indices
            match = matches[inlier_indices[i]]
            # Add the difference in the x and y coordinates of the inlier matches
            u += f2[match.trainIdx].pt[0] - f1[match.queryIdx].pt[0]
            v += f2[match.trainIdx].pt[1] - f1[match.queryIdx].pt[1]
            
            
            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN
        
        
        # Use the inlier matches to compute the homography
        inlier_matches = [matches[i] for i in inlier_indices]
        # Compute the homography
        M = computeHomography(f1, f2, inlier_matches)
        
        
        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

