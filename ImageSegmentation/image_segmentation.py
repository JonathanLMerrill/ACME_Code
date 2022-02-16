# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name>
<Class>
<Date>
"""

import numpy as np
import scipy
from imageio import imread
from matplotlib import pyplot as plt
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    D = np.zeros(A.shape)
    for i in range(len(A)):
        B = A.sum(axis=1)
        D[i][i] = B[i]
    L = D - A
    return L
# =============================================================================
#     #raise NotImplementedError("Problem 1 Incomplete")
# A = np.array([[0,1,0,0,1,1],
#      [1,0,1,0,1,0],
#      [0,1,0,1,0,0],
#      [0,0,1,0,1,1],
#      [1,1,0,1,0,0],
#      [1,0,0,1,0,0]])
# print(laplacian(A))
# print(scipy.sparse.csgraph.laplacian(A))
# =============================================================================
# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    C = 0
    AC = []
    L = laplacian(A)
    E = scipy.linalg.eigvals(L)
    for i in range(len(E)):
        AC.append(np.real(E[i]))
        if np.real(E[i]) < tol:
            C += 1
    AC.sort()
    return C, AC[1]

# =============================================================================
# A = np.array([[0,1,0,0,1,1],
#      [1,0,1,0,1,0],
#      [0,1,0,1,0,0],
#      [0,0,1,0,1,1],
#      [1,1,0,1,0,0],
#      [1,0,0,1,0,0]])
# print(connectivity(A))
# =============================================================================
    #raise NotImplementedError("Problem 2 Incomplete")


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        image = imread(filename)
        scaled = image/255
        self.scaled = scaled
        if self.scaled.ndim > 2:
            brightness = self.scaled.mean(axis=2)
        else: 
            brightness = self.scaled
        D = np.ravel(brightness)
        self.D_array = D
    
        #raise NotImplementedError("Problem 3 Incomplete")
    def y(self):
        print(self.scaled.shape[0:2])
    # Problem 3
    def show_original(self):
        """Display the original image."""
        if self.scaled.ndim > 2:
            plt.imshow(self.scaled)
        else:
            plt.imshow(self.scaled, cmap="gray")
        #raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        
        m,n = self.scaled.shape[0:2]
        A = sp.lil_matrix((m*n,m*n))
        D = np.array([0.0 for i in range(m*n)])
        for i in range((m*n)):
            J, X_norm = get_neighbors(i, r, m, n)
            B_abs = np.abs(self.D_array[i] -self.D_array[J])
            W = np.exp((-B_abs/sigma_B2)-(X_norm/sigma_X2))
            A[i,J] = W
            D[i] = np.sum(W)
        A = sp.csc_matrix(A)
        return A,D
        #raise NotImplementedError("Problem 4 Incomplete")

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        m,n = self.scaled.shape[0:2]
        L = sp.csgraph.laplacian(A)
        D_one_half = sp.diags(1/np.sqrt(D))
        DLD = D_one_half@L@D_one_half
        E_vecs = sp.linalg.eigsh(DLD, which = "SM", k=2)[1]
        E_vecs = E_vecs[:,1].reshape((m,n))
        mask = E_vecs > 0
        return mask
        
        #raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r,sigma_B,sigma_X)
        mask = self.cut(A,D)
        if self.scaled.ndim > 2:
            mask = np.dstack((mask,mask,mask))
        pos_mask = ~mask
        mask = self.scaled*mask
        pos_mask = self.scaled*pos_mask
        
        A1 = plt.subplot(131)
        if self.scaled.ndim > 2:
            A1.imshow(self.scaled)
        else:
            A1.imshow(self.scaled, cmap="gray")
            
        A2 = plt.subplot(132)
        if pos_mask.ndim > 2:
            A2.imshow(pos_mask)
        else:
            A2.imshow(pos_mask, cmap="gray")
            
        A3 = plt.subplot(133)
        if mask.ndim > 2:
            A3.imshow(mask)
        else:
            A3.imshow(mask, cmap="gray")
        
        plt.show()
        
        
        #raise NotImplementedError("Problem 6 Incomplete")


# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
