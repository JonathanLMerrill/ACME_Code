#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Volume 1: Facial Recognition.</h1>
# 
#     <Name> (double-click to edit)
#     <Class>
#     <Date>

# In[1]:


import os
import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
import random
from scipy import linalg


# ### Helper Functions

# In[2]:


def get_faces(path="./faces94"):
    """Traverse the specified directory to obtain one image per subdirectory. 
    Flatten and convert each image to grayscale.
    
    Parameters:
        path (str): The directory containing the dataset of images.  
    
    Returns:
        ((mn,k) ndarray) An array containing one column vector per
            subdirectory. k is the number of people, and each original
            image is mxn.
    """
    # Traverse the directory and get one image per subdirectory.
    faces = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for fname in filenames:
            if fname[-3:]=="jpg":       # Only get jpg images.
                # Load the image, convert it to grayscale,
                # and flatten it into a vector.
                faces.append(np.ravel(imread(dirpath+"/"+fname, as_gray=True)))
                break
    # Put all the face vectors column-wise into a matrix.
    return np.transpose(faces)


def sample_faces(k, path="./faces94"):
    """Generate k sample images from the given path.

    Parameters:
        n (int): The number of sample images to obtain. 
        path(str): The directory containing the dataset of images.  
    
    Yields:
        ((mn,) ndarray): An flattend mn-array representing a single
        image. k images are yielded in total.
    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for fname in filenames:
            if fname[-3:]=="jpg":       # Only get jpg images.
                files.append(dirpath+"/"+fname)

    # Get a subset of the image names and yield the images one at a time.
    test_files = np.random.choice(files, k, replace=False)
    for fname in test_files:
        yield np.ravel(imread(fname, as_gray=True))


# ### Problem 1 
# 
# - Implement `show()`.
# - Use `show()` to display a face from the `faces94` dataset.

# In[3]:

#Problem 1?
def show(image, m=200, n=180):
    """Plot the flattened grayscale 'image' of width 'w' and height 'h'.
    
    Parameters:
        image ((mn,) ndarray): A flattened image.
        m (int): The original number of rows in the image.
        n (int): The original number of columns in the image.
    """
    picture = np.reshape(image,(m,n))
    plt.imshow(picture, cmap = "gray")
    plt.show()

""" uncommenting these lines below will print out a random one of the 153 
images in black and white (grayscale), to make sure the get_faces() function
pulled the right images from ./faces94 I moved the faces 94 into my facial
recognition folder. This funciton and these commented lines below worked for
me to display the images """    
#i = random.randint(0,152)
#image = get_faces()[:,i]
#show(image)

    #raise NotImplementedError("Problem 1 Incomplete")


# In[ ]:

# =============================================================================
# A = np.identity(6)
# A[0,3] = 7
# B = A.copy()
# B = B.transpose()
# B += [0,1,2,3,4,5]
# B = B.transpose()
# print(B)
# =============================================================================
#A = [7,9,2,3,4,5,6,7]
#print(np.argmin(A))

# In[ ]:


class FacialRec(object):
    """Class for storing a database of face images, with methods for
    matching other faces to the database.
    
    Attributes:
        F ((mn,k) ndarray): The flatten images of the dataset, where
            k is the number of people, and each original image is mxn.
        mu ((mn,) ndarray): The mean of all flatten images.
        Fbar ((mn,k) ndarray): The images shifted by the mean.
        U ((mn,k) ndarray): The U in the compact SVD of Fbar;
            the columns are the eigenfaces.
    """
    # Problems 2-3
    def __init__(self, path='./faces94'):
        """Initialize the F, mu, Fbar, and U attributes.
        This is the main part of the computation.
        """
        self.F = get_faces(path)
        print(1)
        self.mu = np.mean(self.F,axis = 1)
        print(2)
        self.Fbar = self.F.copy()
        print(3)
        self.Fbar = self.Fbar.transpose()
        print(4)
        self.Fbar -= self.mu
        print(5)
        self.Fbar = self.Fbar.transpose()
        print(6)
        A = np.reshape(self.Fbar[:,0],(200,180))
        print(A.shape)
        self.U,S,vh = linalg.svd(A)
        print(7)
        show(self.mu)
        i = random.randint(0,152)
        show(self.F[:,i])
        show(self.Fbar[:,i])
        
        #raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def project(self, A, s):
        """Project a face vector onto the subspace spanned by the first s
        eigenfaces, and represent that projection in terms of those eigenfaces.
        
        Parameters:
            A((mn,) or (mn,l) ndarray): The array to be projected. 
            s(int): the number of eigenfaces.
        Returns: 
            ((s,) ndarray): An array of the projected image of s eigenfaces.
        """
        return (self.U[:,:s]).T@A
        #raise NotImplementedError("Problem 3 Incomplete")

    # Problem 5
    def find_nearest(self, g, s=38):
        """Find the index j such that the jth column of F is the face that is
        closest to the face image 'g'.
        
        Parameters:
            g ((mn,) ndarray): A flattened face image.
            s (int): the number of eigenfaces to use in the projection.

        Returns:
            (int): the index of the column of F that is the best match to
                   the input face image 'g'.
        """
        gbar = g - self.mu
        F_hat = self.project(self.Fbar,s)
        g_hat = self.project(gbar,s)
        check = []
        for image in range(len(F_hat[0])):
            check.append(linalg.norm(F_hat[:,image] - g_hat))
        face = np.argmin(check)
        return face
    
        
        #raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def match(self, image, s=38, m=200, n=180):
        """Display an image along with its closest match from the dataset. 
        
        Parameters:
            image ((mn,) ndarray): A flattened face image.
            s (int): The number of eigenfaces to use in the projection.
            m (int): The original number of rows in the image.
            n (int): The original number of columns in the image.
        """
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        ax1.imshow(image.reshape((m,n)), cmap = "gray")
        ax1.axis("off")
        ax2.imshow(self.F[:,self.find_nearest(image,s)].reshape((m,n)), cmap="gray")
        ax2.axis("off")
        ax1.set_title("Original Image")
        ax2.set_title("Closest Match")
        #raise NotImplementedError("Problem 6 Incomplete")


# ### Problem 2
# 
# - In `FacialRec.__init__()`, compute $F$, the mean face $\boldsymbol{\mu}$, and the mean-shifted faces $\bar{F}$.
# Store each as an attribute.
# 
# - Initialize a `FacialRec` object and display its mean face, plus an original image and its shifted face.

# In[ ]:





# In[ ]:





# ### Problem 3
# 
# - In `FacialRec.__init__()`, compute the compact SVD of $\bar{F}$ and store the $U$ as an attribute.
# - Use `show()` to display some of the eigenfaces (the columns of $U$).
# - Implement `FacialRec.project()` (in a single line).
# 

""" I'm not sure if I was supposed to include these in the code..... I 
commented this out just in case I wasn't supposed to. Hopefully you'll see 
this. I tested these and they worked. """
#face = FacialRec()
#show(face.U[:,50])
#show(face.U[:,1])
#show(face.U[:,100])

# In[ ]:





# ### Problem 4
# 
# - Select one of the shifted images $\bar{\mathbf{f}}_i$.
# - For at least 4 values of $s$, use `FacialRec.project()` to compute the corresponding $s$-projection $\widehat{\mathbf{f}}_i$, then compute the reconstruction $\widetilde{\mathbf{f}}_i$.
# - Display each of the reconstructions and the original image.

# In[ ]:
""" I'm not sure if I was supposed to include these in the code..... I 
commented this out just in case I wasn't supposed to. Hopefully you'll see 
this. I tested these and they worked. """
#face = FacialRec()
#for i in [1,20,40,80]:
#    U_S = face.project(face.Fbar[:,50],i)
#    show(face.U[:,:i]@U_S + face.mu)
#show(face.F[:,50])





# ### Problem 5 
# Implement `FacialRec.find_nearest()`.

# ### Problem 6
# 
# - Implement `FacialRec.match()`.
# - Generate some random faces with `sample_faces()`, and use `FacialRec.match()` to find the closest match (let $s=38$).
# - For each test, display the sample image and its closest match.

# In[ ]:

image = FacialRec()


