# qr_decomposition.py
"""The QR Decomposition.
Jonathan Merrill
"""
import numpy as np
import scipy
from scipy import linalg as la

def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m,n = A.shape    #store the dimenstion of A
    Q = A.copy()       #make a copy of A
    R = np.zeros((n,n))
    for i in range(0,n):
        R[i,i] = scipy.linalg.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i,i]
        #scipy.linalg.norm(Q[:,i])     #normalize the columns of Q
        for j in range(i+1,n):
            R[i,j] = np.dot(np.transpose(Q[:,j]),Q[:,i])
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]  #Orthogonalize jth column of Q
    return Q,R


def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    Q,R = la.qr(A, mode = "economic")
    return abs(np.prod(np.diag(R)))
    

def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    Q,R = la.qr(A, mode = "economic")
    y = np.dot(np.transpose(Q),b)
    return la.solve(R,y)


def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    m,n = A.shape
    R = A.copy()
    Q = np.identity(m)
    for k in range(0,n):
        u = R[k:,k].copy()
        sign = lambda x: 1 if x >= 0 else -1
        u[0] = u[0] + sign(u[0])*scipy.linalg.norm(u)
        u = u/scipy.linalg.norm(u)
        
        R[k:,k:] = R[k:,k:] - 2*np.outer(u, (np.dot(np.transpose(u),R[k:,k:])))
        Q[k:,:] = Q[k:,:] - 2*np.outer(u, (np.dot(np.transpose(u),Q[k:,:])))
        return np.transpose(Q), R
    

def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    m,n = A.shape
    H = A.copy()
    Q = np.identity(m)
    for k in range(0,n-2):
        u = np.copy(H[k+1:,k])
        sign = lambda x: 1 if x >= 0 else -1
        u[0] = u[0] + sign(u[0])*scipy.linalg.norm(u)
        u = u/scipy.linalg.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u,(np.dot(np.transpose(u),H[k+1:,k:])))
        H[:,k+1:] = H[:,k+1:] - 2*np.outer(np.dot(H[:,k+1:],u),np.transpose(u))
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u,(np.dot(np.transpose(u),Q[k+1:,:])))
    return H, np.transpose(Q) 


def test_function():
    A = np.random.random((6,4))
    Q,R = qr_gram_schmidt(A)
    print(qr_gram_schmidt(A))
    print(la.qr(A, mode = "economic"))
    print(A.shape,Q.shape,R.shape)
    print(np.allclose(np.triu(R), R))
    print(np.allclose(Q.T @ Q, np.identity(4)))
    print(np.allclose(Q @ R, A))


    A = np.random.random((5, 3))
    Q,R = qr_householder(A) # Get the full QR decomposition.
    print(A.shape, Q.shape, R.shape)
    print(np.allclose(Q @ R, A))

    A = np.random.random((8,8))
    H, Q = hessenberg(A)
    print(np.allclose(np.triu(H, -1), H))
    print(np.allclose(Q @ H @ Q.T, A))
