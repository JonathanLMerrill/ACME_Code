# montecarlo_integration.py
"""Monte Carlo Integration.
Jonathan Merrill
"""

import numpy as np
import scipy.linalg as la
from scipy import stats
from matplotlib import pyplot as plt

def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    points = np.random.uniform(-1, 1, (n,N))  #define N points inside an n dimensional square 
    lengths = la.norm(points,axis = 0)    #find the distance from the origin of all the points
    num_within = np.count_nonzero(lengths < 1)  #find the points that are within distance 1 of the origin 
    return 2**n*(num_within/N)



def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    points = np.random.uniform(a,b,N)
    V = b-a  #define our Volume
    x = []
    for i in range(N):  #take f of each point
        x.append(f(points[i]))
    return (V/N)*np.sum(x)  #return the mean* the volume


def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    n = len(mins)
    points = np.random.uniform(0,1,(n,N)) #find N points in N dimensions from 0 to 1
    p = []
    for i in range(n):
        points[i] *= ((maxs[i]-mins[i])) #adjust the points by multiplying them by bi-ai
        points[i] += mins[i]      #add ai to the points
        p.append(maxs[i] - mins[i])
    V = np.prod(p)  #find the volumes (which is the product of bi - ai)
    return (V/N)*np.sum(f(points))  #return the means*volume


def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    #define the probability density function with its mins and maxs
    f = lambda x: 1/((2*np.pi)**2)*np.exp(-(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2)/2)
    mins = [-1.5, 0, 0, 0]
    maxs = [.75, 1, .5, 1]
    
    #find the integral using scipy
    means,cov = np.zeros(4),np.eye(4)
    integral = stats.mvn.mvnun(mins,maxs,means,cov)[0]
    
    
    N = np.logspace(1,5,20)
    f_hat = []
    for i in range(len(N)):  #take the integral for each N value
        f_hat.append(mc_integrate(f, mins, maxs,int(N[i])))
        
    #find the error between the actual integral and our problem 3 for varying points
    error = []
    for i in range(len(f_hat)):
        error.append(np.abs((integral - f_hat[i])/integral))
    
    #plot the error next to the 1/sqrt(n) which is approximately what the error should look like with increased N values
    plt.loglog(N,error,label = "Relative Error")
    plt.loglog(N,1/np.sqrt(N), label = "1/sqrt(N)")
    plt.title("Relative Error of Probability Density Function")
    plt.legend()


def test2():
    f = lambda x: x**2
    a = -4
    b = 2
    print(mc_integrate1d(f, a, b))
    g = lambda x: np.abs(np.sin(10*x)*np.cos(10*x) + np.sqrt(x)*np.sin(3*x))
    a = 1
    b = 5
    print(mc_integrate1d(g, a, b))
    h = lambda x: np.sin(x)
    a = -2*np.pi
    b = 2*np.pi
    print(mc_integrate1d(h, a, b))
    
def test3():
    f = lambda x: x[0]**2 + x[1]**2
    mins = [0,0]
    maxs = [1,1]
    print(mc_integrate(f, mins, maxs))
    g = lambda x: x[0] + x[1] - (x[3]*x[2]**2)
    mins = [-1,-2,-3,-4]
    maxs = [1,2,3,4]
    print(mc_integrate(g, mins, maxs,1000000))
    h = lambda x: 3*x[0] - 4*x[1] + x[1]**2
    mins = [1,-2]
    maxs = [3,1]
    print(mc_integrate(h, mins, maxs))
    