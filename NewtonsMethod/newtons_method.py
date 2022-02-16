# newtons_method.py
"""Volume 1: Newton's Method.
<Name>
<Class>
<Date>
"""

import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from autograd import jacobian

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=20, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    i = 0
    if np.isscalar(x0) == True:
        x = x0 - alpha*(f(x0)/Df(x0))  #implement function 9.4 (backtracking) with the alpha variable so that we can take fractions of steps to hit every point
        converge = False  #set the converge variable to false initially
        while i < maxiter and abs(x-x0) >= tol:  #run through the loop until the number of iterations is fun or the level of tolerence we desire is reached
            x0 = x   #update x0
            x = x0 - alpha*(f(x0)/Df(x0))    #update x
            i += 1                          #add 1 to the number of iterations
        if i < maxiter:
            converge = True   #if the function took less than the max iterations, the function converged
        return x,converge,i    #return the last x value, whether it converged, and the number of iterations
    
    else:
        x = x0 - alpha*(la.solve(Df(x0),f(x0)))
        converge = False
        while i < maxiter and la.norm(x-x0) >= tol:
            x0 = x
            x = x0 - alpha*(la.solve(Df(x0),f(x0)))
            i += 1
        if i < maxiter:
            converge = True
        return x, converge, i
        
    #raise NotImplementedError("Problem 1 Incomplete")
# =============================================================================
# f = lambda x: x**2 - 3
# df = lambda x: 2*x
# x0 = 16
# print(newton(f,x0,df))
# print(np.sqrt(3))
# =============================================================================

# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    r = sy.symbols('r')     
    f = P2*(1-((1+r)**(-N2))) - (P1*((1+r)**(N1) - 1))  #define the function
    df = sy.diff(f,r)     #take the derivative
    df = sy.lambdify(r,df)     #lambdify  f and df
    f = sy.lambdify(r,f)       
    return newton(f,.1, df)[0]   #use function from prob 1 to return the apporximate zeros of the function with initial starting point of .1
    
    #raise NotImplementedError("Problem 2 Incomplete")


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    a = np.linspace(.001,1.01,100)  #define our different values of a from 0 to 1
    iterations = []
    for i in range(100):      #make a list of the number of iterations to find the zeros given the different a values
        iterations.append(newton(f,x0,Df,tol,maxiter,a[i])[2])
    plt.plot(a,iterations)  #plot the a values against the number of interations
    plt.xlabel("alpha values")
    plt.ylabel("Number of Iterations")
    plt.show()
    return a[np.argmin(iterations)]  #return the lowest number of iterations
    
    #raise NotImplementedError("Problem 4 Incomplete")
# =============================================================================
# x = sy.symbols('x')
# f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
# df = lambda x: np.sign(x) * (1./3.) * np.power(np.abs(x), -2./3.)
# x0 = .01
# print(optimal_alpha(f, x0, df))
# =============================================================================


# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    alpha = [1,0.55]
    f1 = lambda x,y: 5*x*y - x*(1+y)  #define the array of f function
    f2 = lambda x,y: -x*y + (1-y)*(1+y)
    f = lambda x: np.array([f1(x[0],x[1]),f2(x[0],x[1])])  
    Df1 = lambda x,y: 5*y - (1+y)  #take the partial derivatives to find the Jacobian
    Df2 = lambda x,y: 5*x - x
    Df3 = lambda x,y: -y
    Df4 = lambda x,y: -x - 2*y
    J = lambda x: np.array([[Df1(x[0],x[1]),Df2(x[0],x[1])],[Df3(x[0],x[1]),Df4(x[0],x[1])]]) #define the Jacobian as the matrix of those derivatives
    x = np.linspace(-.25,0,100)
    y = np.linspace(0,.25,100)
    for i in x:
        for j in y:
            x0 = np.array([i,j])  #at each value check if both conditions are satisfied and if they are return the x0 where they are
            if (np.allclose(newton(f,x0,J,alpha = alpha[0])[0],np.array([0,1])) or np.allclose(newton(f,x0,J,alpha = alpha[0])[0],np.array([1,0]))) and np.allclose(newton(f,x0,J,alpha = alpha[1])[0],np.array([3.75,.25])):
                return x0
    
    #raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    x_real = np.linspace(domain[0], domain[1], res) # Real parts.
    x_imag = np.linspace(domain[2], domain[3], res) # Imaginary parts.
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j*X_imag    #define our initial x0
    for i in range(iters):   #cycling through newtons method iters amount of times
        X_1 = X_0 - f(X_0)/Df(X_0)  #update the x0 and x1 iters numbers of times to find the zeros of the function
        X_0 = X_1
    Y = np.zeros((res,res))  #define an array of zeros
    for i in range(res):     #update the empty array at each value to show which zero the function is converging to
        for j in range(res):
            Y[i][j] = np.argmin(abs(X_1[i][j] - zeros))  #minus the zeros from each value in x1 and take the smallest one, this gives us the zero that value in the function converges to
    
    plt.pcolormesh(X_real,X_imag,Y,cmap = 'brg')  #plot these values on a colormesh brg scale 
    plt.title("Color map of Zeros of a function")
    plt.show()
    #raise NotImplementedError("Problem 7 Incomplete")


def test7():
    f = lambda x: x**3 - 1
    Df = lambda x: 3*x**2
    zeros = [1,-.5+ 1j*np.sqrt(3)/2,-.5 - 1j*np.sqrt(3)/2]
    domain = [-1.5,1.5,-1.5,1.5]
    plot_basins(f, Df, zeros, domain)






