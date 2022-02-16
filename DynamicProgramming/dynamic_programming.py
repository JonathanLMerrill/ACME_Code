# dynamic_programming.py
"""Dynamic Programming.
Jonathan Merrill
"""

import numpy as np
from matplotlib import pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    e_value = [0]  #start with the first expected value of 0
    for i in range(N-1):
        e_value.append(max(((N-i-1)/(N-i))*e_value[i] + 1/N,e_value[i]))  #add the fraction of (N-1)/N * the previous value in our array added to 1/N to the array if it is greater than the previous value
    return max(e_value),N - e_value.index(max(e_value)) #return the greatest expected value and the index (stopping point)



def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    stopping_percent = []
    max_prob = []
    N = []
    for i in range(3,M,1):
        stopping_percent.append(calc_stopping(i)[1]/i) #use our previous function to find the stopping percent at each N value
        max_prob.append(calc_stopping(i)[0]) #use our previous function to find the maximum probability at each M value
        N.append(i)
    
    #plot the functions for max probability and stopping percent
    plt.plot(N,max_prob,'k',label = "Maximum Probability")
    plt.plot(N,stopping_percent,'r', label = "Optimal Stopping Percent")
    plt.legend(loc = "best")
    plt.xlabel("N")
    plt.ylabel("Percent")
    plt.title("Stopping Percent and Expected Value")
    
    return stopping_percent[-1]


def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    w = np.arange(N+1)/N  #define our W array
    C = np.zeros((N+1,N+1))  #define an empty C matrix
    for i in range(N+1):
        for j in range(N+1):
            if i > j:
                C[i][j] = u(w[i] - w[j])  #update values of C for each i > j
    return C



def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    P = np.zeros((N+1,T+1))  #define a zero P matrix
    w = np.arange(N+1)/N  #define our w array
    A = np.zeros((N+1,T+1))  #define a zero A matrix
    
    #define our initial A,P matrix which are just the last column 
    for i in range(N+1):
        A[i][-1] = u(w[i])
        P[i][-1] = w[i]
    CV = np.zeros((N+1,N+1))  #define a zero CV matrix
    for t in range(T):  #for each t, define our CV^t
        for i in range(N+1):
            for j in range(N+1):
                if w[i] >= w[j]:
                    #define our CV matrix at each w[i] > w[j] for each t,i,j
                    CV[i][j] = u(w[i]-w[j]) + B*A[j][T-t]
        #use the found CV^t to define our A,P Matrices column by column 
        for i in range(N+1):
            A[i][T-t-1] = max(CV[i])
            P[i][T-t-1] = w[i] - w[np.argmax(CV[i])]
    #return A and P
    return A,P

