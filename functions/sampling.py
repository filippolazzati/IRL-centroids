import numpy as np

def get_p(n: int):
    """
    This method takes in input the dimension of the grid-world n, and returns
    the transition model p described in the paper, where each arrow action
    brings to the cell pointed by it.
    """
    p = np.zeros((n,n,5,n,n))

    # 0 = right, 1 = left, 2 = top, 3 = bottom, 4 = stay
    for i in range(n):
        for j in range(n):
            if j < n-1:
                # move right
                p[i,j,0,i,j+1] = 1
            else:
                # stay here
                p[i,j,0,i,j] = 1
            
            if j > 0:
                # move left
                p[i,j,1,i,j-1] = 1
            else:
                # stay here
                p[i,j,1,i,j] = 1

            if i > 0:
                # move top
                p[i,j,2,i-1,j] = 1
            else:
                # stay here
                p[i,j,2,i,j] = 1

            if i < n-1:
                # move bottom
                p[i,j,3,i+1,j] = 1
            else:
                # stay here
                p[i,j,3,i,j] = 1

            p[i,j,4,i,j] = 1
    return p

def opposite_p(n: int):
    """
    This method takes in input the dimension of the grid-world n, and returns
    the transition model p' described in the paper, where each arrow action
    brings to the cell in the opposite direction w.r.t. its head.
    """
    p = np.zeros((n,n,5,n,n))

    # 1 = right, 0 = left, 3 = top, 2 = bottom, 4 = stay
    for i in range(n):
        for j in range(n):
            if j < n-1:
                # move right
                p[i,j,1,i,j+1] = 1
            else:
                # stay here
                p[i,j,1,i,j] = 1
            
            if j > 0:
                # move left
                p[i,j,0,i,j-1] = 1
            else:
                # stay here
                p[i,j,0,i,j] = 1

            if i > 0:
                # move top
                p[i,j,3,i-1,j] = 1
            else:
                # stay here
                p[i,j,3,i,j] = 1

            if i < n-1:
                # move bottom
                p[i,j,2,i+1,j] = 1
            else:
                # stay here
                p[i,j,2,i,j] = 1

            p[i,j,4,i,j] = 1
    return p

def get_piE_stoch(d, thr=1e-4):
    """
    Take in input a state-action occupancy measure d and clip it to values above
    the specified threshold. Then, re-normalize and return the resulting policy.
    
    This function is useful to construct a stochastic expert's policy whose
    support SE is a strictly smaller subset of the state space S.
    """
    n = d.shape[0]

    d_sa = np.copy(d)

    for i in range(n):
        for j in range(n):
            for a in range(5):
                if d_sa[i,j,a] < thr:
                    d_sa[i,j,a] = 0
    # normalize
    d_sa /= np.sum(d_sa)

    d_s = np.sum(d_sa, axis=2)

    pi = -np.ones((n,n,5))
    for i in range(n):
        for j in range(n):
            if d_s[i,j] != 0:
                pi[i,j,:] = d_sa[i,j,:] / d_s[i,j]

    return pi, d_s, d_sa