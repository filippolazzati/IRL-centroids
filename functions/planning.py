import numpy as np

def V_iteration(
        gamma: float,
        p: np.ndarray,
        r: np.ndarray,
        C: tuple = None,
        eps: float = 1e-5,
        T: int = 15,
        verbose: bool = False,
) -> tuple:
    """
    Perform V-iteration on the given (constrained) MDP until the difference in
    max norm between the V-functions of two iterations is bounded by eps. Note
    that we return both the Q-function, and the V-function. For managing
    constraints, we simply subtract -np.inf to the input reward.
    """
    n = p.shape[0]

    if C is not None:
        r = np.copy(r)
        r[C[0],C[1]] = -np.inf

    # initialize V and Q
    V = np.zeros((n,n))
    Q = np.zeros((n,n,5))

    # initialize counter
    counter = 0

    while True:
        # if p=0, then avoid multiplication, because V might be -np.inf, and
        # product becomes nan
        EV = np.multiply(p,V,out=np.zeros_like(p),where=p!=0)
        EV = np.sum(EV, axis=(3,4))

        Q = r + gamma * EV

        V_prime = np.max(Q, axis=2)
        
        counter += 1       

        if counter % T == 0:
            # avoid np.inf-np.inf using masks
            mask_V = V != -np.inf
            mask_V_prime = V_prime != -np.inf
            # if masks do not coincide, diff is infinite
            if not np.all(mask_V == mask_V_prime):
                delta_norm = np.inf
            # otherwise, compute difference based on other values
            else:
                delta_norm = np.max(np.abs(V_prime[mask_V] - V[mask_V]))

            if verbose:
                print('iteration: '+str(counter)+', norm difference: '+str(delta_norm))
            if delta_norm <= eps:
                break
        
        # update V
        V = V_prime

    return Q, V_prime, np.argmax(Q, axis=2)

def soft_V_iteration(
        gamma: float,
        p: np.ndarray,
        r: np.ndarray,
        lam: float,
        eps: float = 1e-5,
        T: int = 15,
        verbose: bool = False,
) -> tuple:
    """
    Perform soft V-iteration on the given MDP until the difference in max norm
    between the Q-functions of two iterations is bounded by eps. Note that we
    return both the soft Q-function, and the soft V-function.
    """
    # if lambda=0 return optimal policy
    if lam == 0:
        return V_iteration(gamma,p,r,None,eps,T,verbose)

    n = p.shape[0]

    # initialize V and Q
    V = np.zeros((n,n))
    Q = np.zeros((n,n,5))

    # initialize counter
    counter = 0

    while True:
        
        # vectorized soft Bellman update
        EV = np.sum(p*V, axis=(3,4))
        Q_prime = r + gamma * EV
        # subtract max for numerical stability in logsumexp
        M = np.max(Q, axis=2)
        if lam == 0:
            V = M
        else:
            exp = np.exp((Q_prime-M[:,:,np.newaxis])/lam)
            V = M + lam*np.log(np.sum(exp, axis=2))
        
        counter += 1

        if counter % T == 0:
            delta_norm = np.max(np.abs(Q-Q_prime))

            if verbose:
                print('iteration: '+str(counter)+', norm difference: '+str(delta_norm))
            if delta_norm <= eps:
                break
        
        # update Q
        Q = Q_prime

    # find MCE policy
    piMCE = np.exp((Q_prime - V[:,:,np.newaxis])/lam)

    return Q_prime, V, piMCE

def boltzmann_policy(
        gamma: float,
        p: np.ndarray,
        r: np.ndarray,
        beta: float,
        eps: float = 1e-5,
        T: int = 15,
        verbose: bool = False,
) -> np.ndarray:
    """
    Compute the Boltzmann-rational policy in the given MDP using an error
    tolerance eps.
    """
    
    # do planning
    Q, V, pi = V_iteration(gamma,p,r,None,eps,T,verbose)
    
    # if beta=0 then return optimal policy
    if beta == 0:
        return pi
    
    # compute BIRL policy from Q (subtract V for numerical stability)
    piBIRL = np.exp((Q - V[:,:, np.newaxis]) / beta)
    norm = np.sum(piBIRL, axis=2)
    rows, cols = np.where(norm)
    piBIRL[rows, cols, :] /= norm[rows, cols][:, np.newaxis]

    return piBIRL

def compute_occupancy(
        pi: np.ndarray,
        p: np.ndarray,
        gamma: float,
        s0x: int,
        s0y: int,
        max_iter: int = 10000,
        tol: float = 1e-8
) -> np.ndarray:
    """
    Compute the occupancy measure induced by the input policy pi in the input
    MDP\R. Terminate when max_iter iterations have performed, or an error
    smaller than tol is achieved.
    """
    n = pi.shape[0]

    # flag if pi is deterministic
    det = True
    if len(pi.shape) == 3:
        det = False
    
    # initial distribution
    d = np.zeros((n,n))
    d[s0x, s0y] = 1
    
    # state occupancy
    occ = np.zeros((n, n))

    for _ in range(max_iter):
        occ += d
        d_new = np.zeros_like(d)

        for a in range(5):
            # compute state-action occupancy measure
            if det:
                # distinguish deterministic policy
                mask = (pi == a).astype(float)  # shape (n, n)                
                d_sa = d * mask
            else:
                # from stochastic policy
                d_sa = d * pi[:,:,a]  # shape: (n, n, 5)
            
            # compute its contribution to the discounted probability of next stage
            d_new += gamma * np.tensordot(d_sa, p[:, :, a, :, :], axes=([0, 1], [0, 1]))

        if np.max(np.abs(d_new - d)) < tol:
            break
        d = d_new

    # multiply by the effective horizon
    occ *= 1-gamma

    # occupancy state-action
    if not det:
        occ_sa = occ[:, :, None] * pi
    else:
        occ_sa = np.zeros((n, n, 5))
        idx = pi.astype(int)  # action indices
        rows, cols = np.indices((n, n))
        occ_sa[rows, cols, idx] = occ

    return occ, occ_sa
