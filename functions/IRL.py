import numpy as np
import cvxpy as cp

def IRL_OPT(
        pi: np.ndarray,
        d: np.ndarray,
) -> np.ndarray:
    """
    Take in input a deterministic policy pi and its state-only occupancy
    measure, and return the reward centroid of the OPT feasible set.
    """
    n = pi.shape[0]

    # initialize the reward
    r = np.ones((n,n,5)) / 5

    for i in range(n):
        for j in range(n):
            if d[i,j] > 0:
                r[i,j,:] = 0
                r[i,j,int(pi[i,j])] = 1

    return r

def IRL_MCE(
        pi: np.ndarray,
        d: np.ndarray,
        pimin: float = 1e-4,
) -> np.ndarray:
    """
    Take in input a stochastic policy pi, its state-only occupancy measure and
    the hyperparameter pimin, and return the reward centroid of the MCE feasible
    set.
    """
    n = pi.shape[0]

    # initialize the reward
    r = np.zeros_like(pi)+pimin

    for i in range(n):
        for j in range(n):
            if d[i,j] > 0:
                for a in range(5):
                    r[i,j,a] = np.maximum(pi[i,j,a],pimin)

    return np.log(r)

def IRL_BIRL(
        pi: np.ndarray,
        d: np.ndarray,
        pimin: float = 1e-4,
) -> np.ndarray:
    """
    Take in input a stochastic policy pi, its state-only occupancy measure and
    the hyperparameter pimin, and return the reward centroid of the BIRL feasible
    set.
    """
    n = pi.shape[0]

    # initialize the reward
    r = np.zeros_like(pi)+pimin

    for i in range(n):
        for j in range(n):
            if d[i,j] > 0:
                for a in range(5):
                    r[i,j,a] = np.maximum(pi[i,j,a],pimin)
                r[i,j,:] = r[i,j,:] / np.max(r[i,j,:])

    return np.log(r)

def best_case_OPT(
        pi: np.ndarray,
        d: np.ndarray,
        gamma: float,
        p: np.ndarray,
        k_V: float = 1,
        k_Adv: float = 0.5,
) -> tuple:
    """
    Compute an arbitrary reward in the OPT feasible set by first sampling
    uniformly the optimal actions in the states outside the support SE, and then
    by sampling uniformly the value function and advantage function of pi.
    """
    n = pi.shape[0]

    # initialize reward
    r = np.zeros((n,n,5))

    # sample V uniformly
    V = np.random.uniform(low=-k_V, high=+k_V, size=(n,n))

    # sample Adv uniformly
    Adv = np.random.uniform(low=0, high=+k_Adv, size=(n,n,5))

    # sample the optimal action in every state (the actions sampled in the
    # support of the expert will be discarded)
    opt = np.random.choice(5, size=(n,n))

    # construct the reward
    for i in range(n):
        for j in range(n):
            # if state visited by the expert, make expert's action optimal
            if d[i,j] > 0:
                opt[i,j] = pi[i,j]

            # set opt action optimal
            Adv[i,j,opt[i,j]] = 0

            for a in range(5):
                EV = np.sum(p[i,j,a,:,:]*V[:,:])
                r[i,j,a] = V[i,j]-gamma*EV - Adv[i,j,a]

    return r, opt

def best_case_MCE(
        pi: np.ndarray,
        d: np.ndarray,
        gamma: float,
        p: np.ndarray,
        lam: float,
        k_V: float = 1,
) -> np.ndarray:
    """
    Compute an arbitrary reward in the MCE feasible set by sampling uniformly
    the value function of pi. It requires a policy pi defined on the whole state
    space S.
    """
    if lam == 0:
        return best_case_OPT(pi,d,gamma,p)

    n = pi.shape[0]

    # sample V uniformly
    V = np.random.uniform(low=-k_V, high=+k_V, size=(n,n))

    # construct reward
    EV = np.sum(p*V, axis=(3,4))
    r = -gamma*EV + V[:,:,np.newaxis] + lam*np.log(pi)

    return r

def best_case_BIRL(
        pi: np.ndarray,
        d: np.ndarray,
        gamma: float,
        p: np.ndarray,
        beta: float,
        k_V: float = 1,
) -> np.ndarray:
    """
    Compute an arbitrary reward in the BIRL feasible set by sampling uniformly
    the value function of pi. It requires a policy pi defined on the whole state
    space S.
    """
    if beta == 0:
        return best_case_OPT(pi,d,gamma,p)

    n = pi.shape[0]

    # sample V uniformly
    V = np.random.uniform(low=-k_V, high=+k_V, size=(n,n))

    # construct reward
    EV = np.sum(p*V, axis=(3,4))
    ratio = pi / (np.max(pi,axis=2)[:,:,np.newaxis])
    r = -gamma*EV + V[:,:,np.newaxis] + beta*np.log(ratio)

    return r

def pi_BC(
        pi: np.ndarray,
        d: np.ndarray,
) -> np.ndarray:
    """
    Return a policy that prescribes the same actions as the given policy in the
    support of d, and 1/A outside.
    """
    n = pi.shape[0]

    det = True
    if len(pi.shape) == 3:
        det = False

    piBC = np.ones((n,n,5))/5

    for i in range(n):
        for j in range(n):
            if d[i,j] > 0:
                if det:
                    piBC[i,j,:] = 0
                    piBC[i,j,int(pi[i,j])] = 1
                else:
                    piBC[i,j,:] = pi[i,j,:]

    return piBC

def match_state_action_occupancy(
        gamma: float,
        s0x: int,
        s0y: int,
        p: np.ndarray,
        d_bar: np.ndarray,
        C: tuple = None,
        verbose: bool = False):
    """
    Find the feasible state-action occupancy measure in an MDP\R M' with initial
    state (s0x,s0y), transition model p, discount factor gamma, constraints C,
    that is closest to the input state-action occupancy measure d_bar.
    Return also the corresponding policy.
    """
    n = p.shape[0]
    A = 5

    # initialize result occupancy measure
    d = cp.Variable((n, n, A))

    # create variables for flow constraints
    inflow = gamma * cp.sum(cp.multiply(p, d[:, :, :, None, None]), axis=(0, 1, 2))
    outflow = cp.sum(d, axis=2)
    rhs = np.zeros((n, n))
    rhs[s0x, s0y] = 1 - gamma

    # Bellman flow constraint (elementwise)
    flow_constraint = (outflow == rhs + inflow)
    
    # total occupancy constraint
    total_constraint = cp.sum(d) == 1

    # non-negativity constraint
    non_negativity = d >= 0

    # objective: minimize ||d - d_bar||_1
    objective = cp.Minimize(cp.norm1(d - d_bar))

    # create constraints list
    constraints = [flow_constraint, total_constraint, non_negativity]

    # add forbidden states (through constraints C)
    if C is not None:
        constraints_C = d[C[0],C[1],:] == 0
        constraints.append(constraints_C)

    # solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND)

    # extract and normalize policy
    d_sa = d.value
    # avoid a negative occupancy
    min_d = np.min(d_sa)
    if min_d < 0:
        d_sa += -min_d
    d_s = np.sum(d_sa, axis=2, keepdims=True)
    # set pi uniform if not visited
    pi = np.divide(d_sa, d_s, out=np.zeros_like(d_sa)+0.2, where=(d_s != 0))

    return d_sa, d_s.squeeze(-1), pi
