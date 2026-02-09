# Generalizing Behavior via Inverse Reinforcement Learning with Closed-Form Reward Centroids

The code can be executed through the notebook ```main.ipynb```. This file relies
on the Python functions provided by the files contained into the
```functions/``` folder:

- ```sampling.py```: Provides the functions necessary for constructing the transition
  models and the expert's policies used in the experiments conducted in the
  paper.
  
- ```IRL.py```: Implements the various methods described in the paper for
  addressing the problem of generalizing the behavior of the expert.

- ```planning.py```: Provides the planning subroutines necessary for computing
  the generalizing policies in the new environments from the given rewards.

- ```plotting.py```: Provides the necessary plotting methods.

## Requirements:

The code has been written and executed in Python 3.11.9 with these packages:

- numpy 2.2.4
- cvxpy 1.6.5
- matplotlib 3.10.1