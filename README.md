# Homework 3

This is a team assignment. As a team you can work in parallel on different components. 

In this home assignment, you will implement the PPO algorithm, and apply 
it to solve the task of swinging up an inverted pendulum, which is a  continuous dynamical 
system with continuous state, s, and action, a, spaces. Don't use the existing PPO codebases
in your solution (avoid plagiarism). Design your own solution/architecture. 

The actions are torque to the pivot point of the pole. The actions are one-dimensional
random variables, distributed with the Gaussian probability density function with the 
mean mu(s) and standard deviation std(s). For simplicity, in the basic implementation, 
you can implement a state independent standard deviation, which will be a 
hyperparameter (scalar) which you tune manually. In the basic implementation "a ~ N(mu(s), std)",
cf., "a ~ N(mu(s), std(s))" in the general case.

You are provided with the required utilities, which will help you to implement 
the PPO algorithm, given by the pseudocode in the PPO paper.
Different software-engineering design choices are possible.

Feel free to refer to the code from homework 1 as an inspiration. 


## Guidelines

* Start by implementing an environment interaction loop. You may refer to homework 1 for inspiration. 

* Create and test an experience replay buffer with a random policy, which is the 
Gaussian distribution with arbitrary (randomly initialized) weights of the policy feed-forward network,
receiving state, s, and returning the mean, mu(s) and the log_std, log_stg(s) 
(natural logarithm of the standard deviation) of actions.  As mentioned above, you can use 
a state-independent standard variance.

* Make an episode reward processing function to turn one-step rewards into discounted rewards-to-go:
R(s_1) = sum_{t=1} gamma^{t-1} r_t, which is the discounted reward, starting from the state, s_1.

* Start the model by implementing a vanilla policy gradient agent, where the gradient ascent steps
are done with the average of the gradient of log-likelihood over a trajectory weight by rewards-to-go   
from each state. Try different step sizes in the gradient ascent.  

* Pendulum is a continuous action space environment. 
Check out the example in `Modules.py` for torch implementation of the Gaussian module.  (if you work in Julia, speak with me regarding the pendulum dynamics in Julia, and Flux for DNNs.)

* Add a feed-forward network for the critic, accepting the state, s=[sin(angle), cos(angle), angular velocity], and returning a scalar for the value of the state, s.

* Implement the generalized advantage, see Eq11-12 in the PPO, to be used instead of rewards-to-go.

* Implement the surrogate objective for the policy gradient, see Eq7, without and without clipping. 

* Implement the total loss, see Eq9 in the PPO.    

* Combine all together to Algorithm 1 in the PPO paper. (In your basic implementation, you can collect data with a single actor, N=1)

* You should see progress with default hyperparameters, but you can try tuning those to 
see how it will improve your results. 
 

## to extend HW3 to the final project

Do any three out of the following possible extensions:   
1. Use an ensemble of B critics instead of a single critic: Vens(s) = 1/B sum_b V_b(s), where each V_b(s) parametrized by a separate network.  
2. Add recurrence to the policy and train in the environment with partial observability. E.g., use only the angular velocity, rather than the full state, s =[angle, angular velocity]. 
3. Use images instead of state vectors. Note that since you won't have 
access to velocity, you will need either to stack a few last images or to use a recurrent policy.
4. Pre-train state representation, Z, from images using VAE, and then, utilize Z for training the policy. Compare 3 and 4 with regard to the total sampling complexity, total 
- your ideas. (share please with me before implementing). 


## What To Submit
- Your runnable code.
- PDF with the full solution, (e.g., export of a notebook to PDF), including the code and the plots with "Learning curves" and "Loss curves":
   Learning curves should show the accumulated discounted reward by a policy at the k-th training episode, for each setting of the algorithm/parameters, e.g., i) w/, w/o clipping, ii) w/, w/o generalized advantage, iii) w, w/o stacking images (if you choose to extend HW3 to the final project), iv) etc. 
  Loss, Eq9, for each setting of the algorithm. e.g., i) w/, w/o clipping, ii) w/, w/o generalized advantage, etc.
- visualize the 2D landscape of V(s) on the gird of states defined by theta=range(-pi, +pi,  100) , .theta=range(-8, +8, 100), where theta and .theta are the angle and the angular velocity respectively. use imshow(reshape(V(s)), (100, 100)). This landscape should have the maximum at the goal state (the top) and minimum at the bottom. Explain how the 2D landscape of V(s) guides the agent to move to the top. 


Please submit the files separately (PDF, py, ipynb) rather than in a single zip file. 
