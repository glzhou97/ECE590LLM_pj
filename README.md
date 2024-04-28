# ECE590LLM_pj
Used for ECE590 project

# Initial idea
Project plan:
Build softmax matrix and the approximated matrix using two ideas, prove they are similar 
Idea 1: Gaussian kernel approximation - this one is supposed to be unstable due to negative values 
Idea 2: Positive random features - this one is supposed to be more stable 
Idea 3 (if we have time): orthogonal random features - supposed to be even better 
Vary the number of random feature sampled and compare efficiency & accuracy 
Put the attention approximation in an actual model and compare performance
Maybe try different kinds of models? 

# Experiments
Based on jax implementation, experiment 1 will first compare the approximated matrix with the groud truth matrix. How the loss will change vs. different kernel settings

Experiment 2: Evaluate on real dataset, e.g. /google-research/protein_lm/