# viscous-burgers-vae-rom

The goal of this work is to develop a generalized probabilistic reduced order model for viscous Burgers equation using VAEs.
- Develop reduced order model for a single initial condition using as little data as possible
- Generalize to a class of initial conditions
- Extract disentangled parameters of the dynamical system
- Create a probabilistic reduced order model generalized to a class of initial conditions and dynamical system parameters

## Intermediate Steps

- [x] First, begin with 1D and try to reconstruct single time snapshots rather than the entire spatio-temporal snapshot. 
- [ ] Obtain accurate prediction on single time snapshots past the training times
- [ ] Train a reduced order model on the latent space (try on single initial condition data first)
- [x] Reconstruct single time snapshots over a class of initial conditions
- [ ] Obtain accurate prediction on single time snapshots past training times (for a class of initial conditions)
- [ ] Train a reduced order model on the latent space (


- [ ] Extract dynamical system parameters from lower dimensional representation using hierarchical priors.
- [ ] Predict dynamical system from lower dimensional representation. 
