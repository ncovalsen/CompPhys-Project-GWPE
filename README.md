# Accelerated Likelihood Evaluation for Gravitational Wave Parameter Estimation: Relative Binning

This is a github repository for the final project of PHY381c which aims to implement an accelerated likelihood evaluation technique for parameter estimation of long duration gravitational wave signals. The specific technique we will be implementing is called 'relative binning' or 'heterodyning'.  
Conventionally, a parameter estimation code involves thousands/millions of likelihood evaluations (inner products) for a single run. The heterodyning approach assumes that likelihood evaluations made at 2 close points in the parameter space will be correlated by a smoothly varying function thereby increasing efficiency.  
Our idea is to implement the same for a toy inference model confined to a subset of the original parameter list. We will quantify comparisons across baseline (conventional) and heterodyning (accelerated) approach. We will then try and apply classical methods to further improve the implementation.  

## Planned directories:
- data
- code  
- results

## Tentative workflow:  
- Write code for simplified parameter inference model
- Test code for inference of small subset of total parameters using real and/or fiducial gravitational wave data (time series or frequency series)
- Implement heterodyning technique (see references)
- Re-test code with same data as last test
- Compare run times, accuracy
- Add improvements to implementation if possible
- Re-test and re-compare results

## References:  
https://arxiv.org/pdf/1007.4820  
https://arxiv.org/pdf/1806.08792
