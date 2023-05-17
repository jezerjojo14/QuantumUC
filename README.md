# Combining the QAOA and HHL Algorithm to achieve a Substantial Quantum Speedup for the Unit Commitment Problem

## Abstract

In this paper, we propose a quantum algorithm to solve the unit commitment (UC) problem at least cubically faster than existing classical approaches. This is accomplished by calculating the energy transmission costs using the HHL algorithm inside a QAOA routine. We verify our findings experimentally using quantum circuit simulators in a small case study. Further, we postulate the applicability of the concepts developed for this algorithm to be used for a large class of optimization problems that demand solving a linear system of equations in order to calculate the cost function for a given solution.

Paper Link:  [[ArXiv](https://arxiv.org/pdf/2305.08482.pdf)]

To run the fortran code in Linux, `gfortran taylor_precomp_fortran.f90 -o taylor_precomp_fortran`.

The python program runs the fortran code using wexpect. On Linux or Mac, one might need to install pexpect instead.

To cite  this work:
Jonas Stein et al,"Combining the QAOA and HHL Algorithm to achieve a Substantial Quantum Speedup for the Unit Commitment Problem".
