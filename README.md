# Numerical Optimization of Polynomial Function
### Polynomial Curve Fitting using Genetic Algorithms

Designing a genetic algorithm to approximate to a polynomial function specified by the user. This was part of the coursework for ME40212 Biomimetics at the University of Bath, which involved a literature review of evolutionary algorithms for optimization problems.  

Two approaches were developed, tested, and compared. For the first approach (ga1), a range of points on the curve characterized by the polynomial function is generated using the polyval function. These points act as a list of target values that the genetic algorithm attempts to find. Due to the nature of genetic algorithms, the points are initially converted to integers. For the second approach (ga2), the coefficients of the polynomial are set as the target, represented by a list of integer. The genetic algorithm then attempts to evolve the individuals, each represented by a list of six integers, towards the target list. 