# PolyGenetic
### Genetic Algorithms for Polynomial Curve Fitting

Designing a gentic algorithm to approximate to a polynomial function specific by the user. Two approaches were tested and compared.

* Approach 1 — The five coefficients of the polynomial are set as the target, represented by a list of integer. The genetic algorithm then attempts to evolve the individuals, each represented by a list of six integers, towards the target list.
* Approach 2 — A range of points on the curve characterized by the polynomial function is generated using the polyval function. These points act as a list of target values that the genetic algorithm attempts to find. Due to the nature of genetic algorithms, the points are initially converted to integers.
