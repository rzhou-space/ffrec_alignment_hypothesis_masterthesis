Asym_1D file contains python files for asymmetric recurrent networks.


1. AsymNetworks.py

Contains four constructions for asymmetric recurrent interaction metrices.
Generate eigenvalue distributions for Figure 2.3 and Figure 3.14.

2. AsymOperations.py

Contains mainly the modeling of four response properties (for each property a class)
and plots for their correlation against feedforward recurrent alignment (ffrec).
Functions are applied for the CompareOperation.py.

3. CompareOperation.py

Contains functions for comparisons between different symmetry in the interaction matrices
for one response property.
For each response property exists one class for full-rank asymmetric matrices and one
class for low-rank asymmetric matrices.
For full-rank asymmetric matrices: generate the Figure 3.8, Figure 3.9, and Figure 3.10.
For low-rank asymmetric matrices: generate the Figure 3.15 and Figure 3.16.

4. FfrecAlign.py

Contains four modifications of the feedforward recurrent alignment. 
Genrate the monotony plots for feedforward recurrent alignment score: Figure 3.5, Figure
3.6, and Figure 3.17
