NN_1D file contains the python files for symmetric recurrent networks.

1. Networks.py

Contains the methods for different constructions of symmetric interaction matrices. 
Including full-rank symmetric interaction matrix and low-rank interaction matrices.
There are low-rank without noise and low-rank with noise constructions. 
Generate eigenvalue distribution Figure 3.11 in the thesis.

2. NWoperations.py

Contains the methods for modeling response properties for symmetric recurrent 
networks. Those are trial-to-trial correlation, intra-trial stability, dimensionality, 
and alignment to spontaneous activity. Can be applied for different symmetric 
network construction from Networks.py.
Generate Figure 3.1, Figure 3.2, Figure 3.3, Figure 3.4, Figure 3.12, Figure 3.13 
in the thesis. 
