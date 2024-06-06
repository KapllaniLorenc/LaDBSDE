# Implementation of the deep learning BSDE algorithms [1] in TensorFlow 2.x.

# Training
To train the algorithms use:

python train_valid.py "LaDBSDE"

Note: the last argument represents the method, e.g. "LaDBSDE" for the LaDBSDE method in [1].

In the file train_valid.py the example type, number of algorithm trainings (runs), problem related parameters (dimension, maturity, discretization points) are selected.
In the file initvar.py the network paramter are selected.

# Results
After training, each model is saved. Different metrics such as loss, error for Y_0 and Z_0 can are calculated using: 

python results.py

The results produced are not similar to the ones in [1] as:
- different metrics are considered
- learning approach is different; a piecewise constant learning rate approach instead of the learning approach in [1]
- ln-transform is applied when the diffusion term of the SDE is not constant
The results produced from this codes will be referenced in the near future.

# Reference
[1] Kapllani, L., Teng, L. Deep Learning algorithms for solving high-dimensional nonlinear Backward Stochastic Differential Equations. Discrete Contin. Dyn. Syst. - B, (2023). 
https://www.aimsciences.org//article/doi/10.3934/dcdsb.2023151
