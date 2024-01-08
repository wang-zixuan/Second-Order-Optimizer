# Second-order optimizers with JAX

## Getting started
We implemented native Newton's method, Hessian-free Newton's method, and Hessian-free Gauss-Newton method for our shallow neural network. The dataset is MNIST. Please refer to the report for details.

## Run the code
For example, to test Hessian-free Newton's method with damping parameter equals to 4, you can test with
```
python3 main.py -on cg -dp 4
```
