# Numerical Optimization

A Numerical Optimization unconstrained optimizer, including **Gradient Descent, Newton, Newton-CG, BFGS, Cautious-BFGS, LBFGS**......

## 0. About

**Numerical Optimization** is a **C++** library for **unconstrained optimization**. This is the algorithm I reproduced in the process of learning numerical optimization, which takes into account many algorithmic engineering techniques.

## 1. How to use

Please refer to "solver_example.cpp" in the test_pkg package for the calling process. You may need to install Eigen via "sudo apt install libeigen3-dev". We use Eigen to get better performance.

## 2. Features

- Only one function package **opt_solver** is needed.
- The library implements methods such as **Gradient Descent, Newton, Newton-CG, BFGS, Cautious-BFGS, LBFGS**.

## 3. Quick Start

```cmd
git clone https://github.com/peiyu-cui/numerical_opt.git
cd numerical_opt
catkin_make
source devel/setup.bash
roslaunch test_pkg test_solver_node.launch
```

## 4. Tests

* Implemented N-dimensional non-convex function RosenbrockFunc() in "**func.hpp**" to test the performance of different methods

* Below are the test results:

  ```cmd
  Gradient Descent: Optimization successful!
  Optimization finished use:0.0502647 seconds
  Optimization finished use:44854 iterations
  optimal x = 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  optimal f(x) = 6.16137e-17
  Newton: Optimization succeeded!
  Newton Optimization finished use:0.00276089 seconds
  Newton Optimization finished use:110 iterations
  Newton optimal x = 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  Newton optimal f(x) = 1.97215e-31
  BFGS: Optimization successful!
  BFGS Optimization finished use:0.080783 seconds
  BFGS Optimization finished use:961 iterations
  BFGS optimal x = 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  BFGS optimal f(x) = 4.39709e-18
  Cautious-BFGS: Optimization successful!
  Cautious BFGS Optimization finished use:0.0817627 seconds
  Cautious BFGS Optimization finished use:961 iterations
  Cautious BFGS optimal x = 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  Cautious BFGS optimal f(x) = 4.39709e-18
  NewtonCG: Optimization succeeded!
  Newton CG Optimization finished use:0.000726617 seconds
  Newton CG Optimization finished use:71 iterations
  Newton CG optimal x = 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  Newton CG optimal f(x) = 4.70861e-23
  LBFGS: Optimization successful!
  L-BFGS Optimization finished use:0.00998493 seconds
  L-BFGS Optimization finished use:6264 iterations
  L-BFGS optimal x = 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  L-BFGS optimal f(x) = 1.20384e-16
  ```

  
