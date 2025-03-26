#ifndef BFGS_H
#define BFGS_H

#include <Eigen/Core>
#include <iostream>
#include <opt_solver/gradient_descent.h>

namespace numerical_optimization {

// An implementation of BFGS algorithm
// Scope of application: strictly convex and smooth or non-smooth
// use Weak Wolfe condition
class BFGS : public GradientDescent {
  private:
    Eigen::MatrixXd B; // inverse of Hessian matrix
    
  public:
    bool optimize() override;

}; 

}

#endif // BFGS_H