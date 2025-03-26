#ifndef NEWTON_H
#define NEWTON_H

#include <opt_solver/gradient_descent.h>

namespace numerical_optimization {

// Newton method
// need to compute Hessian matrix and inverse Hessian matrix
// Hessian matrix is sparse matrix, so we use Eigen::SparseMatrix<double>
// Scope of application: strictly convex and smooth function
class Newton : public GradientDescent {
  private:
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

  public:
    bool optimize() override;
};

} // namespace numerical_optimization

#endif // NEWTON_H