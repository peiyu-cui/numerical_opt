#include <opt_solver/gradient_descent.h>


namespace numerical_optimization {

// An implementation of Cautious-BFGS algorithm
// make sure the convergence
// use Lewis & Overton line search and weak wolfe condition
// Scope of application: convex or non-convex and smooth or non-smooth

class CautiousBFGS : public GradientDescent {
  private:
    Eigen::MatrixXd B;
  public:
    bool optimize() override;
};


} // namespace numerical_optimization