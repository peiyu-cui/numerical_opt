#ifndef LBFGS_H
#define LBFGS_H

#include <opt_solver/gradient_descent.h>

namespace numerical_optimization {

class LBFGS : public GradientDescent {
  private:
   int mem_size;

   Eigen::VectorXd lbfgs_direction(const std::vector<Eigen::VectorXd> &s, 
                                  const std::vector<Eigen::VectorXd> &y, 
                                  const std::vector<double> &rho, 
                                  const Eigen::VectorXd &grad);

  public:
    bool optimize() override;
    void setMemorySize(int mem_size_) { mem_size = mem_size_; };
};

} // namespace numerical_optimization

#endif // LBFGS_H