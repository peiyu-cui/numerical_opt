#include <opt_solver/gradient_descent.h>

namespace numerical_optimization {

  bool GradientDescent::optimize()
  {
    Eigen::VectorXd grad;
    Eigen::VectorXd x_cur;
    Eigen::VectorXd x_new;
    double fx_cur; 
    double fx_new;
    // start iteration
    x_cur = x0;
    for (int i = 0; i < max_iter; i++) {
      // compute gradient
      func->getGradient(x_cur,grad);
      // check stopping criteria
      if (grad.norm() < epsilon) 
      {
        solution = x_cur;
        f_min = fx_new;
        iter = i + 1;
        std::cout << "\033[1;32mGradient Descent: Optimization successful!\033[0m" << std::endl;
        return true;
      }
      // compute fval
      func->getValue(x_cur, fx_cur);

      // update x_k with armijo condition
      double tau = 1.0;
      x_new = x_cur + tau * (-grad);
      func->getValue(x_new, fx_new);
      while (fx_new > (fx_cur + scale * tau * (-grad).dot(grad)))
      {
        tau *= 0.5;
        x_new = x_cur + tau * (-grad);
        func->getValue(x_new, fx_new);
      }
      x_cur = x_new;
    }
    std::cout << "\033[1;31mGradient Descent: Maximum number of iterations reached, Optimization failed!\033[0m" << std::endl;
    return false;
  }

  void GradientDescent::setInitialGuess(const Eigen::VectorXd& x0_)
  {
    x0 = x0_;
  }

  void GradientDescent::setMaxIter(int max_iter_)
  {
    max_iter = max_iter_;
  }

  void GradientDescent::setStoppingCriteria(double epsilon_) 
  {
    epsilon = epsilon_;
  }

  void GradientDescent::setArmijoScale(double scale_)
  {
    scale = scale_;
  }

  void GradientDescent::setCurvatureScale(double curvature_scale_)
  {
    scale_curvature = curvature_scale_;
  }

  void GradientDescent::setObjectiveFunction(FunctionBase* func_)
  {
    func = func_;
  }

  Eigen::VectorXd GradientDescent::getSolution()
  {
    return solution;
  }

  double GradientDescent::getMinValue()
  {
    return f_min;
  }

  int GradientDescent::getIterationNumber()
  {
    return iter;
  }


}  // namespace numerical_optimization