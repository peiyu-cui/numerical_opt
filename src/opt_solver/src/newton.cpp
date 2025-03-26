#include <opt_solver/newton.h>

namespace numerical_optimization {

  bool Newton::optimize()
  {
    // initialize variables
    Eigen::VectorXd grad_cur, grad_new;
    Eigen::VectorXd x_cur, x_new;
    double fx_cur, fx_new;
    Eigen::VectorXd d;
    Eigen::SparseMatrix<double> H;

    x_cur = x0;
    func->getGradient(x_cur, grad_cur);
    func->getValue(x_cur, fx_cur);

    // Newton's method loop
    iter = 0;
    while (grad_cur.norm() > epsilon && iter < max_iter)
    {
      // compute search direction
      func->getHessian(x_cur, H);
      
      solver.compute(H);
      d = solver.solve(-grad_cur);
      // backtracking line search
      // Wolfe condition
      double tau = 1.0;
      double tau_min = 0.0;
      double tau_max = 1e10;
      x_new = x_cur + tau * d;
      func->getValue(x_new, fx_new);
      func->getGradient(x_new, grad_new);
      while (fx_new > fx_cur + scale * tau * d.dot(grad_cur) || d.dot(grad_new) < scale_curvature * d.dot(grad_cur))
      {
        if (fx_new > fx_cur + scale * tau * d.dot(grad_cur))
        {
          // Armijo condition failed, update tau_max
          tau_max = tau;
        }
        else if (d.dot(grad_new) < scale_curvature * d.dot(grad_cur))
        {
          // curvature condition failed, update tau_min
          tau_min = tau;
        }

        if (tau_max < 1e10)
        {
          // update tau
          tau = (tau_max + tau_min) / 2.0;
          x_new = x_cur + tau * d;
          func->getValue(x_new, fx_new);
          func->getGradient(x_new, grad_new);
        }
        else
        {
          tau = 2.0 * tau_min;
          x_new = x_cur + tau * d;
          func->getValue(x_new, fx_new);
          func->getGradient(x_new, grad_new);
        }
      }
      x_cur = x_new;
      fx_cur = fx_new;
      grad_cur = grad_new;
      iter++;
    }

    if (iter == max_iter)
    {
      std::cout << "\033[1;31m" << "Newton: Maximum number of iterations reached, optimization failed!" << "\033[0m" << std::endl;
      return false;
    }
    else
    {
      solution = x_cur;
      f_min = fx_cur;
      std::cout << "\033[1;32m" << "Newton: Optimization succeeded!" << "\033[0m" << std::endl;
      return true;
    }
  }

} // namespace numerical_optimization