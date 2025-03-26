#include <opt_solver/cautious_bfgs.h>

namespace numerical_optimization {

  bool CautiousBFGS::optimize()
  {
    // initialize the variables
    Eigen::VectorXd grad_cur, grad_new;
    Eigen::VectorXd x_cur, x_new;
    double fx_cur, fx_new;
    Eigen::VectorXd d; // search direction
    double eta = 1e-6; // scaling factor for cautious update

    x_cur = x0;
    func->getGradient(x_cur, grad_cur);
    func->getValue(x_cur, fx_cur);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_cur.size(), x_cur.size());
    B = I;
    iter = 0;

    // start iteration
    while (grad_cur.norm() > epsilon && iter < max_iter)
    {
      // calculate search direction
      d = -B * grad_cur;

      // Lewis & Overton line search
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


      // inexact line search
      // while (fx_new > fx_cur + scale * tau * d.dot(grad_cur) || d.dot(grad_new) < scale_curvature * d.dot(grad_cur))
      // {
      //   tau *= 0.5;
      //   x_new = x_cur + tau * d;
      //   func->getValue(x_new, fx_new);
      //   func->getGradient(x_new, grad_new);
      // }

      // update B with Cautious-BFGS
      Eigen::VectorXd delta_x = x_new - x_cur;
      Eigen::VectorXd delta_g = grad_new - grad_cur;
      
      if (delta_g.dot(delta_x) > eta * grad_cur.norm() * delta_x.dot(delta_x))
      {
        Eigen::MatrixXd A = I - (delta_x * delta_g.transpose()) / (delta_g.dot(delta_x));
        B = A * B * A.transpose() + (delta_x * delta_x.transpose()) / (delta_g.dot(delta_x));
      }

      iter++;
      x_cur = x_new;
      fx_cur = fx_new;
      grad_cur = grad_new;

    }

    // check iteration limit
    if (iter == max_iter)
    {
      std::cout << "\033[1;31m" << "Cautious-BFGS: Maximum number of iterations reached, optimization failed!" << "\033[0m" << std::endl;
      return false;
    }
    else
    {
      solution = x_cur;
      f_min = fx_cur;
      std::cout << "\033[1;32m" << "Cautious-BFGS: Optimization successful!" << "\033[0m" << std::endl;
      return true;
    }
    
  }

} // namespace numerical_optimization