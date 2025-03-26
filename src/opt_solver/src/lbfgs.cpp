#include <opt_solver/lbfgs.h>

namespace numerical_optimization {

bool LBFGS::optimize()
{
  // initialize
  Eigen::VectorXd grad_cur, grad_new;
  Eigen::VectorXd x_cur, x_new;
  double fx_cur, fx_new;
  Eigen::VectorXd d;

  x_cur = x0;
  func->getGradient(x_cur, grad_cur);
  func->getValue(x_cur, fx_cur);

  std::vector<Eigen::VectorXd> s, y;
  std::vector<double> rho;
  iter = 0;

  while (grad_cur.norm() > epsilon && iter < max_iter)
  {
    //update direction using memory of previous m steps
    d = lbfgs_direction(s, y, rho, -grad_cur);

    // backtracking line search to find the best step size
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

    Eigen::VectorXd delta_x = x_new - x_cur;
    Eigen::VectorXd delta_g = grad_new - grad_cur;

    if (delta_x.dot(delta_g) > 1e-10)
    {
      if (s.size() == mem_size)
      {
        s.erase(s.begin());
        y.erase(y.begin());
        rho.erase(rho.begin());
      }
      s.push_back(delta_x);
      y.push_back(delta_g);
      rho.push_back(1.0 / delta_g.dot(delta_x));
    }

    iter++;
    x_cur = x_new;
    grad_cur = grad_new;
    fx_cur = fx_new;

  }

  if (iter == max_iter)
  {
    std::cout << "\033[1;31m" << "LBFGS: Maximum number of iterations reached, optimization failed!" << "\033[0m" << std::endl;
    return false;
  }
  else
  {
    solution = x_cur;
    f_min = fx_cur;
    std::cout << "\033[1;32m" << "LBFGS: Optimization successful!" << "\033[0m" << std::endl;
    return true;
  }
  
}

Eigen::VectorXd LBFGS::lbfgs_direction(const std::vector<Eigen::VectorXd> &s, 
  const std::vector<Eigen::VectorXd> &y, 
  const std::vector<double> &rho, 
  const Eigen::VectorXd &grad)
{
  int history_size = s.size();
  std::vector<double> alpha(history_size);
  Eigen::VectorXd d = grad;

  for (int i = history_size - 1; i >= 0; i--) {
    alpha[i] = rho[i] * s[i].dot(d);
    d -= alpha[i] * y[i];
  }

  double gamma = (history_size > 0) ? rho.back() * y.back().dot(y.back()) : 1.0;
  d = d / gamma;

  for (int i = 0; i < history_size; i++) {
    double beta = rho[i] * y[i].dot(d);
    d += (alpha[i] - beta) * s[i];
  }

  return d;
}

} // namespace numerical_optimization