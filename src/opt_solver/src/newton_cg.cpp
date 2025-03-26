#include <opt_solver/newton_cg.h>

namespace numerical_optimization {

bool NewtonCG::optimize()
{
  // initialize
  Eigen::VectorXd grad_cur, grad_new;
  Eigen::VectorXd x_cur, x_new;
  Eigen::VectorXd v_cur, v_new;
  Eigen::VectorXd u_cur, u_new;
  double fx_cur, fx_new;
  Eigen::VectorXd d; 
  Eigen::VectorXd d_pre;
  Eigen::SparseMatrix<double> H;
  
  double var_eps;       // enhance the convergence of CG, when gradient is large, solve Ax=b coarsely
  double alpha;         // exact line search step size
  double beta;          // step size for CG update

  x_cur = x0;
  func->getGradient(x_cur, grad_cur);
  func->getValue(x_cur, fx_cur);
  iter = 0;
  d.resize(x_cur.size());
  
  while (grad_cur.norm() > epsilon && iter < max_iter)
  {
    var_eps = std::min(1.0, grad_cur.norm()) / 10.0;
    d.setZero();
    v_cur = -grad_cur;
    u_cur = v_cur;
    cg_iter = 0;
    func->getHessian(x_cur, H);
    // begin CG loop to solve H*d = -grad_cur
    // calculate exact line search direction d
    while (v_cur.norm() > var_eps * grad_cur.norm())
    {
      // // calculate Hessian vector product Hu of x_cur
      // Eigen::VectorXd x_temp = x_cur + cg_alpha * u_cur;
      // Eigen::VectorXd grad_temp;
      // func->getGradient(x_temp, grad_temp);
      // Eigen::VectorXd Hu = (grad_temp - grad_cur) / cg_alpha;

      // use exact Hessian to calculate H*d
      Eigen::VectorXd Hu = H * u_cur;

      if (u_cur.dot(Hu) <= 0)
      {
        if (cg_iter == 0) d = -grad_cur;
        else d = d_pre;
        break;
      }

      alpha = v_cur.dot(v_cur) / (u_cur.dot(Hu));
      d_pre = d;
      d += alpha * u_cur;
      v_new = v_cur - alpha * Hu;
      beta = v_new.dot(v_new) / v_cur.dot(v_cur);
      u_new = v_new + beta * u_cur;
      cg_iter++;
      v_cur = v_new;
      u_cur = u_new;
    }
    
    if (cg_iter == max_cg_iter)
    {
      std::cout << "\033[1;31m" << "NewtonCG: CG failed to converge, optimization is not precise enough!" << "\033[0m" << std::endl;
      return false;
    }

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

    iter++;
    x_cur = x_new;
    fx_cur = fx_new;
    grad_cur = grad_new;

  }

  if (iter == max_iter)
  {
    std::cout << "\033[1;31m" << "NewtonCG: Maximum number of iterations reached, optimization failed!" << "\033[0m" << std::endl;
    return false;
  }
  else
  {
    solution = x_cur;
    f_min = fx_cur;
    std::cout << "\033[1;32m" << "NewtonCG: Optimization succeeded!" << "\033[0m" << std::endl;
    return true;
  }


}


} // namespace numerical_optimization