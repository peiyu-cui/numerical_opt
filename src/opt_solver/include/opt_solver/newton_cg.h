#ifndef NEWTON_CG_H
#define NEWTON_CG_H

#include <opt_solver/gradient_descent.h>

namespace numerical_optimization {

  class NewtonCG : public GradientDescent {
    private:
      int cg_iter;
      int max_cg_iter;
      double cg_alpha;     // small disturbance of approximately compute the Hessian vector product, default is 1e-8

    public:
      bool optimize();
      void setMaxCGIter(int max_iter) { max_cg_iter = max_iter; }
      void setCGAlpha(double alpha) { cg_alpha = alpha; }

      int getCGIterationNumber() const { return cg_iter; }
  };

} // namespace numerical_optimization


#endif // NEWTON_CG_H