#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <Eigen/Core>
#include <opt_solver/func.hpp>
#include <iostream>

namespace numerical_optimization {
// An implementation of steepest gradient descent algorithm
// using line search and armijo condition
// as the SolverBase class, other class need to inherit from this class 
// and overload the optimize() function
class GradientDescent {
    protected:
        int max_iter;           // maximum number of iterations
        double epsilon;         // tolerance for stopping criteria
        double scale;           // scaling factor for armijo condition
        double scale_curvature; // scaling factor for curvature condition
        Eigen::VectorXd x0;     // initial guess
        Eigen::VectorXd solution; // solution vector
        double f_min;             // minimum value of objective function
        FunctionBase* func;     // pointer to objective function
        int iter;               // current iteration number

    public:
        GradientDescent() {};
        ~GradientDescent() {};

        // main optimization function
        virtual bool optimize();

        // main interface function
        void setInitialGuess(const Eigen::VectorXd& x0_);
        void setMaxIter(int max_iter_);
        void setStoppingCriteria(double epsilon_);
        void setArmijoScale(double scale_);
        void setCurvatureScale(double scale_);
        void setObjectiveFunction(FunctionBase* func_);

        Eigen::VectorXd getSolution();
        double getMinValue();
        int getIterationNumber();

};

} // namespace numerical_optimization
#endif // GRADIENT_DESCENT_H