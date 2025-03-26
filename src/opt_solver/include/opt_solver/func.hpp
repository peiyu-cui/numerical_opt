#ifndef TEST_PKG_ROSENBROCK_FUNC_HPP
#define TEST_PKG_ROSENBROCK_FUNC_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cmath>

using namespace std;

namespace numerical_optimization {
// main function examples

// FunctionBase: base class for all functions
    class FunctionBase {
        public:
            virtual ~FunctionBase() = default;

            virtual void getValue(const Eigen::VectorXd &x, double &f_value)
            {
                f_value = 0.0;
            }
            virtual void getGradient(const Eigen::VectorXd &x, Eigen::VectorXd &f_grad)
            {
                f_grad.setZero();
            }
            virtual void getHessian(const Eigen::VectorXd &x, Eigen::SparseMatrix<double> &f_hess)
            {
                f_hess.setZero();
            }

    }; // class FunctionBase

    // Rosenbrock function: non convex and smooth function
    // overloading getValue and getGradient
    class RosenbrockFunc : public FunctionBase {
        private:
            int N; // N is even integer

        public:
            RosenbrockFunc(int dimension)
                : N(dimension) {
                    assert(N % 2 == 0);
                };
            ~RosenbrockFunc() {};

            // main interface function
            // f_value: function value
            void getValue(const Eigen::VectorXd &x, double &f_value) override 
            {
                assert(x.size() == N);
                f_value = 0.0;
                for (int i = 1; i <= N / 2; i++)
                {
                    f_value += 100.0 * pow(pow(x(2 * i - 1 - 1), 2) - x(2 * i - 1), 2) + pow(x(2 * i - 1 - 1) - 1.0, 2);
                }

            };

            // f_grad: function gradient
            void getGradient(const Eigen::VectorXd &x, Eigen::VectorXd &f_grad) override 
            {
                assert(x.size() == N);
                f_grad.resize(N);
                f_grad.setZero();
                for (int i = 1; i <= N / 2; i++)
                {
                    f_grad(2 * i - 1 - 1) = 400.0 * x(2 * i - 1 - 1) * (pow(x(2 * i - 1 - 1), 2) - x(2 * i - 1)) + 
                                           2.0 * (x(2 * i - 1 - 1) - 1.0);
                    f_grad(2 * i - 1) = -200.0 * (pow(x(2 * i - 1 - 1), 2) - x(2 * i - 1));
                }
            };

            // f_hess: function hessian
            void getHessian(const Eigen::VectorXd &x, Eigen::SparseMatrix<double> &f_hess) override 
            {
                assert(x.size() == N);
                f_hess.resize(N, N);
                f_hess.setZero();
                for (int i = 1; i <= N / 2; i++)
                {
                    f_hess.insert(2 * i - 1 - 1, 2 * i - 1 - 1) = 400.0 * (pow(x(2 * i - 1 - 1), 2) - x(2 * i - 1)) + 
                                                                  800.0 * pow(x(2 * i - 1 - 1), 2) + 2.0;
                    f_hess.insert(2 * i - 1 - 1, 2 * i - 1) = -400.0 * x(2 * i - 1 - 1);
                    f_hess.insert(2 * i - 1, 2 * i - 1 - 1) = -400.0 * x(2 * i - 1 - 1);
                    f_hess.insert(2 * i - 1, 2 * i - 1) = 200.0;
                }
            }

    };

} // namespace numerical_optimization

#endif //TEST_PKG_ROSENBROCK_FUNC_HPP