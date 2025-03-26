#include <iostream>
#include <opt_solver/func.hpp>
#include <opt_solver/gradient_descent.h>
#include <opt_solver/bfgs.h>
#include <opt_solver/cautious_bfgs.h>
#include <opt_solver/newton.h>
#include <opt_solver/newton_cg.h>
#include <opt_solver/lbfgs.h>
#include <ros/ros.h>
#include <memory>

int main(int argc, char **argv) {
    ros::init(argc, argv, "test_solver_node");
    ros::NodeHandle nh("~");

    // create a Rosenbrock function object
    // dimension = 60
    numerical_optimization::FunctionBase* func_ptr = new numerical_optimization::RosenbrockFunc(60);
    // set random initial guess
    // scope: [-10, 10]
    double x_lower = -10.0;
    double x_upper = 10.0;
    Eigen::VectorXd x = x_lower + (x_upper - x_lower) * Eigen::VectorXd::Random(60).array();

    std::unique_ptr<numerical_optimization::GradientDescent> solver(new numerical_optimization::GradientDescent());
    solver->setInitialGuess(x);
    solver->setMaxIter(100000);
    solver->setStoppingCriteria(1e-8);
    solver->setArmijoScale(1e-4);
    solver->setCurvatureScale(0.9);
    solver->setObjectiveFunction(func_ptr);

    std::unique_ptr<numerical_optimization::Newton> newton_solver(new numerical_optimization::Newton());
    newton_solver->setInitialGuess(x);
    newton_solver->setMaxIter(10000);
    newton_solver->setStoppingCriteria(1e-8);
    newton_solver->setArmijoScale(1e-4);
    newton_solver->setCurvatureScale(0.9);
    newton_solver->setObjectiveFunction(func_ptr);

    std::unique_ptr<numerical_optimization::BFGS> bfgs_solver(new numerical_optimization::BFGS());
    bfgs_solver->setInitialGuess(x);
    bfgs_solver->setMaxIter(10000);
    bfgs_solver->setStoppingCriteria(1e-8);
    bfgs_solver->setArmijoScale(1e-4);
    bfgs_solver->setCurvatureScale(0.9);
    bfgs_solver->setObjectiveFunction(func_ptr);

    std::unique_ptr<numerical_optimization::CautiousBFGS> cau_bfgs_solver(new numerical_optimization::CautiousBFGS());
    cau_bfgs_solver->setInitialGuess(x);
    cau_bfgs_solver->setMaxIter(10000);
    cau_bfgs_solver->setStoppingCriteria(1e-8);
    cau_bfgs_solver->setArmijoScale(1e-4);
    cau_bfgs_solver->setCurvatureScale(0.9);
    cau_bfgs_solver->setObjectiveFunction(func_ptr);

    std::unique_ptr<numerical_optimization::NewtonCG> newton_cg_solver(new numerical_optimization::NewtonCG());
    newton_cg_solver->setInitialGuess(x);
    newton_cg_solver->setMaxIter(10000);
    newton_cg_solver->setStoppingCriteria(1e-8);
    newton_cg_solver->setArmijoScale(1e-4);
    newton_cg_solver->setCurvatureScale(0.9);
    newton_cg_solver->setObjectiveFunction(func_ptr);
    newton_cg_solver->setMaxCGIter(1000);
    newton_cg_solver->setCGAlpha(1e-3);

    std::unique_ptr<numerical_optimization::LBFGS> lbfgs_solver(new numerical_optimization::LBFGS());
    lbfgs_solver->setInitialGuess(x);
    lbfgs_solver->setMaxIter(10000);
    lbfgs_solver->setStoppingCriteria(1e-8);
    lbfgs_solver->setArmijoScale(1e-4);
    lbfgs_solver->setCurvatureScale(0.9);
    lbfgs_solver->setObjectiveFunction(func_ptr);
    lbfgs_solver->setMemorySize(16);

    ros::Time start_time = ros::Time::now();
    if (solver->optimize())
    {
        ros::Time end_time = ros::Time::now();
        std::cout << "Optimization finished use:" << (end_time - start_time).toSec() << " seconds" << std::endl;
        std::cout << "Optimization finished use:" << solver->getIterationNumber() << " iterations" << std::endl;
        std::cout << "optimal x = " << solver->getSolution().transpose() << std::endl;
        std::cout << "optimal f(x) = " << solver->getMinValue() << std::endl;
    }

    ros::Time start_time_newton = ros::Time::now();
    if (newton_solver->optimize())
    {
        ros::Time end_time_newton = ros::Time::now();
        std::cout << "Newton Optimization finished use:" << (end_time_newton - start_time_newton).toSec() << " seconds" << std::endl;    
        std::cout << "Newton Optimization finished use:" << newton_solver->getIterationNumber() << " iterations" << std::endl;
        std::cout << "Newton optimal x = " << newton_solver->getSolution().transpose() << std::endl;
        std::cout << "Newton optimal f(x) = " << newton_solver->getMinValue() << std::endl;
    }

    ros::Time start_time_bfgs = ros::Time::now();
    if (bfgs_solver->optimize())
    {
        ros::Time end_time_bfgs = ros::Time::now();
        std::cout << "BFGS Optimization finished use:" << (end_time_bfgs - start_time_bfgs).toSec() << " seconds" << std::endl;
        std::cout << "BFGS Optimization finished use:" << bfgs_solver->getIterationNumber() << " iterations" << std::endl;
        std::cout << "BFGS optimal x = " << bfgs_solver->getSolution().transpose() << std::endl;
        std::cout << "BFGS optimal f(x) = " << bfgs_solver->getMinValue() << std::endl;
    }

    ros::Time start_time_cau_bfgs = ros::Time::now();
    if (cau_bfgs_solver->optimize())
    {
        ros::Time end_time_cau_bfgs = ros::Time::now();
        std::cout << "Cautious BFGS Optimization finished use:" << (end_time_cau_bfgs - start_time_cau_bfgs).toSec() << " seconds" << std::endl;
        std::cout << "Cautious BFGS Optimization finished use:" << cau_bfgs_solver->getIterationNumber() << " iterations" << std::endl;
        std::cout << "Cautious BFGS optimal x = " << cau_bfgs_solver->getSolution().transpose() << std::endl;
        std::cout << "Cautious BFGS optimal f(x) = " << cau_bfgs_solver->getMinValue() << std::endl;
    }

    ros::Time start_time_newton_cg = ros::Time::now();
    if (newton_cg_solver->optimize())
    {
        ros::Time end_time_newton_cg = ros::Time::now();
        std::cout << "Newton CG Optimization finished use:" << (end_time_newton_cg - start_time_newton_cg).toSec() << " seconds" << std::endl;
        std::cout << "Newton CG Optimization finished use:" << newton_cg_solver->getIterationNumber() << " iterations" << std::endl;
        // std::cout << "Newton CG Optimization finished use:" << newton_cg_solver->getCGIterationNumber() << " CG iterations" << std::endl;
        std::cout << "Newton CG optimal x = " << newton_cg_solver->getSolution().transpose() << std::endl;
        std::cout << "Newton CG optimal f(x) = " << newton_cg_solver->getMinValue() << std::endl;
    }

    ros::Time start_time_lbfgs = ros::Time::now();
    if (lbfgs_solver->optimize())
    {
        ros::Time end_time_lbfgs = ros::Time::now();
        std::cout << "L-BFGS Optimization finished use:" << (end_time_lbfgs - start_time_lbfgs).toSec() << " seconds" << std::endl;
        std::cout << "L-BFGS Optimization finished use:" << lbfgs_solver->getIterationNumber() << " iterations" << std::endl;
        std::cout << "L-BFGS optimal x = " << lbfgs_solver->getSolution().transpose() << std::endl;
        std::cout << "L-BFGS optimal f(x) = " << lbfgs_solver->getMinValue() << std::endl;
    }


    delete func_ptr;

    return 0;

}

