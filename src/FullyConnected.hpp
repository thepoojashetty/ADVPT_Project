#ifndef ADVPT_MPU_FULLY_CONNECTED_HPP
#define ADVPT_MPU_FULLY_CONNECTED_HPP

#include "Eigen/Dense"
#include "Optimizers.hpp"
#include <iostream>

class FullyConnected
{
private:
    Eigen::MatrixXd weights;
    size_t input_size;
    size_t output_size;
    double range = 1.0;

    Eigen::MatrixXd input_tensor;

public:
    FullyConnected() {
    }
    FullyConnected(size_t in, size_t out) : input_size(in), output_size(out)
    {
        range = 1.0 / sqrt(input_size);
        weights = Eigen::MatrixXd::Random(input_size+1, output_size);
        weights = weights * range;
    };

    void setWeights(Eigen::MatrixXd w) {
        // for testing purposes
        weights = w;
    }

    Eigen::MatrixXd forward(Eigen::MatrixXd input) {
        // save input tensor for backward-pass
        input_tensor = Eigen::MatrixXd(input.rows(), input.cols()+1);
        auto ones = Eigen::MatrixXd::Constant(input.rows(), 1, 1.0);
        input_tensor << input, ones;
        Eigen::MatrixXd output= input_tensor*weights;
        return output;
    }

    Eigen::MatrixXd backward(Eigen::MatrixXd error_tensor, SGD sgd) {
        Eigen::MatrixXd gradient_weights(weights.rows(), weights.cols());
        gradient_weights = input_tensor.transpose()*error_tensor;
        // std::cout << "bfr_weights: " << weights.row(0).segment(0,7) << std::endl;
        weights = sgd.updateWeights(weights, gradient_weights);
        return error_tensor*weights.transpose();
    }

    ~FullyConnected() {}
};


#endif //ADVPT_MPU_FULLY_CONNECTED_HPP
