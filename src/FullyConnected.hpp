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

    Eigen::MatrixXd input_tensor;

public:
    FullyConnected() {
    }
    FullyConnected(size_t in, size_t out) : input_size(in), output_size(out)
    {
        weights = Eigen::MatrixXd::Random(input_size+1, output_size);
    };

    void setWeights(Eigen::MatrixXd w) {
        // for testing purposes
        weights = w;
    }

    Eigen::MatrixXd forward(Eigen::MatrixXd input) {
        // save input tensor for backward-pass
        input_tensor = input;

        auto W = weights.block(0, 0, input_size, output_size);
        auto mul = input*W; //input * weights[:-1, :]
        auto bias = weights.block(input_size, 0, 1, output_size); // weights[-1, :]
        auto vbias = Eigen::VectorXd {bias.reshaped()};

        return mul.rowwise() + vbias.transpose();
    }

    Eigen::MatrixXd backward(Eigen::MatrixXd error_tensor, SGD sgd) {
        auto W = weights.block(0, 0, input_size, output_size);
        auto error = error_tensor*W.transpose();
    
        Eigen::MatrixXd extended(input_tensor.rows(), input_tensor.cols()+1);
        auto ones = Eigen::MatrixXd::Constant(input_tensor.rows(), 1, 1.0);

        extended << input_tensor, ones;

        Eigen::MatrixXd gradient_weights(weights.rows(), weights.cols());
        gradient_weights = extended.transpose()*error_tensor;
        sgd.updateWeights(weights, gradient_weights);

        return error;
    }

    ~FullyConnected() {}
};


#endif //ADVPT_MPU_FULLY_CONNECTED_HPP
