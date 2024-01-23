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
        std::cout << ("$ $ $ $ $ $ $ $ $ $ --- no param constructor") << std::endl;
    }
    FullyConnected(size_t in, size_t out) : input_size(in), output_size(out)
    {
        std::cout << ("$ $ $ $ $ $ $ $ $ $ --- FC constructor") << std::endl;
        weights = Eigen::MatrixXd::Random(input_size+1, output_size);
    };

    Eigen::MatrixXd forward(Eigen::MatrixXd input) {
        // save input tensor for backward-pass
        input_tensor = input;

        auto W = weights.block(0, 0, input_size, output_size);
        auto mul = input*W; //input * weights[:-1, :]
        auto bias = weights.block(input_size, 0, 1, output_size); // weights[-1, :]
        auto vbias = Eigen::VectorXd {bias.reshaped()};

        return mul.rowwise() + vbias.transpose();
    }

    Eigen::MatrixXd backward(Eigen::MatrixXd error, SGD sgd) {
        // save input tensor for backward-pass
        auto W = weights.block(0, 0, input_size, output_size);

        return input_tensor;
    }

    ~FullyConnected() {}
};


#endif //ADVPT_MPU_FULLY_CONNECTED_HPP
