#include <iostream>
#include <string>
#include "NeuralNetwork.hpp"
#include "Eigen/Dense"

int main(int argc, char **argv)
{
    // Eigen::MatrixXd x = Eigen::MatrixXd::Random(2, 2);
    // x -= 0.5;
    // Relu r = Relu();

    // std::cout << x << std::endl;
    // std::cout << r.forward(x) << std::endl;

    NeuralNetwork nn(0.01, 10, 1, 128,
    "/home/cip/ai2023/ys05ydac/fau/ws2023-group-32-mpu/mnist-datasets/single-image.idx3-ubyte",
    "/home/cip/ai2023/ys05ydac/fau/ws2023-group-32-mpu/mnist-datasets/single-label.idx1-ubyte",
    "/home/cip/ai2023/ys05ydac/fau/ws2023-group-32-mpu/mnist-datasets/single-image.idx3-ubyte",
    "/home/cip/ai2023/ys05ydac/fau/ws2023-group-32-mpu/mnist-datasets/single-label.idx1-ubyte",
    "/home/cip/ai2023/ys05ydac/fau/ws2023-group-32-mpu/prediction.log.txt");
    nn.train();
    // nn.test();
    return 0;
}