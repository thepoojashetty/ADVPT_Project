#include <iostream>
#include <string>
#include "NeuralNetwork.hpp"
#include "Eigen/Dense"

int main(int argc, char **argv)
{
    //parse argv for hyperparameters
    double learningRate = std::stod(argv[1]);
    int numEpochs = std::stoi(argv[2]);
    int batchSize = std::stoi(argv[3]);
    int hiddenLayerSize = std::stoi(argv[4]);
    std::string trainDataPath = argv[5];
    std::string trainLabelsPath = argv[6];
    std::string testDataPath = argv[7];
    std::string testLabelsPath = argv[8];
    std::string predictionLogFilePath = argv[9];

    NeuralNetwork nn(learningRate, numEpochs, batchSize, hiddenLayerSize, trainDataPath, trainLabelsPath, testDataPath, testLabelsPath, predictionLogFilePath);
    //nn.train();
    nn.test();
    return 0;
}