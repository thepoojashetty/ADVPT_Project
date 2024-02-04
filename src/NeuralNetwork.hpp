#pragma once

#include "Eigen/Dense"
#include "Loss.hpp"
#include "Optimizers.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "FullyConnected.hpp"
#include "data.hpp"
#include "label.hpp"

class NeuralNetwork
{
    private:
        double learningRate;
        int numEpochs;
        int hiddenLayerSize;
        int batchSize;
        int inputsize = 784;
        FullyConnected fc1;
        FullyConnected fc2;
        Relu relu;
        Softmax softmax;
        CrossEntropyLoss celoss;
        SGD sgd;
        std::string trainDataPath;
        std::string trainLabelsPath;
        std::string testDataPath;
        std::string testLabelsPath;
        std::string predictionLogFilePath;
    public:
        NeuralNetwork(double, int, int, int, std::string, std::string, std::string, std::string, std::string);
        ~NeuralNetwork();

        Eigen::MatrixXd forward(const Eigen::MatrixXd &);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &);
        void train();
        void test();

};

NeuralNetwork::NeuralNetwork(double learningRate, 
                            int numEpochs, 
                            int batchSize, 
                            int hiddenLayerSize,
                            std::string trainDataPath,
                            std::string trainLabelsPath,
                            std::string testDataPath,
                            std::string testLabelsPath,
                            std::string predictionLogFilePath){
    this->learningRate = learningRate;
    this->numEpochs = numEpochs;
    this->batchSize = batchSize;
    this->hiddenLayerSize = hiddenLayerSize;
    this->trainDataPath = trainDataPath;
    this->trainLabelsPath = trainLabelsPath;
    this->testDataPath = testDataPath;
    this->testLabelsPath = testLabelsPath;
    this->predictionLogFilePath = predictionLogFilePath;

    sgd = SGD(learningRate);
    fc1 = FullyConnected(inputsize,hiddenLayerSize);
    fc2 = FullyConnected(hiddenLayerSize,10);
}

NeuralNetwork::~NeuralNetwork(){}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd &inputTensor){
    Eigen::MatrixXd fc1Forward = fc1.forward(inputTensor);
    Eigen::MatrixXd reluForward = relu.forward(fc1Forward);
    Eigen::MatrixXd fc2Forward = fc2.forward(reluForward);
    Eigen::MatrixXd softmaxForward = softmax.forward(fc2Forward);
    return softmaxForward;
}

Eigen::MatrixXd NeuralNetwork::backward(const Eigen::MatrixXd &errorTensor){
    Eigen::MatrixXd softmaxBackward = softmax.backward(errorTensor);
    Eigen::MatrixXd fc2Backward = fc2.backward(softmaxBackward,sgd);
    Eigen::MatrixXd reluBackward = relu.backward(fc2Backward);
    Eigen::MatrixXd fc1Backward = fc1.backward(reluBackward,sgd);
    return fc1Backward;
}

void NeuralNetwork::train(){
    // load data
    // Log the training process go cout
    std::cout<<"Training started"<<std::endl;
    DataSetImages trainData(batchSize);
    trainData.readImageData(trainDataPath);
    DatasetLabels trainLabels(batchSize);
    trainLabels.readLabelData(trainLabelsPath);

    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Number of epochs: " << numEpochs << std::endl;
    for (int i=0;i<numEpochs;i++){
        std::cout<<"Epoch: "<<i<<std::endl;
        // for each batch
        for (size_t j=0;j<trainData.getNoOfBatches();j++){
            // forward
            Eigen::MatrixXd predictedOutput = forward(trainData.getBatch(j));
            // loss
            double loss = celoss.forward(predictedOutput, trainLabels.getBatch(j));
            // backward
            Eigen::MatrixXd lossBackward = celoss.backward(trainLabels.getBatch(j));
            backward(lossBackward);
            std::cout<<" - Batch: "<<j<<". Loss: "<<loss<<std::endl;
        }
    }
}

void NeuralNetwork::test(){
    // load data
    DataSetImages testData(batchSize);
    testData.readImageData(testDataPath);
    DatasetLabels testLabels(batchSize);
    testLabels.readLabelData(testLabelsPath);

    std::ofstream predictionLogFile(predictionLogFilePath);

    // for each batch
    for (size_t j=0;j<testData.getNoOfBatches();j++){
        predictionLogFile<<"Current batch: "<<j<<std::endl;
        // Also print to cout
        std::cout<<"Current batch: "<<j<<std::endl;
        // forward
        Eigen::MatrixXd predictedOutput = forward(testData.getBatch(j));

        for (int i=0;i<predictedOutput.rows();i++){
            Eigen::Index predLabel;
            predictedOutput.row(i).maxCoeff(&predLabel);
            Eigen::Index actualLabel;
            testLabels.getBatch(j).row(i).maxCoeff(&actualLabel);
            predictionLogFile<<" - image "<<j*batchSize+i<<": Prediction="<<predLabel<<". Label="<<actualLabel<<std::endl;
            // Also print to cout
            std::cout<<" - image "<<j*batchSize+i<<": Prediction="<<predLabel<<". Label="<<actualLabel<<std::endl;
        }
    }
    predictionLogFile.close();
}   


