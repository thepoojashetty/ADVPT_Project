#include <iostream>
#include <string>
#include "label.hpp"

int main(){
    // WIP: test code for label.hpp
    // input file path should be "mnist-datasets/single-label.idx1-ubyte"
    // output file path should be "mnist-temp-output/single-label.txt"

    std::string input_filepath = "mnist-datasets/single-label.idx1-ubyte";
    std::string output_filepath = "mnist-temp-output/single-label.txt";
    // size_t index = std::stoi(0);
    DatasetLabels labels(5000); //batch size set to 5000
    labels.readLabelData(input_filepath);
    // labels.writeImageToFile(output_filepath,index);
    return 0;
}