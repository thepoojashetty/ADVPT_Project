#include <iostream>
#include <string>
#include "label.hpp"

int main(int argc,char* argv[]){
    std::string input_filepath = argv[1];
    std::string output_filepath = argv[2];
    size_t index = std::stoi(argv[3]);
    DatasetLabels labels(5000); //batch size set to 5000
    labels.readLabelData(input_filepath);
    labels.writeLabelToFile(output_filepath,index);
    return 0;
}