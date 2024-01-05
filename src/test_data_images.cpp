#include <iostream>
#include <string>
#include "data.hpp"

int main(int argc,char* argv[]){
    std::string input_filepath = argv[1];
    std::string output_filepath = argv[2];
    size_t index = std::stoi(argv[3]);
    DataSetImages data(5000); //batch size set to 5000
    data.readImageData(input_filepath);
    data.writeImageToFile(output_filepath,index);
    return 0;
}