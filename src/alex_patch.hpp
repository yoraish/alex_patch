/*
Yorai for CSAIL RRG March 2020.

API for interfacing with pretrained AlexNet, without its last layer. 
To be used as a descriptor generation tool for image patch matching.
*/

#ifndef ALEX_PATCH_HPP
#define ALEX_PATCH_HPP

// #include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <deque>
#include <vector>

class AlexPatch
{
private:
    torch::jit::script::Module model; 
public:
    AlexPatch(/* args */);
    ~AlexPatch();
    void SetupModel();

    // Get similarity score between two image patches.
    // The lower the score the more similar the patches are.
    /**
     * Inputs: 
               cv::Mat patch_1
               cv::Mat patch_2
     * Outputs:
               float similarity score
    */
    float GetSimilarityBwPatches(cv::Mat patch_1, cv::Mat patch_2);
    
    // Transform cv::Mat image to the AlexNet input requirements.
    torch::Tensor ImageToTensorImagenet(cv::Mat img);

};



#endif

