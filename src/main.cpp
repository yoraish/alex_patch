#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>

#include "alex_patch.hpp"



int main(){
    // Instance of AlexPatch.
    AlexPatch alex_patch;
    
    // Patch image (cv::Mat) from path.
    std::string patch_1_path = "media/0000/000000/0.png";
    cv::Mat patch1 = cv::imread(patch_1_path);

    std::string patch_2_path = "media/0000/000000/1.png";
    cv::Mat patch2 = cv::imread(patch_2_path);

    // Ask for similarity between two patches.
    float simi_score = alex_patch.GetSimilarityBwPatches(patch1, patch2);
    std::cout<<"Similarity Score is " << simi_score << "\n";

    alex_patch.EvaluateSequence("0000");
}