#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>

#include "alex_patch.hpp"

// struct path_leaf_string
// {
//     std::string operator()(const boost::filesystem::directory_entry& entry) const
//     {
//         return entry.path().leaf().string();
//     }
// };

// void read_directory(const std::string& name, std::vector<std::string>& v)
// {
//     boost::filesystem::path p(name);
//     boost::filesystem::directory_iterator start(p);
//     boost::filesystem::directory_iterator end;
//     std::transform(start, end, std::back_inserter(v), path_leaf_string());
// }


// void EvaluateSomePatches(){
//     // Simple function that evaluates the samples in teh media folder and saves a CSV file with the results.
//     // This is hard coded to evaluate the images from sequence 000000.
//     // std::vector<std::string> 
//     std::vector<std::string> v;
//     read_directory(".", v);
//     std::copy(v.begin(), v.end(),
//          std::ostream_iterator<std::string>(std::cout, "\n"));
// }

int main(){
    // Instance of AlexPatch.
    AlexPatch alex_patch;
    
    // Patch image (cv::Mat) from path.
    std::string patch_1_path = "media/0000/000000/1.png";
    cv::Mat patch1 = cv::imread(patch_1_path);

    std::string patch_2_path = "media/0000/000000/2.png";
    cv::Mat patch2 = cv::imread(patch_2_path);


    // Ask for similarity between two patches.
    float simi_score = alex_patch.GetSimilarityBwPatches(patch1, patch2);
    std::cout<<"Similarity Score is " << simi_score << "\n";

    alex_patch.EvaluateSequence("0000");
}