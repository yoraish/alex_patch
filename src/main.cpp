/**
 * @copyright Copyright MIT 2020
 *
 * @file alex_patch.cpp
 *
 * @date 2020-04-13
 * @author Yorai Shaoul (yorai@mit.edu)
 *
 * @brief Driver code for alex_patch.
 *
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include "alex_patch.hpp"

int main() {
  // Instance of AlexPatch.
  AlexPatch alex_patch;

  // Patch image (cv::Mat) from path.
  std::string patch_1_path = "media/0000/000000/0.png";
  cv::Mat patch1 = cv::imread(patch_1_path);

  std::string patch_2_path = "media/0000/000000/1.png";
  cv::Mat patch2 = cv::imread(patch_2_path);

  // Ask for similarity between two patches.
  cv::Mat desc2;
  cv::Mat desc1;
  float simi_score = alex_patch.PatchDistanceL2(patch1, patch2, &desc1, &desc2);
  std::cout << "Similarity Score is " << simi_score << "\n";

  alex_patch.EvaluateSequence("0000");
}