
/**
 * @copyright Copyright MIT 2020
 *
 * @file alex_patch.hpp
 *
 * @date 2020-04-13
 * @author Yorai Shaoul (yorai@mit.edu)
 *
 * @brief Computes the matching cost between two patches using learned networks.
 *
 */

#pragma once

#include <deque>
#include <iostream>
#include <vector>

#include <torch/script.h>  // One-stop header.

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace alex_patch {

class AlexPatch final {
 public:
  explicit AlexPatch(const std::string& model_path);

  /**
   * Similarity value vetween two patches. Updates descriptors associated with
   * each patch.

   * @param  {const cv::Mat &} patch1 :
   * @param  {const cv::Mat &} patch2 :
   * @param  {cv::Mat*} desc1         : to be modified with the first descriptor
   * vector.
   * @param  {cv::Mat*} desc2         : to be modified with the second
   * descriptor vector.

   * @return {float}                  : the similarity score between the
   * patches. (Lower is more similar.)
   */

  float PatchDistanceL2(const cv::Mat& patch1, const cv::Mat& patch2,
                        cv::Mat* desc1 = nullptr, cv::Mat* desc2 = nullptr);

  // Modifies a pointer to a cv::Mat, stores extracted descriptor.
  void DescriptorFromPatch(const cv::Mat& patch, cv::Mat* desc = nullptr);

 private:
  torch::jit::script::Module model;

  /**
   * ImageToTensorImagenet
   * Converts image to be consistent with Alexnet input format.
   *
   * @param  {cv::Mat} img    :
   * @return {torch::Tensor}  :
   */
  const torch::Tensor ImageToTensorImagenet(cv::Mat img);

  /**
   * PatchToTensor
   *
   * @param  {const cv::Mat &} patch : the image we are interested to get a
   * descriptor for.
   * @return {torch::Tensor}         : the descriptor vector tensor.
   */
  torch::Tensor PatchToDescTensor(const cv::Mat& patch);

  /**
   * TensorDistanceL2
   *
   * @param  {const torch::Tensor &} tensor1 :
   * @param  {const torch::Tensor &} tensor2 :
   * @return {float}                         : L2 distance between input
   * tensors.
   */
  float TensorDistanceL2(const torch::Tensor& tensor1,
                         const torch::Tensor& tensor2);
};

}  // namespace alex_patch