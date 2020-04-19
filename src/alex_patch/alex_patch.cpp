
/**
 * @copyright Copyright MIT 2020
 *
 * @file alex_patch.cpp
 *
 * @date 2020-04-13
 * @author Yorai Shaoul (yorai@mit.edu)
 *
 * @brief compute the learned matching cost between image patches.
 *
 */
#include "alex_patch.hpp"

#include <string>
#include <vector>

namespace alex_patch {

AlexPatch::AlexPatch(const std::string& model_path) {
  std::cout << "New AlexPath Instance\n";
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(model_path);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  } catch (...) {
    std::cerr << "unknown error\n";
  }
  std::cout << "ok\n";
}

torch::Tensor AlexPatch::PatchToDescTensor(const cv::Mat& patch) {
  // Transforming image patch to torch tensor, and passing through model.
  torch::Tensor transformed_tensor = ImageToTensorImagenet(patch);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(transformed_tensor);
  // Execute the model and turn its output into a tensor.
  torch::Tensor desc = model.forward(inputs).toTensor();
  return desc;
}

float AlexPatch::TensorDistanceL2(const torch::Tensor& tensor1,
                                  const torch::Tensor& tensor2) {
  torch::Tensor diff = tensor1 - tensor2;
  float dist = diff.norm(2).item<float>();
  return dist;
}

float AlexPatch::PatchDistanceL2(const cv::Mat& patch1, const cv::Mat& patch2,
                                 cv::Mat* desc1, cv::Mat* desc2) {
  // Convert each patch to a torch::Tensor.
  torch::Tensor desc_tensor_2 = PatchToDescTensor(patch2);
  torch::Tensor desc_tensor_1 = PatchToDescTensor(patch1);

  // Update descriptors in cv::Mat format.
  cv::Mat mat_from_desc_1(
      cv::Size(desc_tensor_1.size(1), desc_tensor_1.size(0)), CV_32F,
      desc_tensor_1.data_ptr());
  cv::Mat mat_from_desc_2(
      cv::Size(desc_tensor_2.size(1), desc_tensor_2.size(0)), CV_32F,
      desc_tensor_2.data_ptr());

  (*desc1) = mat_from_desc_1.clone();
  (*desc2) = mat_from_desc_2.clone();
  // Get score.
  float dist = TensorDistanceL2(desc_tensor_1, desc_tensor_2);
  return dist;
}

/*C++ implementation of the following Python transformation.
    def ImageToTensorImageNet(self, img):
        img_size = self.img_size
        # Set up the transformation. For input consistency.
        transformation = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
        img_t = transformation(img)
        # Prepare batch.
        tensor_t = torch.unsqueeze(img_t,0)
        return tensor_t
*/
const torch::Tensor AlexPatch::ImageToTensorImagenet(cv::Mat img_in) {
  // Referenced code from here
  // https://gitmemory.com/issue/pytorch/pytorch/14273/550272489
  // Starting with raw image loaded with cv::imread(path_to_img).
  // Convert color ordering to RGB.
  cv::Mat img;
  cv::cvtColor(img_in, img, cv::COLOR_BGR2RGB);  // Check order.

  // Resize to 227,227,3.
  cv::Size rsz = {227, 227};
  cv::resize(img, img, rsz, cv::INTER_CUBIC);
  // Convert to float and put pixels in range [0,1]
  img.convertTo(img, CV_32FC3, 1.0 / 255.0);

  // Convert to tensor.
  torch::Tensor img_tensor =
      torch::from_blob(img.data, {1, img.rows, img.cols, 3});
  img_tensor = img_tensor.to(torch::kFloat);
  img_tensor = img_tensor.permute({0, 3, 1, 2});

  //  Normalize data
  img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
  img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
  img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

  // std::cout << img_tensor << "\n";

  img_tensor = img_tensor.sub(0.0).div(1.0);

  return img_tensor;
}

}  // namespace alex_patch
