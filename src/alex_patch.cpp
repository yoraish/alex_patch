
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

AlexPatch::AlexPatch(/* args */) {
  std::cout << "New AlexPath Instance\n";
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(
        "/home/racecar/rrg/alex_patch/models/alexnet_fc_torchscript.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  std::cout << "ok\n";
}

torch::Tensor AlexPatch::PatchToDescTensor(const cv::Mat& patch) {
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

  // TODO(yorai): Change the transform to cv::Mat to be in a function.
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
torch::Tensor AlexPatch::ImageToTensorImagenet(cv::Mat img) {
  // Referenced code from here
  // https://gitmemory.com/issue/pytorch/pytorch/14273/550272489
  // Starting with raw image loaded with cv::imread(path_to_img).
  // Convert color ordering to RGB.
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

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

float AlexPatch::EvaluateSequence(std::string seq_name) {
  // We start with a sequence name.
  // NOTE(yorai): Hard code all the corresponding frame names in this sequence.
  // For time efficiency in devlopment.
  // Change the number of frames to one gotten from folder.
  std::vector<std::string> frame_names;
  for (int frame_number = 0; frame_number < 152; frame_number++) {
    std::string frame_number_str = std::to_string(frame_number);
    std::string frame_name =
        std::string(6 - frame_number_str.size(), '0').append(frame_number_str);
    frame_names.push_back(frame_name);
  }

  std::size_t collision_counter = 0;

  // TODO(yorai): Merge this for loop with the one above.
  for (std::size_t frame_name_ix = 0; frame_name_ix < frame_names.size() - 1;
       frame_name_ix++) {
    // Get the objects in the current frame.
    const std::string frame_name_current = frame_names[frame_name_ix];
    std::vector<cv::String> obj_paths_current;
    std::string path_to_curr_frame_folder =
        "/home/racecar/rrg/alex_patch/media/" + seq_name + "/" +
        frame_name_current + "/*.png";
    cv::glob(path_to_curr_frame_folder, obj_paths_current, false);

    // Get the object paths in the next frame.
    const std::string frame_name_next = frame_names[frame_name_ix + 1];
    std::vector<cv::String> obj_paths_next;
    std::string path_to_next_frame_folder =
        "/home/racecar/rrg/alex_patch/media/" + seq_name + "/" +
        frame_name_next + "/*.png";
    cv::glob(path_to_next_frame_folder, obj_paths_next, false);

    // Get similarity value between each object in current, and itself in next
    // frame, if exists.
    // Also, similarity value between each object  current, and not itself in
    // next frame, if not empty.

    for (auto obj_path_curr : obj_paths_current) {
      // Object with itself in next frame. If available.
      std::string path_to_obj_in_next = obj_paths_next[0];
      path_to_obj_in_next =
          path_to_obj_in_next.substr(0, path_to_obj_in_next.find_last_of("/")) +
          obj_path_curr.substr(obj_path_curr.find_last_of("/"),
                               obj_path_curr.size());

      // Check that the next one exists.

      if (std::find(obj_paths_next.begin(), obj_paths_next.end(),
                    path_to_obj_in_next) == obj_paths_next.end()) {
        std::cout << "Not found in next, continuing.\n";
        continue;
      }

      // Get the similarity value between object and itself in the future.
      const float simi_value_same = this->PatchDistanceL2(
          cv::imread(obj_path_curr), cv::imread(path_to_obj_in_next));
      // std::cout << obj_path_curr.substr(42,obj_path_curr.size()) << " <=> "
      // << path_to_obj_in_next.substr(42,path_to_obj_in_next.size()) << "\n";
      // std::cout<<simi_value_same<<"\n";

      // Similarity value between current object and all others in the next
      // frame.
      float simi_value_diff;
      for (auto obj_path_next : obj_paths_next) {
        // We do not want to compare to same object again.
        if (obj_path_next == path_to_obj_in_next) {
          continue;
        }

        simi_value_diff = this->PatchDistanceL2(cv::imread(obj_path_curr),
                                                cv::imread(obj_path_next));
        // std::cout << obj_path_curr.substr(42,obj_path_curr.size()) << " <=> "
        // << obj_path_next.substr(42,obj_path_next.size()) << "\n";
        // std::cout<<simi_value_diff<<" >> " <<simi_value_same <<"\n\n";

        // If the value between different objects is lower than the same object
        // in consecutive frames, we have a collision.

        if (simi_value_diff <= simi_value_same) {
          std::cout << obj_path_curr.substr(42, obj_path_curr.size()) << " <=> "
                    << obj_path_next.substr(42, path_to_obj_in_next.size())
                    << "\n";
          std::cout << simi_value_diff << " >> " << simi_value_same << "\n\n";
          std::cout << "*******************COLLISION**********\n";
          collision_counter++;
        }
      }
    }
  }
  std::cout << "OVERALL COLLISIONS " << collision_counter << "\n";
  return collision_counter;
}
