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

#include "alex_patch/alex_patch.hpp"

float EvaluateSequence(std::string seq_name, alex_patch::AlexPatch alex_patch) {
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
      cv::Mat d1;
      cv::Mat d2;
      float simi_value_same = alex_patch.PatchDistanceL2(
          cv::imread(obj_path_curr), cv::imread(path_to_obj_in_next), &d1, &d2);
      // std::cout << obj_path_curr.substr(42, obj_path_curr.size()) << " <=> "
      //           << path_to_obj_in_next.substr(42, path_to_obj_in_next.size())
      //           << "\n";
      // std::cout << simi_value_same << "\n";

      // Similarity value between current object and all others in the next
      // frame.
      float simi_value_diff;
      for (auto obj_path_next : obj_paths_next) {
        // We do not want to compare to same object again.
        if (obj_path_next == path_to_obj_in_next) {
          continue;
        }
        cv::Mat d11;
        cv::Mat d22;
        simi_value_diff = alex_patch.PatchDistanceL2(
            cv::imread(obj_path_curr), cv::imread(obj_path_next), &d11, &d22);
        // std::cout << obj_path_curr.substr(42,obj_path_curr.size()) << " <=> "
        // << obj_path_next.substr(42,obj_path_next.size()) << "\n";
        // std::cout<<simi_value_diff<<" >> " <<simi_value_same <<"\n\n";

        // If the value between different objects is lower than the same object
        // in consecutive frames, we have a collision.

        if (simi_value_diff <= simi_value_same) {
          // std::cout << obj_path_curr.substr(42, obj_path_curr.size()) << "
          // <=> "
          //           << obj_path_next.substr(42, path_to_obj_in_next.size())
          //           << "\n";
          // std::cout << simi_value_diff << " >> " << simi_value_same <<
          // "\n\n";
          std::cout << "*******************COLLISION**********\n";
          collision_counter++;
        }
      }
    }
  }
  std::cout << "OVERALL COLLISIONS " << collision_counter << "\n";
  return collision_counter;
}

int main() {
  // Instance of AlexPatch.
  const std::string model_path =
      "/home/racecar/rrg/alex_patch/models/alexnet_fc_torchscript.pt";
  alex_patch::AlexPatch my_alex_patch(model_path);

  // Patch image (cv::Mat) from path.
  std::string patch_1_path = "../../media/0000/000000/0.png";
  cv::Mat patch1 = cv::imread(patch_1_path);

  std::string patch_2_path = "../../media/0000/000000/1.png";
  cv::Mat patch2 = cv::imread(patch_2_path);

  // Ask for similarity between two patches.
  cv::Mat desc2;
  cv::Mat desc1;
  float simi_score =
      my_alex_patch.PatchDistanceL2(patch1, patch2, &desc1, &desc2);
  std::cout << "Similarity Score is " << simi_score << "\n";

  EvaluateSequence("0000", my_alex_patch);
}
