
#include<vector>
#include<string>
#include "alex_patch.hpp"

AlexPatch::AlexPatch(/* args */)
{

    std::cout<< "New AlexPath Instance\n";
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load("/home/racecar/rrg/alex_patch/models/alexnet_fc_torchscript.pt");
    }
    catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    }
    std::cout << "ok\n";
}



AlexPatch::~AlexPatch()
{
}



float AlexPatch::GetSimilarityBwPatches(cv::Mat patch_1,  cv::Mat patch_2){
    // Work on first image.
    // Transform the first image to comply with the AlexNet input requierments.
    torch::Tensor transformed_patch_1 = ImageToTensorImagenet(patch_1);
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs_1;
    inputs_1.push_back(transformed_patch_1);

    // Execute the model and turn its output into a tensor.
    torch::Tensor desc_1 = model.forward(inputs_1).toTensor();

    // Work on second image.
    // Transform the first image to comply with the AlexNet input requierments.
    torch::Tensor transformed_patch_2 = ImageToTensorImagenet(patch_2);
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs_2;
    inputs_2.push_back(transformed_patch_2);

    // Execute the model and turn its output into a tensor.
    torch::Tensor desc_2 = model.forward(inputs_2).toTensor();   
    torch::Tensor diff = desc_1 - desc_2; 

    // torch::Tensor diff = torch::randn({1, 4});

    float similarity_value = 0;
    for (int i = 0; i < diff.sizes()[1]; i++){
        auto entry = diff[0][i].item<float>();

        similarity_value+= std::pow(entry,2);
    }
    similarity_value = std::pow(similarity_value,0.5);

    return similarity_value;
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
torch::Tensor AlexPatch::ImageToTensorImagenet(cv::Mat img){
    // Resize to 227,227,3.
    cv::resize(img, img, cv::Size(227, 227), cv::INTER_CUBIC);


    cv::Mat mean;
    cv::Mat stddev;
    cv::meanStdDev(img, mean, stddev);

    if (false){
    cv::imshow("resized", img);
    cv::waitKey();
    }

    // Convert to tensor.
    torch::Tensor img_tensor = torch::from_blob(img.data, { 1, img.rows, img.cols, 3 }, torch::kByte);
    img_tensor = img_tensor.permute({0,3,1,2});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.sub(0.485).div(0.225);
    return img_tensor;


}


float AlexPatch::EvaluateSequence(std::string seq_name){
    // We start with a sequence name.
    // NOTE(yorai): Hard code all the corresponding frame names in this sequence. For time efficiency in devlopment.
    // Change the number of frames to one gotten from folder.
    std::vector<std::string> frame_names;
    for (int frame_number = 0; frame_number < 152; frame_number ++){
        std::string frame_number_str = std::to_string(frame_number);
        std::string frame_name = std::string( 6-frame_number_str.size(), '0').append( frame_number_str);
        frame_names.push_back(frame_name);
    }

    std::size_t collision_counter = 0;

    // TODO(yorai): Merge this for loop with the one above.
    for (int frame_name_ix = 0; frame_name_ix < frame_names.size()-1; frame_name_ix ++){
        // Get the objects in the current frame.
        const std::string frame_name_current = frame_names[frame_name_ix];
        std::vector<cv::String> obj_paths_current;
        std::string path_to_curr_frame_folder = "/home/racecar/rrg/alex_patch/media/"+seq_name+"/"+frame_name_current+"/*.png";
        cv::glob(path_to_curr_frame_folder, obj_paths_current, false);

        // Get the object paths in the next frame.
        const std::string frame_name_next = frame_names[frame_name_ix+1];
        std::vector<cv::String> obj_paths_next;
        std::string path_to_next_frame_folder = "/home/racecar/rrg/alex_patch/media/"+seq_name+"/"+frame_name_next+"/*.png";
        cv::glob(path_to_next_frame_folder, obj_paths_next, false); 

        // Get similarity value between each object in current, and itself in next frame, if exists.
        // Also, similarity value between each object  current, and not itself in next frame, if not empty.

        for (auto obj_path_curr : obj_paths_current){
            // Object with itself in next frame. If available.
            std::string path_to_obj_in_next = obj_paths_next[0];
            path_to_obj_in_next = path_to_obj_in_next.substr(0, path_to_obj_in_next.find_last_of("/")) + obj_path_curr.substr(obj_path_curr.find_last_of("/"), obj_path_curr.size());

            // Check that the next one exists.

            if (std::find(obj_paths_next.begin(), obj_paths_next.end(), path_to_obj_in_next) == obj_paths_next.end()){
                std::cout<<"Not found in next, continuing.\n";
                continue;
            }

            // Get the similarity value between object and itself in the future.
            const float simi_value_same = this->GetSimilarityBwPatches(cv::imread(obj_path_curr), cv::imread(path_to_obj_in_next));
            std::cout << obj_path_curr << " <=> " << path_to_obj_in_next << "\n";
            std::cout<<simi_value_same<<"\n";

            // Similarity value between current object and all others in the next frame.
            float simi_value_diff;
            for (auto obj_path_next : obj_paths_next){
                // We do not want to compare to same object again.
                if (obj_path_next == path_to_obj_in_next){
                    continue;
                }

            simi_value_diff = this->GetSimilarityBwPatches(cv::imread(obj_path_curr), cv::imread(obj_path_next));
            std::cout << obj_path_curr << " <=> " << obj_path_next << "\n";
            std::cout<<simi_value_diff<<"\n";
            
            // If the value between different objects is lower than the same object in consecutive frames, we have a collision.

            if (simi_value_diff <= simi_value_same){
                std::cout<<"COLLISION\n";
                collision_counter ++;
                }
            }
        }
    }
    std::cout << "OVERALL COLLISIONS " << collision_counter << "\n";
    return collisions_counter;

}

