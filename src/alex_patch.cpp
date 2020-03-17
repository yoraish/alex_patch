
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

void AlexPatch::SetupModel(){
}

AlexPatch::~AlexPatch()
{
}



float AlexPatch::GetSimilarityBwPatches(cv::Mat patch_1,  cv::Mat patch_2){
    std::cout<< "Called GetSimilarityBwPatches" << std::endl;
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
    std::cout << "Size of diff tensor " << diff.sizes()[1]<<"\n"; 

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
    std::cout<<"Converting image to tensor.\n";
    // Resize to 227,227,3.
    cv::resize(img, img, cv::Size(227, 227), cv::INTER_CUBIC);


    cv::Mat mean;
    cv::Mat stddev;
    cv::meanStdDev(img, mean, stddev);

    cv::imshow("resized", img);
    cv::waitKey();


    // Convert to tensor.
    torch::Tensor img_tensor = torch::from_blob(img.data, { 1, img.rows, img.cols, 3 }, torch::kByte);
    img_tensor = img_tensor.permute({0,3,1,2});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.sub(0.485).div(0.225);
    return img_tensor;


}
