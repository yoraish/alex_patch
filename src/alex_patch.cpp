
#include "alex_patch.hpp"

AlexPatch::AlexPatch(/* args */)
{
}

AlexPatch::~AlexPatch()
{
}

float AlexPatch::GetSimilarityBwPatches(/*cv::patch_1, cv::patch_2*/){
    std::cout<< "Called GetSimilarityBwPatches" << " Ready to develop." << std::endl;
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
}
