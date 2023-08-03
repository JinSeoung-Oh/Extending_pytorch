### warp-perspective/
###   op.cpp
###   CmakeLists.txt

### write op.cpp

#include <opencv2/opencv.hpp>
#include <torch/script.h>

torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp){
  cv::Mat image_mat(/*rows=*/image.size(0),
                    /*cols=*/image.size(1),
                    /*type=*/CV_32FC1,
                    /*data=*/image.data_ptr<float>());
  cv::Mat warp_mat(/*rows=*/warp.size(0),
                   /*cols=*/warp.size(1),
                   /*type=*/CV_32FC1,
                   /*data=*/warp.data_ptr<float>());
  cv::Mat output_mat;
  cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{8,8};
  torch::Tensor output = torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{8,8});
  return output.clone();
}

### Registering the custom op with torchscript (single function)
### 
TORCH_LIBRARY(my_ops, m) {
  m.def("warp_perspective", warp_perspective);
}


### building the custom operator
### building with CMake
### write CmakeLists.txt
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(warp_perspective)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_library(warp_perspective SHARED op.cpp)
# Enable C++14
target_compile_features(warp_perspective PRIVATE cxx_std_14)
target_link_libraries(warp_perspective "${TORCH_LIBRARIES}")
target_link_libraries(warp_perspective opencv_core opencv_imgproc)

# --> mkdir build
# --> cd build
# --> cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"

### Using the torchscript custom operator in python

import torch
torch.ops.load_library("build/libwarp_persepctive.so")
