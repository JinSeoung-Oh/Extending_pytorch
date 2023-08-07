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
print(torch.ops.my_ops.warp_perspective(torch.randn(32, 32), torch.rand(3, 3)))   # usage example

### Using the Operator with Tracing
def compute(x,y,z):
  x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
  return x.matmul(y) + torch.relu(z)

inputs = [torch.randn(4,8), torch.randn(8,5), torch.randn(4,5)]
trace = torch.jit.trace(compute, inputs) # it will forward to our implementation to record the sequence of operations that occur as the inputs flow through it
print(trace.graph) # producing graph


### Using Custom Operator with Script
torch.ops.load_library("libwarp_perspective.so")

@torch.jit.script
def compute(x,y):
  if bool(x[0][0] == 42):
    z = 5
  else:
    z = 10
  x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
  return x.matmul(y) + z

compute.graph


### Using the TorchScript Custom Operator in C++
### main.cpp

#include <torch/script.h> // one-stop header
#include <iostream>
#include <memory>

int main(int argc, const char*argv[]){
  if (argc !=2){
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
   }
   torch::jit::script::Moudle module = torch.::jit::load(argv[1]);
   std::vector<torch::jit::IValue> inputs;
   inputs.push_back(torch::randn({4,8}));
   inputs.push_back(torch::randn({8,5}));
   torch::Tensor output = module.forward(std::move(inputs)).toTensor();

   std::cout << output << std::endl;
}


### CmakeLists.txt
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(example_app)

find_package(Torch REQUIRED)

add_executable(example_app main.cpp)
target_link_libraries(example_app "${TORCH_LIBRARIES}")
target_compile_features(example_app PRIVATE cxx_range_for)

