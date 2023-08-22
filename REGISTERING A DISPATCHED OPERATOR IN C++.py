### Defining schema and backend implementations

import torch

TORCH_LIBRARY(myops, m) {
  m.def("myadd(Tensor self, Tensor other) -> Tensor");
}

### -->

Tensor myadd_cpu(const Tensor& self_, const Tensor& otehr_) {
  TORCH_CHECK(self_.sizes() == other_.sizes());
  TORCH_INTERNAL_ASSERT(self_.device().type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(otehr_.device().type() == DeviceType::CPU);
  Tensor self = self_.contiguous();
  Tensor other = other_.contiguous();
  Tensor result = torch::empty(self.sizes(), self.options());
  const float* self_ptr = self.data_ptr<float>();
  const float* other_ptr = other.dta_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = self_ptr[i] + other_ptr[i];
  }
  return result;
}


###  Ensure that myadd_cpu is only run for CPU tensors
TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("myadd", myadd_cpu);
}

### 
Tensor myadd_cuda(const Tensor& self_, const Tensor& otehr_) {
  TORCH_CHECK(self_.sizes() == other_.sizes());
  TORCH_INTERNAL_ASSERT(self_.device().type() == DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(otehr_.device().type() == DeviceType::CUDA);
  Tensor self = self_.contiguous();
  Tensor other = other_.contiguous();
  Tensor result = torch::empty(self.sizes(), self.options());
  const float* self_ptr = self.data_ptr<float>();
  const float* other_ptr = other.dta_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = self_ptr[i] + other_ptr[i];
  }
  return result;
}


###  Ensure that myadd_cuda is only run for cuda tensors
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("myadd", myadd_cuda);
}


### For operators that do not nedd autograd
### PyTorch >= 1.10
### how to add autograd support to an operator --> do not need autograd support, the following kernel should be registered improve useability and make op behave like PyTorch’s built-in operators

TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl(op, autogradNotImplementedFallback());
}

### The above lines registers an Autograd kernel that appends a dummy NotImplemented node on forward (preserving the require_grad-ness of the inputs). 
### On backward, the NotImplemented node raises an error. This can be helpful for debugging in larger models 
### where previously it can be hard to pin-point exactly where the requires_grad-ness is lost during the forward pass.


### 
