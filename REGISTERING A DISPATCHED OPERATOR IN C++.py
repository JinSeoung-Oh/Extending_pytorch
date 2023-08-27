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


### In-place or view ops
## To ensure correctness and best possible performance
## Register an ADInplaceOrView kernel in addition to the Autograd /  It is important to note that this ADInplaceOrView kernel should only be used with autogradNotImplementedFallback.
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
 m.impl(op, autogradNotImplementedFallback());
}
TORCH_LIBRARY_IMPL(myops, ADInplaceOrView, m) {
 m.impl(op, autogradNotImplementedInplaceOrViewFallback());
}

## The Autograd or ADInplaceOrView boxed kernels registered above rely on operator schema information in their logi

---------- CPU and CUDA implementations with autograd ---------------
### Adding autograd support
Tensor myadd(cost Tensor& self, const Tensor& other) {
 static auto op = torch::Dispatcher::singleton()
   .findSchemaOrThrow("myops:myadd", "")
   .typed<decltype(myadd)>();
 return op.call(self, other);
}

## findSchemaOrThrow :  name of the operator, and the overload name of the operator
## typed casts the dynamically typed handle into a statically typed handle
##  pass it decltype(myadd) since the type of the dispatching function is the same as the type of the underlying kernels registered to the dispatcher
## call(function) the operator handle with all of the arguments passed into the dispatching function
## with dispatch function, autograd kernel is

class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
  public:
   static Tensor forward(
      AutogradContect *ctx, torch::Tensor self, torch::Tensor other) {

      if (self.device().type() == DeviceType::CPU){
        return add_cpus(self, other);
      } else if (self.device().type() == DeviceType::CUDA) {
        return add_cuda(self, other);
      } else{
        TORCH_CHECK(0, 'Unsupported device", self.device().type());
      }
    at::AutoNonVariableTypeMode g;
    return myadd(self, other);
  }

   static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
     auto grad_output = grad_output = grad_outputs[0];
     return {grad_output, grad_output};
    }
};

Tensor myadd_autograd(const Tensor& self, const Tensor& other) {
  return MyAddFunction::apply(self, other)[0];
}

## at::AutoNonVariableTypeMode <-- Turn off autograd / to avoid infinite loop

## register 
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl("myadd", myadd_autograd);
}


## why dispatcher?
## can assemble all of the pieces of an operator (CPU, CUDA, Autograd) without having to write a single, centralized if statement that refers to all of them.
## It supports more dispatch keys --> see c10/core/DispatchKey.h in Pytorch
## The dispatcher implements support for boxed fallback functions, which are functions that can be implemented once and apply to all operators in the system

### Autocast
## Autocast wrapper for hypothetical custom matmul
#include <ATen/autocast_mode.h>
Tensor mymatmul_autocast(const Tensor& self, const Tensor& other) {
  c10::impl::ExcludeDispatchKeyGuard
no_autocast(c10::DispatchKey::Autocast);
  return mymatmul(at::autocast::cached_cast(at::kHalf, self),
                  at::autocast::cached_cast(at::kHalf, other));
}

TORCH_LIBRAYR(myops, AUtocast, m) {
  m.impl("mymatmul", mymatmul_autocast);
}

## cached_cast(kHalf, tensor) casts tensor to float16 if tensor is CUDA and float32, otherwise, it leaves tensor unchanged
## This ensures if the network calls mymatmul on any mixture of float16 and float32 CUDA tensors, mymatmul runs in float16

## ops with multiple floating-point tensor

#incluse <ATen/autocast_mode.h>

Tensor my_multiple_input_op_autocast(const Tensor& t0, const Tensor& t1){
  c10::impl::ExcludeDispatchKeyGuard
no_autocast(c10::DispatchKey::Autocat);
  auto exec_type = at::autocast::promote_type(at::kHalf, t0, t1);
  return my_multiple_input_op(at::autocast::cached_cast(exec_type, t0),
                              at::autocast::cached_cast(exce_type, t1));
}


Tensor myadd_autocast(const Tensor& self, const Tensor& other) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return myadd(at::autocast::cached_cast(<desired dtype>, self),
               at::autocast::cached_cast(<desired dtype>, other));
}

TORCH_LIBRARY_IMPL(myops, Autocast, m) {
  m.impl("myadd", myadd_autocast);
}
