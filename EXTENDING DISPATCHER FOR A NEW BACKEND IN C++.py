### adding new backend
### Get a dispatch key for your backend

## Set dispatch key in TensorImpl constructor

TensorImPl(
  Storage&& starage,
  DispatchKeySet ks,
  const caffe2::TypeMeta data_type);

DispatchKeySet ks = c10::DispatchKeySet{c10::DispatchKey::Privateuse1, c10::DispatchKey::AutogradPrivateUse1};

## TensorImpl class above assumes Tensor is backed by a storage like CPU/CUDA
## OpaqueTensorImpl for backends without a storage

### Get the full list of Pytorch operators
## see : build/aten/src/ATen/RegistrationDeclarations.h in pytorch

### Register Kernels for the new backend

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(<schema_my_op1>, &my_op1);
  m.impl(<schema_my_op2>, &my_op2);
  m.impl(<schema_my_op2_backward, &my_op2_backward);
}

## PyTorch operators can be classified into two categories
## 1. Ops that require registration: PyTorch native implementation for these ops is backend specific and thus it’s required to provide a kernel for customized backend. 
## Otherwise calling such op on the customized backend will error out.
## In RegistrationDeclarations.h 
## these operators have dispatch set to True and default set to False in the metadata found in their accompanying comments.

## 2. Registration is optional: backend extenders can skip registering to these ops without sacrificing any support
## However, if a backend extender wants to override the default kernel provided by PyTorch, 
## they can still register their customized kernel to their backend and the dispatcher will use it for your backend only
## In RegistrationDeclarations.h these operators have dispatch set to False or default set to True in the metadata found in their accompanying comments.

### Autograd support for the new backend
Tensor my_op1(const Tensor& self, const Tensor& other) {
  //call your backend-specific APIs to implement my_op so that
  // it matches PyTorch's native behavior
}
TORCH_LIBRARY_IMPL(aten, privateUse1, m) {
  m.impl(<schema_my_op1>, &my_op);
}

Tensor my_op2(const Tensor& self, const Tensor& other) {
  //call your backend-specific APIs to implement my_op so that
  // it matches PyTorch's native behavior
}

Tensor my_op2_backward(const Tensor& self, const Tensor& other) {
  // call your backend-specific APIs to implement my_op2_backward so that
  // it matches PyTorch's native behavior
}

TORCH_LIBRARY_IMPL(aten, privateUse1, m) {
  m.impl(<schema_my_op2>, &my_op);
  m.impl(<schema_my_op2_backward>, &my_op2_backward);
}  

## In a few rare cases, PyTorch’s gradient formula for certain operators may have assumptions that don’t generalize for all backends. In those cases backend extenders can optionally 
## override PyTorch Autograd layer by registering a kernel from torch::autograd::Function to the corresponding dispatch key

class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
  public:
  static Tensor forward(AutogradContext *ctx, torch::Tensor self, torch::Tensor other) {
    at::AutoNonVariableTypeMode g;
    return myadd(self, other);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto grad_output = grad_outputs[0];
    return {grad_output, grad_output};
  }
};

Tensor myadd_autograd(const Tensor& self, const Tensor& other) {
  return MyAddFunction::apply(self, other)[0];
}

// Register the autograd kernel to AutogradPrivateUse1
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl(<myadd_schema>, &myadd_autograd);
}

// Register the inference kernel to PrivateUse1
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(<myadd_schema>, &myadd);
}

## build an extension

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='torch_xla',
    ext_modules=[
        CppExtension(
            '_XLAC',
            torch_xla_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args + \
                [make_relative_rpath('torch_xla/lib')],
        ),
    ],
    cmdclass={
        'build_ext': Build,  # Build is a derived class of BuildExtension
    }
    # more configs...
)

