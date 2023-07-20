import torch
from setuptools import setup, Extesion
from torch.utils import cpp_extension


# vanilla LLTM

class LLTM(torch.nn.Module):
  def __init__(self, input_features, state_size):
    super(LLTM, self).__init__()
    self.input_features = input_features
    self.state_size = state_size
    self.weight = torch.nn.Parameter(torch.empty(3*state_size, input_features + state_size))
    self.bias = torch.nn.Parameter(torch.empty(3*state_size))
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.state_size)
    for weight in self.parameters():
      weight.data.uniform(-stdv, +stdv)

  def forward(self, input, state):
    old_h, old_cell = state
    x = torch.cat([old_h, input], dim=1)
    gate_weights = F.linear(x, self.weights, self.bias)
    gates = gate_weights.chunk(3, dim=1)
    input_gate = torch.sigmoid(gates[0])
    output_gate = torch.sigmoid(gates[1])
    candidate_cell = F.elu(gates[2])
    new_cell = old_cell + candidate_cell * input_gate
    new_h = torch.tanh(new_cell)*output_gate

    return new_h, new_cell

### Writhing a C++ Extension (setup.py)
# building with setuptools / can add more like python version

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext':cpp_extension.BuildExtension})

# below code is equivalent with building with setuptools
Extension(name='lltm.cpp', sources=['lltm.cpp'], include_dirs=cpp_extension.include_paths(),language='c++')

### Writting with JIP compiling extensions
from torch.utils.cpp_extension import load
lltm_cpp = load(name='lltm.cpp', sources=['lltm.cpp'])


### Writhing the c++ operation  (lltm.cpp)

#include <torch/extension.h>
#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

## <torch/extension.h> is the one-stop header to include all the necessary PyTorch bits to write C++ extensions
## The ATen library, which is our primary API for tensor computation
## pybind11, which is how we create Python bindings for C++ code
## Headers that manage the details of interaction between ATen and pybind11
## Note that CUDA-11.5 nvcc will hit internal compiler error while parsing torch/extension.h on Windows

#include <ATen/ATen.h>
at::Tensor SigmoidAlphaBlendForwardCuda(....)

instead of
# includ <torch/extension.h>
torch::Tensor SigmoidAlphaBlendForwardCuda(...)


## Forwrad pass
# see class LLTM(torch.nn.Module)
#include <vector>
std::vector<at::Tensor> lltm_forward(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor old_h,
  torch::Tensor old_cell) {
auto x = torch::cat({old_h, input}, /*dim=*/1);

auto gate_weights = torch::addmm(bias, x, weights.transpose(0,1));
auto gates = gate_weights.chunk(3, /*dim=*/1);

auto input_gate = torch::sigmoid(gates[0]);
auto output_gate = torch::sigmoid(gates[1]);
auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

auto new_cell = old_cell + candidata_cell * input_gate;
auto new_h = torch::tanh(new_cell)*output_gate;

return {new_h,
        new_cell,
        input_gate,
        output_gate,
        candidata_cell,
        x,
        gate_weights];
       }

## Backward Pass
# The C++ extension API currently does not provide a way of automatically generating a backwards function

//tanh'(z) = 1-tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}


## Binding to Python
# Once you have your operation written in C++ and ATen, you can use pybind11 to bind your C++ functions or classes into Python
# TORCH_EXTENSION_NAME : The torch extension build will define it as the name you give your extension in the setup.py script
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}


## Using Extension
# Directory structure could look something like this:
# main/
#   lltm-extension/
#     lltm.cpp
#     setup.py



### After build setup.py(python setup.py install)
import math
import torch

# See this
import lltm_cpp

class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
