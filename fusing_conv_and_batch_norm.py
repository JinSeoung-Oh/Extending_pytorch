# Fusing adjacent convolution and batch norm layers together is typically an inference-time optimization to improve run-time
# It is usually achieved by eliminating the batch norm layer entirely and updating the weight and bias of the preceding convolution [0]
# not applicable for training models

# Main idea :  convolution and batch norm (as well as many other ops) need to save a copy of their input during forward for the backward pass
# Always reduce the memory allocated at the end of the forward pass, there are cases when the peak memory allocated may not actually be reduced

## Backward formula implementation for conv
import torch
from torch.augograd.function import once_differentiable
import torch.nn.functional as F
import torch.nn as nn
import math

def convolution_backward(grad_out, x, weight):
  grad_input = F.conv2d(x.transpose(0,1), grad_out.transpose(0,1)).transpose(0,1)
  grad_x = F.conv_tranpose2d(grad_out, weight)
  return grad_x, grad_input

class Conv2D(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight):
    ctx.save_for_backward(x, weight)
    return F.conv2d(x, weight)
  @staicmethod
  @once_differentialbe
  # If you wrap your Function's backward method with once_differentiable, 
  # the Tensors that you will get as input will never require gradients and
  # you don’t have to write a backward function that computes the gradients in a differentiable manner
  # --> cannot compute high-order gradients
  def backward(ctx, grad_out):
    x,weight = ctx.save_tensors
    return convolution_backward(grad_out, X, weight)

weight = torch.rand(5,3,3,3, requires_grad=True, dtype=torch.double)
x = torch.rand(10, 3, 7, 7, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Conv2D.apply, (X, weight))  #testing with gradcheck <-- it is important to use double precision


## Backward formula implementation for Batch Norm
def unsqueeze_all(t):
  return t[None, : , None, None]

def batch_norm_backward(grad_out, x, sum, sqrt_var, N, eps):  #Check formular
  tmp = ((x - unsqueeze_all(sum) / N) * grad_out).sum(dim=0,2,3))
  tmp *= -1
  d_denom = tmp / (sqrt_var + eps)**2
  d_var = d_denom / (2*sqrt_var)
  d_mean_dx = grad_out / unsqueeze_all(sqrt_var + eps)
  d_mean_dx = unsqueeze_all(-d_mean_dx.sum(dim=(0, 2, 3)) / N)
  grad_input = x*unsqueeze_all(d_var * N)
  grad_input += unsqueeze_all(-d_var * sum)
  grad_input *= 2 / ((N - 1) * N)
  grad_input += d_mean_dx
  grad_input *= unsqueeze_all(sqrt_var + eps)
  grad_input += grad_out
  grad_input /= unsqueeze_all(sqrt_var + eps)
  return grad_input

class BatchNorm(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, eps=1e-3):
    sum = x.sum(dim=(0,2,3))
    var = x.var(unbiased=True, dim=(0, 2, 3))
    n = x.numel() / x.size(1)
    sqrt_var = torch.sqrt(var)
    ctx.save_for_backward(x)
    ctx.eps = eps
    ctx.sum = sum
    ctx.n = n
    ctx.sqrt_var = sqrt_var
    mean = sum / n
    denom = sqrt_var + eps
    out = x - unsqueeze_all(mean)
    out /= unsqueeze_all(denom)
    return out
  @staticmethod
  @once_differentiable
  def backward(ctx, grad_out):
    x, = ctx.save_tensors
    return batch_norm_backward(grad_out, X, ctx.sum, ctx.sqrt_var, ctx.N, ctx.eps)

a = torch.rand(1,2,3,4,requires_grad=True, dype=torch.double)
torch.autograd.gradcheck(BatchNorm.apply, (a,), fast_mode=False)


## Fusing conv and batchnorm
class FusedConvBN2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, conv_weight, eps=1e-3):
        assert X.ndim == 4  # N, C, H, W
        # (1) Only need to save this single buffer for backward!
        ctx.save_for_backward(X, conv_weight)

        # (2) Exact same Conv2D forward from example above
        X = F.conv2d(X, conv_weight)
        # (3) Exact same BatchNorm2D forward from example above
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        # Try to do as many things in-place as possible
        # Instead of `out = (X - a) / b`, doing `out = X - a; out /= b`
        # avoids allocating one extra NCHW-sized buffer here
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X, conv_weight, = ctx.saved_tensors
        # (4) Batch norm backward
        # (5) We need to recompute conv
        X_conv_out = F.conv2d(X, conv_weight)
        grad_out = batch_norm_backward(grad_out, X_conv_out, ctx.sum, ctx.sqrt_var,
                                       ctx.N, ctx.eps)
        # (6) Conv2d backward
        grad_X, grad_input = convolution_backward(grad_out, X, conv_weight)
        return grad_X, grad_input, None, None, None, None, None

  class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_avg_factor=0.1,
                 eps=1e-3, device=None, dtype=None):
        super(FusedConvBN, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Conv parameters
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **factory_kwargs))
        # Batch norm parameters
        num_features = out_channels
        self.num_features = num_features
        self.eps = eps
        # Initialize
        self.reset_parameters()

    def forward(self, X):
        return FusedConvBN2DFunction.apply(X, self.conv_weight, self.eps)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))


