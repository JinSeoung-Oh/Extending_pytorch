# It is sometimes useful to run backwards twice through backward graph, for example to compute higher-order gradients
# It takes an understanding of autograd and some care to support double backwards, however. 
# Functions that support performing backward a single time are not necessarily equipped to support double backward

# During forward, autograd does not record any the graph for any operations performed within the forward function. 
# When forward completes, the backward function of the custom function becomes the grad_fn of each of the forward’s outputs

# During backward, autograd records the computation graph used to compute the backward pass if create_graph is specified

## Saving the inputs
# It saves an input tensor for backward --> Double backward works automatically when autograd is able to record operation in the backward pass
import torch
import torchviz

class Square(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    # Save non-tensors and non-inputs/non-outputs directly on ctx
    ctx.save_for_backward(x)
    return x**2

  @staticmethod
  def backward(ctx, grad_out):
    # support double backward automatically if autograd is able to record the computations performed in backward
    x, = ctx.saved_tensors
    return grad_out * 2 * x

# Use double precision cuz finite differencing method magnifies errors
x = torch.rand(3,3, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Square.apply, x)
# Use gradcheck to verify second-order derivatives
torch.autograd.gradgradcheck(Square.apply, x)

# vis for the graph
x = torch.tensor(1., requries_grad=True).clone()
out = Square.apply(x)
grad_x, = torch.autograd.grad(out, x, create_graph=True)
torchviz.make_dot((grad_x, x, out), {"grad_x": grad_x, "x":x, "out":out})

## saving the outputs
# save an output instead of input
class Exp(torch.autograd.Funtion):
  @staticmethod
  def forward(ctx, x):
    result = torch.exp(x)
    ctx.save_for_backward(result)
    return result

  @staticmethod
  def backward(ctx, grad_out):
    result, = ctx.saved_tensors
    return result * grad_out

x = torch.tensor(1., requires_grad=True, dtype=torch.double).clone()
torch.autograd.gradcheck(Exp.apply, x)
torch.autograd.gradgradcheck(Exp.apply, x)

# vis for the graph
out = Exp.apply(x)
grad_x, = torch.augograd(out, x, create_graph=True)
torchviz.make_dot((grad_x, x, out), {"grad_x":grad_x, "x":x, "out":out})

## Saving intermediate result
# Intermediate results should not be directly saved and used in backward though. Because forward is performed in no-grad mode, 
# if an intermediate result of the forward pass is used to compute gradients in the backward pass the backward graph of 
# the gradients would not include the operations that computed the intermediate result
class Sinh(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    expx = torch.exp(x)
    expnegx = torch.exp(-x)
    ctx.save_for_backward(expx, expnegx)
    return (expx - expnegx) / 2, expx, expnegx

  @staticmethod
  def backward(ctx, grad_out, _grad_out_exp, _grad_out_negexp):
    expx, expnegx = ctx.saved_tensors
    grad_input = grad_out * (expx + enpnegx) /2
    grad_input += _grad_out_exp * expx
    grad_input -= _grad_out_negexp * expnegx
    return grad_input

def sinh(x):
  return Sinh.apply(x)[0]

x = torch.rand(3,3,requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(sinh, x)
torch.autograd.gradgradcheck(sinh, x)

# vis
out = sinh(x)
grad_x, = torch.autograd.grad(out.sum(), x, create_graph=True)
torchviz.make_dot((grad_x, x, out), params={"grad_x": grad_x, "x": x, "out":out})

# exp and expnegx don't require grad. So grad_x would not even have a backward graph

## When Backward is not Tracked
# An example when it may not be possible for autograd to track gradients for a functions backward at all

def cube_forward(x):
    return x**3

def cube_backward(grad_out, x):
    return grad_out * 3 * x**2

def cube_backward_backward(grad_out, sav_grad_out, x):
    return grad_out * sav_grad_out * 6 * x

def cube_backward_backward_grad_out(grad_out, x):
    return grad_out * 3 * x**2

class Cube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return cube_forward(x)

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return CubeBackward.apply(grad_out, x)

class CubeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, x):
        ctx.save_for_backward(x, grad_out)
        return cube_backward(grad_out, x)

    @staticmethod
    def backward(ctx, grad_out):
        x, sav_grad_out = ctx.saved_tensors
        dx = cube_backward_backward(grad_out, sav_grad_out, x)
        dgrad_out = cube_backward_backward_grad_out(grad_out, x)
        return dgrad_out, dx

x = torch.tensor(2., requires_grad=True, dtype=torch.double)

torch.autograd.gradcheck(Cube.apply, x)
torch.autograd.gradgradcheck(Cube.apply, x)

out = Cube.apply(x)
grad_x, = torch.autograd.grad(out, x, create_graph=True)
torchviz.make_dot((grad_x, x, out), params={"grad_x": grad_x, "x": x, "out": out})
