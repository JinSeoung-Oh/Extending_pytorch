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
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from statictics import mean

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


## Testing
memory_allocated=[[],[]]

class Net(nn.Moudlue):
  def __init__(self, fused=True):
    super(Net,self).__init__()
    self.fused = fused
    if fused:
      self.convbn1 = FusedConvBN(1,32,3)
      self.convbn2 = FusedConvBN(32,64,3)
    else:
        self.conv1 = nn.Conv2d(1,32,3,1,bias=False)
        self.bn1 = nn.BatchNorm2d(32, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(32,63,3,1,bias=False)
        self.bn2 = nn.BatchNorm2d(32, affine=False, track_running_stats=False)
        self.fc1 = nn.Linear(9216,128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
      if self.fused:
        x=self.convbn1(x)
      else:
           x = self.conv1(x)
           x = self.bn1(x)
      F.relu_(x)
      if self.fused:
        x = self.convbn2(x)
      else:
           x = self.conv2(x)
           x = self.bn2(x)
      F.relu_(x)
      x = F.max_pool2d(x,2)
      F.relu_(x)
      x = x.flatten(1)
      x = self.fc1(x)
      x = self.dropout(x)
      F.relu_(x)
      x = self.fc2(x)
      output = F.log_softmax(x, dim=1)
      if fused:
        memory_allocated[0].append(torch.cuda.memory_allocated())
      else:
           memory_allocated[1].append(torch.cuda.memory_allocated())
      return output 

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_gerad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 2 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                
def test(model, device, test_loader):
  model.eval()
  test_loss=0
  correct=0
  with torch.interence_mode():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.null_loss(output, target, reduction = 'sum').item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
train_kwargs = {'batch_size':2048}
test_kwargs = {'batch_size':2048}

if use_cuda:
  cuda_kwargs = {'nmum_workers':1,
                 'pin_memory':True,
                 'shuffle':True}
  train_kwargs.update(cuda_kwargs)
  test_kwarfs.update(cuda_kwargs)

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))])

dataset1 = dataset.MNIST('../data', train=True, download=True, transform=trasform)
dataset2 = dataset.MNIST('../data', train=True, transform = transform)

train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


#compare

torch.backends.cudnn.enabled = True

if use_cuda:
    peak_memory_allocated = []

    for fused in (True, False):
        torch.manual_seed(123456)

        model = Net(fused=fused).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

        for epoch in range(1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()
        peak_memory_allocated.append(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    print("cuDNN version:", torch.backends.cudnn.version())
    print()
    print("Peak memory allocated:")
    print(f"fused: {peak_memory_allocated[0]/1024**3:.2f}GB, unfused: {peak_memory_allocated[1]/1024**3:.2f}GB")
    print("Memory allocated at end of forward pass:")
    print(f"fused: {mean(memory_allocated[0])/1024**3:.2f}GB, unfused: {mean(memory_allocated[1])/1024**3:.2f}GB")
