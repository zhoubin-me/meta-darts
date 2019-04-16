import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from torch.distributions import Categorical
from MobileNetV2 import InvertedResidual

class MixedOp(nn.Module):

  def __init__(self, C_in, C_out, stride, binarize=False):
    super(MixedOp, self).__init__()
    self.binarize = binarize
    self._alpha = nn.Parameter(torch.ones(len(PRIMITIVES)))
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C_in, C_out, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)



  def forward(self, x):
    dist = Categorical(logits=self._alpha)
    if self.binarize:
      idxs = torch.multinomial(dist.probs, 2)
      probs_scale = dist.probs[idxs].sum()
      out = self._ops[idxs[0]](x) * dist.probs[idxs[0]] + self._ops[idxs[1]](x) * dist.probs[idxs[1]]
      return out / probs_scale
    else:
      return sum(w * op(x) for w, op in zip(dist.probs, self._ops))





class Network(nn.Module):

  def __init__(self, C, num_classes, criterion, layers=20, binarize=True):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion

    self.stem = nn.Sequential(
      nn.Conv2d(3, 32, 3, padding=1, bias=False),
      nn.BatchNorm2d(32),
      InvertedResidual(32, 16, 1, 1),
    )


    self.cells = nn.ModuleList()
    self.reduce_layers = [1, 3, 7, 9]
    C_curr = 16

    for i in range(layers):
      if i in self.reduce_layers:
        self.cells.append(MixedOp(C_curr, C_curr * 2, stride=2, binarize=binarize))
        C_curr *= 2
      else:
        self.cells.append(MixedOp(C_curr, C_curr, stride=1))

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_curr, num_classes)

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
      x.data.copy_(y.data)
    return model_new

  def forward(self, input):

    x = self.stem(input)
    for i, cell in enumerate(self.cells):
      x = cell(x)
    out = self.global_pooling(x)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)


  def arch_parameters(self):
    alphas = torch.cat(tuple([m._dist.logits for m in self.cells]), dim=-1)
    return alphas

  # def genotype(self):
  #
  #   def _parse(weights):
  #     gene = []
  #     n = 2
  #     start = 0
  #     for i in range(self._steps):
  #       end = start + n
  #       W = weights[start:end].copy()
  #       edges = sorted(range(i + 2),
  #                      key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
  #       for j in edges:
  #         k_best = None
  #         for k in range(len(W[j])):
  #           if k != PRIMITIVES.index('none'):
  #             if k_best is None or W[j][k] > W[j][k_best]:
  #               k_best = k
  #         gene.append((PRIMITIVES[k_best], j))
  #       start = end
  #       n += 1
  #     return gene
  #
  #   gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
  #   gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
  #
  #   concat = range(2 + self._steps - self._multiplier, self._steps + 2)
  #   genotype = Genotype(
  #     normal=gene_normal, normal_concat=concat,
  #     reduce=gene_reduce, reduce_concat=concat
  #   )
  #   return genotype

if __name__ == '__main__':
  model = Network(32, 10, None, 20, True)
  optimizer = torch.optim.SGD(model.parameters(), 0.1)
  model.cuda()
  for _ in range(10):
    x = torch.rand(2, 3, 224, 224)
    y = model(x.cuda())
    loss = y.sum().norm()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)
