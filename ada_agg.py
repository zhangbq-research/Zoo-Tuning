import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
import math
import random

class AdaAggLayer(nn.Module):
    r"""Applies an adaptive aggregate conv2d to the incoming data:.`
    """
    __constants__ = ['in_planes', 'out_planes', 'kernel_size', 'experts']

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, experts=5, align=True, lite=False, replace_pro=0.2):
        super(AdaAggLayer, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.experts = experts
        self.align = align
        self.lite = lite
        self.replace_pro = replace_pro
        self.m = 0.1
        self.weight = nn.Parameter(torch.randn(experts, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=False)
        self.s_weight = nn.Parameter(torch.Tensor(out_planes, in_planes // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(experts, out_planes))
            self.s_bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
            self.s_bias = None

            # channel-wise align
        if self.align and self.kernel_size > 1:
        # if self.align:
            align_conv = torch.zeros(self.experts * out_planes, out_planes, 1, 1)

            for i in range(self.experts):
                for j in range(self.out_planes):
                    align_conv[i * self.out_planes + j, j, 0, 0] = 1.0

            self.align_conv = nn.Parameter(align_conv, requires_grad=True)
        else:
            self.align = False

        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.experts):
            nn.init.kaiming_uniform_(self.weight[i])
        nn.init.kaiming_uniform_(self.s_weight, a=math.sqrt(5))
        if self.s_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.s_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, xs):
        x, expert_i = xs
        if expert_i == 0:
            if self.s_bias is not None:
                y = F.conv2d(x, weight=self.s_weight, bias=self.s_bias, stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, groups=self.groups)
            else:
                y = F.conv2d(x, weight=self.s_weight, bias=None, stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, groups=self.groups)
        else:
            expert_i = expert_i -1
            # channel-wise align
            if random.random() < self.replace_pro:
                if self.align:
                    weight = rearrange(self.s_weight.unsqueeze(0).expand(self.experts, -1, -1, -1, -1), '(d e) o i j k->d (e o) i (j k)', d=1)
                    # weight = self.weight.view(1, self.experts * self.out_planes, self.in_planes, self.kernel_size * self.kernel_size)
                    weight = F.conv2d(weight, weight=self.align_conv, bias=None, stride=1, padding=0, dilation=1, groups=self.experts)
                    weight = rearrange(weight, 'd (e o) i (j k)->(d e) o i j k', e=self.experts, j=self.kernel_size)
                else:
                    weight = self.s_weight.unsqueeze(0).expand(self.experts, -1, -1, -1, -1)

                if self.s_bias is not None:
                    bias = self.s_bias
                    y = F.conv2d(x, weight=weight[expert_i], bias=bias[expert_i], stride=self.stride, padding=self.padding,
                                      dilation=self.dilation, groups=self.groups)
                else:
                    y = F.conv2d(x, weight=weight[expert_i], bias=None, stride=self.stride, padding=self.padding,
                                      dilation=self.dilation, groups=self.groups)
            else:
                if self.bias is not None:
                    y = F.conv2d(x, weight=self.weight[expert_i], bias=self.bias[expert_i], stride=self.stride, padding=self.padding,
                                      dilation=self.dilation, groups=self.groups)
                else:
                    y = F.conv2d(x, weight=self.weight[expert_i], bias=None, stride=self.stride, padding=self.padding,
                                      dilation=self.dilation, groups=self.groups)

        return y
