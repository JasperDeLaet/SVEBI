import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg

'''
This file implements the spiking VGG19 (SVGG19) model. 
The implementation of the spiking VGG19 is based on the non-spiking VGG models in which we overwrite the forward pass to
include spiking behaviour. In this we preserve the initial topology of the VGG19 such that we can use its weights as
starting point to train from. 

The surrogate gradient function has been inspired by the surrogate gradient function of the work: 
"Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch"
Github: https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time?tab=readme-ov-file
Preprint: https://arxiv.org/abs/2010.01729
'''


# Custom forward and backward function to implement surrogate gradient
class Surrogate_BP_Function(torch.autograd.Function):
    # Custom spiking function for SNN with surrogate gradient for backward pass
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


class svgg19(vgg.VGG):
    def __init__(self, num_steps=25, leak_mem=0.95, img_size=224, num_cls=50):
        super(svgg19, self).__init__(vgg.make_layers([64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"], False))
        self.num_steps = num_steps
        self.num_cls = num_cls
        self.batch_num = self.num_steps
        self.leak_mem = leak_mem
        self.spike_fn = Surrogate_BP_Function.apply
        self.img_size = img_size

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, x):
        batch_size = x.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv8 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv9 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv10 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv11 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv12 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv13 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv14 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv15 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv16 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7, mem_conv8, mem_conv9, mem_conv10, mem_conv11, mem_conv12, mem_conv13, mem_conv14, mem_conv15, mem_conv16]

        mem_fc1 = torch.zeros(batch_size, 4096).cuda()
        mem_fc2 = torch.zeros(batch_size, 4096).cuda()
        mem_fc3 = torch.zeros(batch_size, self.num_cls).cuda()

        mem_fc_list = [mem_fc1, mem_fc2, mem_fc3]

        for t in range(self.num_steps):
            out_prev = x

            conv_layer_counter = 0
            for module_idx, module in enumerate(self.features):
                if isinstance(module, nn.Conv2d):
                  out_prev = module(out_prev)
                elif isinstance(module, nn.ReLU):
                  mem_conv_list[conv_layer_counter] = self.leak_mem * mem_conv_list[conv_layer_counter] + out_prev
                  mem_thr = (mem_conv_list[conv_layer_counter] / self.features[module_idx-1].threshold) - 1.0
                  out = self.spike_fn(mem_thr)
                  rst = torch.zeros_like(mem_conv_list[conv_layer_counter]).cuda()
                  rst[mem_thr > 0] = self.features[module_idx-1].threshold
                  mem_conv_list[conv_layer_counter] = mem_conv_list[conv_layer_counter] - rst
                  out_prev = out.clone()
                  conv_layer_counter += 1
                elif isinstance(module, nn.MaxPool2d):
                  out = module(out_prev)
                  out_prev = out.clone()

            out = self.avgpool(out_prev)
            out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            fc_counter = 0
            for module_idx, module in enumerate(self.classifier):
                  if isinstance(module, nn.Linear):
                    out_prev = module(out_prev)
                  elif isinstance(module, nn.ReLU):
                      mem_fc_list[fc_counter] = self.leak_mem * mem_fc_list[fc_counter] + out_prev
                      mem_thr = (mem_fc_list[fc_counter] / self.classifier[module_idx-1].threshold) - 1.0
                      out = self.spike_fn(mem_thr)
                      rst = torch.zeros_like(mem_fc_list[fc_counter]).cuda()
                      rst[mem_thr > 0] = self.classifier[module_idx-1].threshold
                      mem_fc_list[fc_counter] = mem_fc_list[fc_counter] - rst
                      out_prev = out.clone()
                      fc_counter += 1
                  elif isinstance(module, nn.Dropout):
                    out_prev = module(out_prev)

            mem_fc_list[-1] = mem_fc_list[-1] + out_prev

        out_voltage = mem_fc_list[-1] / self.num_steps

        return out_voltage
