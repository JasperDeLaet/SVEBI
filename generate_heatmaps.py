import torchvision.models as models
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import logging
import subprocess
import sys

import os.path

# import cv2
import torch
import numpy as np
from torchvision.models import vgg, vgg19_bn, vgg19
import torch.utils.data as torchdata
from PIL import Image
import logging

output_dir = "./project_antwerp/Interpretation_of_SNN/snn/svgg19/explanation/output/SNN_grad_per_class/"

# Logging
logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(output_dir + 'heatmap_generation.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def log_string(str):
    logger.info(str)
    print(str)


working_dir = "./project_antwerp/Interpretation_of_SNN/snn/svgg19/"

seed = 0


def normalize_numpy(batch):
    normalized_batch = np.empty_like(batch)

    for i in range(batch.shape[0]):
        A = batch[i]
        if np.max(A) == 0.0:
            normalized_batch[i] = A
        else:
            normalized_batch[i] = (A - np.min(A)) / (np.max(A) - np.min(A))

    return normalized_batch


def spike_function(input):
    out = torch.zeros_like(input).cuda()
    out[input > 0] = 1.0
    return out


class svgg19(vgg.VGG):
    def __init__(self, num_steps=25, leak_mem=0.95, img_size=224, num_cls=50):
        super(svgg19, self).__init__(vgg.make_layers(
            [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
            False))
        self.num_steps = num_steps
        self.num_cls = num_cls
        self.batch_num = self.num_steps
        self.leak_mem = leak_mem
        self.img_size = img_size

        self.return_spike_maps = True

        bias_flag = False

        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, x):
        with torch.no_grad():

            batch_size = x.size(0)
            mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
            mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
            mem_conv3 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
            mem_conv4 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
            mem_conv5 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
            mem_conv6 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
            mem_conv7 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
            mem_conv8 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
            mem_conv9 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
            mem_conv10 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
            mem_conv11 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
            mem_conv12 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
            mem_conv13 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
            mem_conv14 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
            mem_conv15 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
            mem_conv16 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
            mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7, mem_conv8,
                             mem_conv9, mem_conv10, mem_conv11, mem_conv12, mem_conv13, mem_conv14, mem_conv15,
                             mem_conv16]

            mem_fc1 = torch.zeros(batch_size, 4096).cuda()
            mem_fc2 = torch.zeros(batch_size, 4096).cuda()
            mem_fc3 = torch.zeros(batch_size, self.num_cls).cuda()

            mem_fc_list = [mem_fc1, mem_fc2, mem_fc3]

            # To collect all spikemaps and return
            all_spikemaps = [list() for conv in mem_conv_list]

            for t in range(self.num_steps):
                out_prev = x

                # Keep track of which actual CONV layer were currently pushing input through
                conv_layer_counter = 0
                for module_idx, module in enumerate(self.features):
                    if isinstance(module, nn.Conv2d):
                        out_prev = module(out_prev)
                    # Replace ReLU by the spiking activation function
                    elif isinstance(module, nn.ReLU):
                        mem_conv_list[conv_layer_counter] = self.leak_mem * mem_conv_list[conv_layer_counter] + out_prev
                        mem_thr = (mem_conv_list[conv_layer_counter] / self.features[module_idx - 1].threshold) - 1.0
                        out = spike_function(mem_thr)
                        rst = torch.zeros_like(mem_conv_list[conv_layer_counter]).cuda()
                        rst[mem_thr > 0] = self.features[module_idx - 1].threshold
                        mem_conv_list[conv_layer_counter] = mem_conv_list[conv_layer_counter] - rst
                        out_prev = out.clone()

                        # To collect
                        all_spikemaps[conv_layer_counter].append(out_prev.detach())

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
                        mem_thr = (mem_fc_list[fc_counter] / self.classifier[module_idx - 1].threshold) - 1.0
                        out = spike_function(mem_thr)
                        rst = torch.zeros_like(mem_fc_list[fc_counter]).cuda()
                        rst[mem_thr > 0] = self.classifier[module_idx - 1].threshold
                        mem_fc_list[fc_counter] = mem_fc_list[fc_counter] - rst
                        out_prev = out.clone()
                        fc_counter += 1
                    elif isinstance(module, nn.Dropout):
                        out_prev = module(out_prev)

                mem_fc_list[-1] = mem_fc_list[-1] + out_prev

            all_spikemaps = [torch.mean(torch.stack(s, dim=0).permute(1, 0, 2, 3, 4), dim=1) for s in all_spikemaps]

            out_voltage = mem_fc_list[-1] / self.num_steps
            if self.return_spike_maps:
                return out_voltage, all_spikemaps
            else:
                return out_voltage


model = svgg19(num_steps=25, leak_mem=0.95, img_size=224, num_cls=50)
model.classifier._modules['6'] = nn.Linear(4096, 50)
model.load_state_dict(torch.load(working_dir + '/output/svgg19_AwA2.pth'))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data_path = "./project_antwerp/Interpretation_of_SNN/data/Animals_with_Attributes2/JPEGImages/"
dataset = datasets.ImageFolder(data_path, transform=transform)

total_count = len(dataset.imgs)
train_count = int(0.8 * total_count)
test_count = total_count - train_count
train_dataset, test_dataset = torchdata.random_split(dataset, (train_count, test_count),
                                                     generator=torch.Generator().manual_seed(seed))

target_layers = [11, 13, 14]

amount_of_classes = 50

with open(working_dir + 'VEBI/output/lambda_w/lambda20_w.npy', 'rb') as file_add:
    w = pickle.load(file_add)

w_decoded = []

list_of_conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
for class_index in range(50):
    w_row = w[class_index]

    non_zero_w_indices = torch.nonzero(torch.from_numpy(w[class_index]))
    sub_w_rows = []
    current_w_start_idx = 0
    for conv_idx in range(len(list_of_conv_layers)):
        to_append = []
        current_w_end_idx = current_w_start_idx + list_of_conv_layers[conv_idx].out_channels
        for non_zero_w_idx in non_zero_w_indices:
            if current_w_start_idx <= non_zero_w_idx < current_w_end_idx:
                to_append.append(non_zero_w_idx - current_w_start_idx)
        sub_w_rows.append(to_append)
        current_w_start_idx = current_w_end_idx
    w_decoded.append(sub_w_rows)

batch_size = 1
for class_index in range(1):
    class_index = 23
    for target_layer in target_layers:

        targets = [test_dataset.dataset.targets[i] for i in test_dataset.indices]
        indices = torch.tensor(targets) == class_index
        new_dataset = torch.utils.data.dataset.Subset(test_dataset, np.where(indices == 1)[0])
        test_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

        heatmaps = []
        for i, data in enumerate(test_loader):
            if i == 6:
                model.return_spike_maps = True
                inputs, labels = data
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)
                output, all_spikemaps = model(inputs)

                spikemaps = all_spikemaps[target_layer]
                model.return_spike_maps = False

                preds = torch.argmax(output, dim=1)

                w_for_layer = w_decoded[preds[0]][target_layer]
                avg_map = torch.zeros(spikemaps[:, 0].shape)
                for filter_idx in w_for_layer:
                    filter_output = spikemaps[:, filter_idx, :, :]
                    avg_map = torch.add(avg_map.cuda(), filter_output.squeeze())

                resized_avg_map = F.interpolate(avg_map.unsqueeze(1), size=(224, 224), mode='bilinear',
                                                align_corners=False)

                resized_avg_map = normalize_numpy(resized_avg_map.cpu().numpy())
                heatmaps.append(resized_avg_map[0])

        with open('./project_antwerp/Interpretation_of_SNN/snn/svgg19/explanation/output/single_class/' + str(
                class_index) + '_' + str(target_layer) + '.npy', 'wb') as file_add:
            pickle.dump(heatmaps, file_add)
