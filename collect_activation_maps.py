import pickle
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import logging
from torchvision.models import vgg
import torch.utils.data as torchdata

from util import *

output_dir = "./output/activation_maps/"


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Logging
logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(output_dir + 'log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Logger function wrapper
def log_string(string):
    logger.info(string)
    print(string)


log_string("START")


# Spike function here only has a forward pass implemented as we collect activation maps
def spike_function(input):
    out = torch.zeros_like(input).cuda()
    out[input > 0] = 1.0
    return out


# We alter the spiking vgg19 implementation here such that it does not track gradients for memory reasons
# In addition, we return all average activation maps per filter
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
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

            # To collect all activation maps and return
            all_activation_maps = [list() for conv in mem_conv_list]

            for t in range(self.num_steps):
                out_prev = x

                # Keep track of which actual CONV layer is currently input being pushed through
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
                        all_activation_maps[conv_layer_counter].append(out_prev.detach())

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

            # Average (binary) activation maps per filter over all timesteps
            all_activation_maps = [torch.mean(torch.stack(s, dim=0).permute(1, 0, 2, 3, 4), dim=1) for s in all_activation_maps]

            out_voltage = mem_fc_list[-1] / self.num_steps

            return out_voltage, all_activation_maps


img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Define model
model = svgg19(num_steps=25, leak_mem=0.95, img_size=224, num_cls=50)
model.classifier._modules['6'] = nn.Linear(4096, 50)
model.load_state_dict(torch.load('./models/svgg19_AwA2.pth'))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
model.eval()

# Define data source and details
data_path = "./data/Animals_with_Attributes2/JPEGImages/"
dataset = datasets.ImageFolder(data_path, transform=transform)

# Generate data split
total_count = len(dataset.imgs)
train_count = int(0.8 * total_count)
test_count = total_count - train_count
train_dataset, test_dataset = torchdata.random_split(dataset, (train_count, test_count),
                                                     generator=torch.Generator().manual_seed(seed))

# Define embedding necessary for generating binary labels for building L matrix
number_of_classes = 50
emb = torch.nn.Embedding(number_of_classes, number_of_classes)
emb.weight.data = torch.eye(number_of_classes)

for class_index in range(number_of_classes):
    '''
    Here, we collect activation maps for all samples per class
    This way we can investigate model behaviour per class
    This loop produces the X and L necessary to produce W split over all classes
    '''
    log_string('Class: ' + str(class_index))
    targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    indices = torch.tensor(targets) == class_index
    new_dataset = torch.utils.data.dataset.Subset(train_dataset, np.where(indices == 1)[0])
    train_loader = DataLoader(new_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=2)

    real_labels = None
    pred_labels = None
    X = None
    L = []
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device)
        output, all_spikemaps = model(inputs)

        output = output.detach().cpu()

        inputs = inputs.detach().cpu()
        labels = labels.detach().cpu().numpy()

        pred = torch.argmax(output, dim=1)
        l2_norm = generating_l2_norm(all_spikemaps, labels.shape[0])
        del all_spikemaps

        if X is None:
            X = l2_norm
        else:
            X = np.concatenate((X, l2_norm), axis=0)
        if real_labels is None:
            real_labels = labels
            pred_labels = pred
        else:
            real_labels = np.concatenate((real_labels, labels), axis=0)
            pred_labels = np.concatenate((pred_labels, pred), axis=0)
        for i in range(labels.shape[0]):
            binary_label = emb(torch.from_numpy(np.asarray(pred[i]))).detach().cpu().numpy()
            L.append(binary_label)

    '''
    Save the results X and L for the current class
    '''
    log_string('Saving X and L for class: ' + str(class_index))
    with open(output_dir + str(class_index) + '.npy', 'wb') as file_add:
        pickle.dump((X, L, real_labels, pred_labels), file_add)

    del X
    del L
    del real_labels
    del pred_labels
