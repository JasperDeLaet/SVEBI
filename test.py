from torchvision import datasets, transforms
from model import *
from PIL import ImageFile
import torch.backends.cudnn as cudnn
import torch.utils.data as torchdata
from util import *
import logging

'''
This file implements our testing procedure for the Spiking VGG19 (SVGG19). 

This testing implementation was inspired by the testing implementation of the work:
"Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch"
Github: https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time?tab=readme-ov-file
Preprint: https://arxiv.org/abs/2010.01729
'''


ImageFile.LOAD_TRUNCATED_IMAGES = True

output_dir = './output/models/svgg19/'

# Logging
logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(output_dir + 'test.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def log_string(string):
    logger.info(string)
    print(string)


log_string("START")

cudnn.benchmark = True
cudnn.deterministic = True

# Seed, for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Parameters for SNN and training
leak_mem = 0.95

batch_size = 8
batch_size_test = batch_size
num_epochs = 50
num_steps = 25

# Load dataset
img_size = 224
num_cls = 50
data_path = "./data/Animals_with_Attributes2/JPEGImages/"

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(data_path, transform=transform)

# Split data in train and test
total_count = len(dataset.imgs)
train_count = int(0.8 * total_count)
test_count = total_count - train_count
train_dataset, test_dataset = torchdata.random_split(dataset, (train_count, test_count),
                                                     generator=torch.Generator().manual_seed(seed))

trainloader = torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = torchdata.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


# Define model and load weights
model = svgg19(num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls)
model.classifier._modules['6'] = nn.Linear(4096, 50)
model.load_state_dict(torch.load('./output/models/svgg19/svgg19_AwA2.pth'))

model = model.cuda()

# Configure the loss function and optimizer
criterion = nn.CrossEntropyLoss()
best_acc = 0

# Print the SNN model, optimizer, and simulation parameters
log_string('********** SNN simulation parameters **********')
log_string('Simulation # time-step : {}'.format(num_steps))
log_string('Membrane decay rate : {0:.2f}\n'.format(leak_mem))

log_string('********** SNN learning parameters **********')
log_string('Backprop optimizer     : SGD')
log_string('Batch size (training)  : {}'.format(batch_size))
log_string('Batch size (testing)   : {}'.format(batch_size_test))
log_string('Number of epochs       : {}'.format(num_epochs))

# Testing
log_string('********** SNN testing **********')
test_loss_list = []
test_acc_list = []

test_running_loss = 0
test_correct = 0
model.eval()
with torch.no_grad():
     for j, data in enumerate(testloader, 0):

         images, labels = data
         images = images.cuda()
         labels = labels.cuda()

         out = model(images)
         preds = torch.argmax(out, dim=1)
         loss = criterion(out, labels)
         test_correct += (preds == labels).sum().item()
         test_running_loss += loss.item()
test_loss = test_running_loss/len(testloader.dataset)
test_loss_list.append(test_loss)
test_accuracy = 100. * test_correct/len(testloader.dataset)
test_acc_list.append(test_accuracy)
log_string(f'Test Acc: {test_accuracy:.2f}, Test loss: {test_loss:.2f}')

log_string(test_correct / len(test_dataset))

log_string('Finished Testing')

log_string("DONE")
