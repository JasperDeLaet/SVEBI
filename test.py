import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from model import *

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch.backends.cudnn as cudnn
from utils import *

import logging
import sys

output_dir = './project_antwerp/Interpretation_of_SNN/snn/MNIST/output/'

# Logging
logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(output_dir + 'train.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def log_string(str):
    logger.info(str)
    print(str)


log_string("START")

cudnn.benchmark = True
cudnn.deterministic = True

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Leaky-Integrate-and-Fire (LIF) neuron parameters
leak_mem = 0.95

# SNN learning and evaluation parameters
batch_size = 100
batch_size_test = batch_size
num_epochs = 20
num_steps = 10
lr = 0.3

img_size = 28
num_cls = 10

# dataloader arguments
data_path = '/project_antwerp/Interpretation_of_SNN/data/mnist/'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# --------------------------------------------------
# Instantiate the SNN model and optimizer
# --------------------------------------------------

model = snn_mnist(num_steps=10, leak_mem=0.95, img_size=28, num_cls=10)

model = model.cuda()

# Configure the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
best_acc = 0

log_string('********** SNN MNIST training and evaluation **********')
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

best_model = None
best_test_accuracy = 0
for epoch in range(num_epochs):

    log_string('EPOCH ' + str(epoch) + ':')
    train_running_loss = 0
    train_correct = 0
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        output = model(inputs)
        preds = torch.argmax(output, dim=1)

        loss = criterion(output, labels)
        train_correct += (preds == labels).sum().item()
        train_running_loss += loss.item()

        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    train_acc_list.append(train_accuracy)
    log_string(f'Train Acc: {train_accuracy:.2f} Train loss {train_loss:.2f}')

    test_running_loss = 0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            out = model(images)
            preds = torch.argmax(out, dim=1)
            loss = criterion(out, labels)
            test_correct += (preds == labels).sum().item()
            test_running_loss += loss.item()
    test_loss = test_running_loss / len(test_loader.dataset)
    test_loss_list.append(test_loss)
    test_accuracy = 100. * test_correct / len(test_loader.dataset)
    test_acc_list.append(test_accuracy)
    log_string(f'Test Acc: {test_accuracy:.2f}, Test loss: {test_loss:.2f}')

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_model = model.state_dict()
        PATH = output_dir + 'snn_mnist.pth'
        torch.save(best_model, PATH)
        best_epoch = epoch

log_string('Best epoch: ' + str(best_epoch))
log_string('Best train accuracy: ' + str(train_acc_list[best_epoch]))
log_string('Best validation accuracy: ' + str(test_acc_list[best_epoch]))
log_string('Finished Training')

log_string('Saving model ...')

generate_training_plots(train_acc_list, test_acc_list, train_loss_list, test_loss_list, output_dir)

PATH = output_dir + 'snn_mnist.pth'
torch.save(best_model, PATH)

log_string("DONE")


