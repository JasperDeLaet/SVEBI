import torch.optim as optim
from torchvision import datasets, transforms
from model import *
from torchvision.models import vgg19
import torch.backends.cudnn as cudnn
import torch.utils.data as torchdata
import logging
from util import *
from PIL import ImageFile

'''
This file implements our training procedure for the Spiking VGG19 (SVGG19). 

This training implementation was inspired by the training implementation of the work:
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
file_handler = logging.FileHandler(output_dir + 'train.txt')
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
num_epochs = 50
num_steps = 25
lr = 0.3

# Load data
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
testloader = torchdata.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
# We use the pretrained weights for non-spiking vgg19 trained on ImageNet
model = svgg19(num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls)
model.load_state_dict(vgg19(weights='IMAGENET1K_V1').state_dict())
model.classifier._modules['6'] = nn.Linear(4096, 50)

model = model.cuda()

# Configure the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

best_acc = 0

# Print the SNN model, optimizer, and simulation parameters
log_string('********** SNN simulation parameters **********')
log_string('Simulation # time-step : {}'.format(num_steps))
log_string('Membrane decay rate : {0:.2f}\n'.format(leak_mem))

log_string('********** SNN learning parameters **********')
log_string('Backprop optimizer     : SGD')
log_string('Batch size (training)  : {}'.format(batch_size))
log_string('Batch size (testing)   : {}'.format(batch_size))
log_string('Number of epochs       : {}'.format(num_epochs))
log_string('Learning rate          : {}'.format(lr))

# Training
log_string('********** SNN training and evaluation **********')
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
    for i, data in enumerate(trainloader):
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
    train_loss = train_running_loss / len(trainloader.dataset)
    train_loss_list.append(train_loss)
    train_accuracy = 100. * train_correct / len(trainloader.dataset)
    train_acc_list.append(train_accuracy)
    log_string(f'Train Acc: {train_accuracy:.2f} Train loss {train_loss:.2f}')

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
    test_loss = test_running_loss / len(testloader.dataset)
    test_loss_list.append(test_loss)
    test_accuracy = 100. * test_correct / len(testloader.dataset)
    test_acc_list.append(test_accuracy)
    log_string(f'Test Acc: {test_accuracy:.2f}, Test loss: {test_loss:.2f}')

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_model = model.state_dict()
        PATH = output_dir + 'svgg19_AwA2.pth'
        torch.save(best_model, PATH)
        best_epoch = epoch

log_string('Best epoch: ' + str(best_epoch))
log_string('Best train accuracy: ' + str(train_acc_list[best_epoch]))
log_string('Best validation accuracy: ' + str(test_acc_list[best_epoch]))
log_string('Finished Training')

log_string('Saving model ...')

generate_training_plots(train_acc_list, test_acc_list, train_loss_list, test_loss_list, output_dir)

PATH = output_dir + 'svgg19_AwA2.pth'
torch.save(best_model, PATH)

log_string("DONE")
