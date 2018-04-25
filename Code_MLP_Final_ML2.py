import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# --------------------------------------------------------------------------------------------
# Initiate Parameters
HIDDEN_SIZE = 200
CLASSES_NUM = 10
BATCH_SIZE = 200
IMAGE_DIMEN = 1
IMAGE_SIZE = 28
LEARNING_RATE = 0.01
EPOCH_SIZE = 50

# --------------------------------------------------------------------------------------------
torch.manual_seed(1234)
# --------------------------------------------------------------------------------------------
# Download data and generate train dataset and test dataset
#root is the directory where dataset exists
#train is optional.  Boolean value.  IF it is true gets the data from train set otherwise test set.
#download is optional.  Boolean value.  If it is true gets the data from iternet and place it in root folder.
#transform is a function which takes PIL image and transforms it into another version.  PIL python image library or Pillow is a library for opening, formating various different image format.
#target_transform takes in the target and transform it
#transforms.ToTensor() transforms the common image
torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor())

trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor())

#dataloader combines the dataset and sampler
#shuffle= to have data shuffle at every epoch
#sample is optional is strategy to draw samples from dataset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# MLP module for MINST dataset
class MLP(nn.Module):
	def __init__(self, input_size, hidden_size, classes_num):
		super(MLP, self).__init__()								#call the main MLP function
		self.input_layer = nn.Linear(input_size, hidden_size)	#its a two layer network
		self.output_layer = nn.Linear(hidden_size, classes_num)
		self.tanh = nn.Tanh()									#nn.tanh() applies elementwise tanh(x)
		self.reg_layer = nn.LogSoftmax()					#transfer function are used to introduce non-linearity after parameterized layer

	def forward(self, x):
		out = self.input_layer(x)
		out = self.tanh(out)
		out = self.output_layer(out)
		out = self.reg_layer(out)
		return out

# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Necessary functions
def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def overviewData(data_loader):
	dataiter = iter(data_loader)
	images, labels = dataiter.next()
	imshow(torchvision.utils.make_grid(images))

def trainModel(model, trainloader, epoch, criterion, optimizer):
	print("Training Epoch %i" % (epoch + 1))
	model.train()
	running_loss = 0
	for i, data in enumerate(trainloader, 0):							#set the counter for trainloader and start the index from 0
		images, labels = data
		images= images.view(-1,IMAGE_DIMEN * IMAGE_SIZE * IMAGE_SIZE)
		images, labels = Variable(images), Variable(labels)
		optimizer.zero_grad()											#zero_grad clear gradient of all optimized
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()												#optimize the weight and bias
		running_loss += loss.data[0]
		if (i + 1) % 50 == 0:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
			running_loss = 0.0

def testModel(model, testloader, criterion, optimizer):
	model.eval()
	test_loss = 0
	correct = 0
	for images, targets in testloader:
		images= images.view(-1,IMAGE_DIMEN * IMAGE_SIZE * IMAGE_SIZE)
		images, targets = Variable(images), Variable(targets)
		outputs = model(images)
		test_loss += F.nll_loss(outputs, targets, size_average=False).data[0]
		pred = outputs.data.max(1, keepdim=True)[1]
		correct += pred.eq(targets.data.view_as(pred)).sum()
	test_loss /= len(testloader.dataset)
#	print('pred: ',pred)
#	print('correct: ' ,correct)
#	print('test_loss' ,test_loss)
	print('Test set: Average loss: {:.3f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		  test_loss, correct, len(testloader.dataset),
		  100. * correct / len(testloader.dataset)))

# --------------------------------------------------------------------------------------------
# Main function

def main():
	mlp = MLP(IMAGE_DIMEN*IMAGE_SIZE*IMAGE_SIZE, HIDDEN_SIZE, CLASSES_NUM)
	criterion = nn.NLLLoss()													#Negative loss likelihood loss used to train classification problem with classes
	optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=0.9)
	overviewData(trainloader)
	for epoch in range(EPOCH_SIZE):
		trainModel(mlp, trainloader, epoch, criterion, optimizer)
	testModel(mlp, testloader, criterion, optimizer)


# --------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
