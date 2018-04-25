import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.manifold import TSNE; HAS_SK = True

# --------------------------------------------------------------------------------------------
# hyper Parameters
BATCH_SIZE = 200
IMAGE_DIMEN = 1
LEARNING_RATE = 0.1
EPOCH_SIZE = 50 #train the training data 50 times

# --------------------------------------------------------------------------------------------

torch.manual_seed(1234)
# --------------------------------------------------------------------------------------------
# Download data and generate train dataset and test dataset
torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor())

trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# --------------------------------------------------------------------------------------------
#plot first number for an example
plt.imshow(trainset.train_data[0].numpy(), cmap='gray')
plt.title('%i' % trainset.train_labels[0])
plt.show()
# --------------------------------------------------------------------------------------------
# CNN module for MINST dataset
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  #input height  (1,28,28)
					  out_channels=16, #number of filters
					  kernel_size=5, #the wide and height of filter is 5
					  stride=1, #filter movements
					  padding=2  #In order to let the size(input and output) same padding=(kernal size-1)/2
					  ), #(16,28,28)
            nn.BatchNorm2d(16),
            nn.ReLU(),  #Activation Function #(16,28,28)
            nn.MaxPool2d(kernel_size=2))  # Max pooling over 2*2 windows, output shape(16,14,14)
        self.layer2 = nn.Sequential( #input(16,14,14)
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32), #(32,14,14)
            nn.ReLU(), #(32,14,14)
            nn.MaxPool2d(2)) #(32,7,7)
        self.output_layer = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        out = self.layer1(x)     #input image
        out = self.layer2(out)   #(batch, 32, 7,7)
        out = out.view(out.size(0), -1)  #flatten layer (batch, 32 * 7 * 7)
        out = self.output_layer(out)
        return out
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Dataset Overview and visualization

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def overviewData(data_loader, overview):
	dataiter = iter(data_loader)
	images, labels = dataiter.next()
	imshow(torchvision.utils.make_grid(images))
	print(overview, ' '.join('%5s' % str(labels[j]) for j in range(labels.size(0))))

def plot_with_labels(lowDWeights, labels):
	plt.cla()
	X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
	for x, y, s in zip(X, Y, labels):
		c = cm.rainbow(int(255 * s / 9));
		plt.text(x, y, s, backgroundcolor=c, fontsize=9)
		plt.xlim(X.min(), X.max());
		plt.ylim(Y.min(), Y.max());
		plt.title('Visualize last layer');
		plt.show();
		plt.pause(0.01)

def visualLastLayer(model, testset):
	plt.ion()
	test_x = Variable(torch.unsqueeze(testset.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
	test_y = testset.test_labels[:2000]
	last_layer = model(test_x)
	if HAS_SK:# Visualization of trained flatten layer (T-SNE)
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
		plot_only = 500
		low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
		labels = test_y.numpy()[:plot_only]
		plot_with_labels(low_dim_embs, labels)
	plt.ioff()

# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# train the model
def trainModel(model, dataloader, epoch, criterion, optimizer):
	print("Training Epoch %i" % (epoch + 1))
	model.train()
	running_loss = 0
	for i, data in enumerate(dataloader, 0):
		images, labels = data #get the input
		images, labels = Variable(images), Variable(labels) #prepare for calculating the gradient
		optimizer.zero_grad() #gradient to 0, if not set to 0, the gradient will increase as backward
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward() #get the gradient
		optimizer.step() #update the data, data = data + learning rate * gradient
		running_loss += loss.data[0]
		if (i + 1) % 50 == 0:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
			running_loss = 0.0 #print every 50 mini-batches
	print('Training Completed')
# --------------------------------------------------------------------------------------------
#test the model
#see how the network performs on the whole dataset
def testModel(model, dataloader, criterion, optimizer):
	model.eval() #change model to 'eval' mode
	correct = 0
	total = 0
	for images, targets in dataloader:
		images = Variable(images)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += (predicted == targets).sum()
	print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

#make visualization
def visualTestModel(model, dataloader, criterion, optimizer):
	plt.ion()
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
	accuracies, steps = [], []
	model.eval() #change model to 'eval' mode
	correct = 0
	total = 0
	for images, targets in dataloader:
		images = Variable(images)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += (predicted == targets).sum()
		accuracy = correct / (1.0*total)
		accuracies.append(accuracy)
		steps.append(total)
		ax1.cla()
		for c in range(10):
			bp = ax1.bar(c+0.1, height=sum((predicted == c)), width=0.2, color='red')
			bt = ax1.bar(c-0.1, height=sum((targets == c)), width=0.2, color='blue')
			ax1.set_xticks(range(10), ['0','1','2','3','4','5','6','7','8','9'])
			ax1.legend(handles=[bp, bt], labels=["prediction", "real number"])
			ax2.cla()
			ax2.plot(steps, accuracies, label="accuracy")
			ax2.set_ylim(ymax=1)
			ax2.set_ylabel("accuracy")
			plt.pause(0.01)
	plt.ioff()
	plt.show()



# --------------------------------------------------------------------------------------------
# Main function
def main():
	print(trainset.train_data.size())                 # (60000, 28, 28)
	print(trainset.train_labels.size())               # (60000)
	cnn = CNN()
	print(cnn)  #net architecture
	criterion = nn.CrossEntropyLoss()  #loss
	optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=0.9)  #optimizer
	overviewData(trainloader, 'Randomly Training Samples: ')
	overviewData(testloader, 'Randomly Testing Samples: ')
	for epoch in range(EPOCH_SIZE):
		trainModel(cnn, trainloader, epoch, criterion, optimizer)
		testModel(cnn, testloader, criterion, optimizer)
	visualTestModel(cnn, testloader, criterion, optimizer)
	visualLastLayer(cnn, testset)

# --------------------------------------------------------------------------------------------

if __name__ == '__main__':
	main()

