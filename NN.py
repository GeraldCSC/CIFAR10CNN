import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class Net(nn.Module):
    def __init__(self, numfirst , numsecond): 
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU() ,nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(),nn.MaxPool2d(2), nn.Dropout(0.2))
        self.fc1 = nn.Sequential(nn.Linear(64 * 2 * 2, numfirst), nn.ReLU(), nn.Dropout(0.1))
        self.fc2 = nn.Sequential(nn.Linear(numfirst, numsecond), nn.ReLU(), nn.Dropout(0.2))
        self.fc3 = nn.Sequential(nn.Linear(numsecond, 10), nn.Softmax(dim = 1))


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64*2*2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def train(model, loader, numepoch, weightdecay):
    criterion = nn.CrossEntropyLoss()
    optimizier = optim.Adam(model.parameters() ,weight_decay=weightdecay)
    for e in range(numepoch):
        runningloss = 0.0
        numcorrect = 0.0
        total = 0
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizier.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizier.step()
            runningloss += loss.item()
            total += labels.size(0) 
            pred = torch.max(output.data, 1)[1]
            numcorrect += (pred == labels).sum().item()
        print("Epoch: " + str(e+1))
        print("Accuracy: " + str(numcorrect/total))
        print("Loss: " + str(runningloss / total))
    return model

def test(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("test accuracy: " + str(correct/total))

if __name__ == "__main__":
    batchsize = 50000
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                             shuffle=False)
    model = Net(300, 150).to(device)
    model = train(model, trainloader, 2000, 0)
    test(model,testloader)

