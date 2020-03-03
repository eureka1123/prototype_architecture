import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt 
from helper_func import list_of_distances, list_of_norms

class Autoencoder(nn.Module):
    def __init__(self, channels):
        super(Autoencoder, self).__init__()
        self.econv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1) #not zero padding?
        self.econv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.econv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.econv4 = nn.Conv2d(32, 10, 3, stride=2, padding=1)

        self.dconv1 = nn.ConvTranspose2d(10, 32, 3, stride=3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.dconv3 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1,output_padding = 1)
        self.dconv4 = nn.ConvTranspose2d(32, channels, 3, stride=2, padding=1,output_padding = 1)

    def encode(self, inputs):
        x = torch.sigmoid(self.econv1(inputs))
        x = torch.sigmoid(self.econv2(x))
        x = torch.sigmoid(self.econv3(x))
        x = torch.sigmoid(self.econv4(x))
        return x

    def decode(self, inputs):
        x = inputs.reshape((len(inputs),10,2,2))
        x = torch.sigmoid(self.dconv1(x))
        x = torch.sigmoid(self.dconv2(x))
        x = torch.sigmoid(self.dconv3(x))
        x = torch.sigmoid(self.dconv4(x))
        return x

    def forward(self, inputs):
        transform_input = self.encode(inputs)
        recon_input = self.decode(transform_input)
        return transform_input, recon_input

class Prototype(nn.Module):
    def __init__(self, channels, num_prototypes, num_classes):
        super(Prototype, self).__init__()
        flat_size = 2*2*10
        self.num_prototypes = num_prototypes
        self.autoencoder = Autoencoder(channels)
        self.fc = nn.Linear(num_prototypes, num_classes)
        self.prototypes = nn.Parameter(torch.stack([torch.rand(size = (flat_size,), requires_grad = True) for i in range(self.num_prototypes)]))

    def forward(self, inputs):
        batch_size = len(inputs)
        transform_input, recon_input = self.autoencoder(inputs)
        shape = transform_input.shape
        transform_input = transform_input.reshape(shape[0],-1)
        prototypes_difs = list_of_distances(transform_input,self.prototypes)
        feature_difs = list_of_distances(self.prototypes,transform_input)
        output = F.softmax(self.fc(prototypes_difs))
        return transform_input, recon_input, self.prototypes, output, prototypes_difs, feature_difs

def load_data():
    batch_size_train = 250
    batch_size_test = 1000
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])#, torchvision.transforms.Normalize((0.1307,), (.3081,))])
    train_loader = DataLoader(torchvision.datasets.MNIST('data/', train=True, download=True,
                        transform = transform), batch_size = batch_size_train, shuffle = True)
    test_loader = DataLoader(torchvision.datasets.MNIST('data/', train=False, download=True,
                        transform = transform), batch_size = batch_size_test, shuffle = True)
    return train_loader, test_loader

def train(model, train_loader, optimizer, epochs = 10):
    model.train()
    log_interval = 100
    train_losses=[]
    num_classes = 10
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            total = 0
            # plt.imshow(data.numpy()[0][0], cmap='gray')
            # plt.show()
            optimizer.zero_grad()
            transform_input, recon_input, prototypes, output, prototypes_difs, feature_difs = model(data)
            cross_entropy_loss, recon_loss, r1_loss, r2_loss, loss = loss_func(transform_input, recon_input, data, output, target, prototypes_difs, feature_difs)
            loss.backward()
            optimizer.step()
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum()
            total+=len(pred)
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} {:0f}%)]\tloss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    epoch, batch_idx*len(data), len(train_loader.dataset),
                    100.*batch_idx/len(train_loader), loss.item(), correct, total, 100.*correct/total))
                train_losses.append(loss.item())
                # torch.save(model.state_dict(), 'results/model.pth')    
                # torch.save(optimizer.state_dict(), 'results/optimizer.pth')      

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            transform_input, recon_input, prototypes, output, prototypes_difs, feature_difs = model(data)
            cross_entropy_loss, recon_loss, r1_loss, r2_loss, loss = loss_func(transform_input, recon_input, data, output, target, prototypes_difs, feature_difs)
            test_loss += loss
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))

def loss_func(transform_input, recon_input, input_target, output, output_target, prototypes_difs, feature_difs):
    cl = 20
    l = 1 #.05
    l1 = 1#.05
    l2 = 1#.05
    ce_loss_fn = nn.CrossEntropyLoss()
    cross_entropy_loss = ce_loss_fn(output, output_target)
    recon_loss = list_of_norms(recon_input-input_target)
    recon_loss = torch.mean(recon_loss)
    r1_loss = torch.mean(torch.min(feature_difs,dim=1)[0])
    r2_loss = torch.mean(torch.min(prototypes_difs,dim=1)[0])

    total_loss = cl*cross_entropy_loss + l*recon_loss + l1*r1_loss + l2*r2_loss
    return cross_entropy_loss, recon_loss, r1_loss, r2_loss, total_loss

def run_model(should_train=False, should_test=False, visualize_prototypes=False):
    learning_rate = .002
    model = Prototype(channels = 1, num_prototypes = 15, num_classes = 10)
    train_loader, test_loader = load_data()
    if should_train:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train(model, train_loader, optimizer)
    if should_test:
        model.load_state_dict(torch.load("results/model.pth"))
        test(model,test_loader)
    if visualize_prototypes: 
        model.load_state_dict(torch.load("results/model.pth"))
        autoencoder = model.autoencoder
        decoded_prototypes = autoencoder.decode(model.prototypes)
        for p in decoded_prototypes:
            plt.imshow(p.detach().numpy()[0], cmap='gray')
            plt.show()
    
    
run_model(should_train=True,should_test = False,visualize_prototypes=True)


