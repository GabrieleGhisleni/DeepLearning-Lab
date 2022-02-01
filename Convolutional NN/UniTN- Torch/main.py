import sys
from data_loader import load_stl
from model import ResNet
from solver import ImageClassifier

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader





def main_pytorch():
    # way 2, pytorch uses parallel multi procesing dataloading -> Dataloader

    # init the model
    model = ResNet()

    # we define data transformations
    data_transforms = {'train': transforms.Compose([
                       transforms.RandomResizedCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       'val': transforms.Compose([
                       transforms.Resize(256),
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = ResNet()
    training_data, training_labels, test_data, test_labels = load_stl()
    ic = ImageClassifier(batch_size=32, epochs=2)
    ic.train(training_data, training_labels, test_data, test_labels, model)
