import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, img_dirs):
        # transforming image type to floats
        self.transform = transforms.ConvertImageDtype(torch.float)
        self.img_labels = []
        self.images = []
        # load images the their respective labels into lists
        for id, img_dir in enumerate(img_dirs):
            for img_file in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_file)

                image = read_image(img_path)
                image = self.transform(image)
                self.images.append(image)

                self.img_labels.append(id)

    def __len__(self):
        # define the len() function for the class 
        return len(self.images)

    def __getitem__(self, idx):
        # define a way to index the class
        return self.images[idx], self.img_labels[idx]


img_dirs = ['GTSRB_subset_2/class1/', 'GTSRB_subset_2/class2/']

# creating the dataset and dividing it to train and test
dataset = CustomImageDataset(img_dirs=img_dirs)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# using dataloader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# creating the torch network 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=64*64*3, out_features=100)
        self.dense2 = nn.Linear(in_features=100, out_features=100)
        self.dense3 = nn.Linear(in_features=100, out_features=2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        # softmax activation because we want to predict a class
        output = F.log_softmax(x, dim=1)

        return output

# defining parametes
model = Net()
model.train()
num_epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# training model and calculating loss
for n in range(num_epochs):
    loss_e = 0
    for tr_images, tr_labels in train_loader:
        y_pred = model(tr_images)
        loss = loss_fn(y_pred.float(), tr_labels)
        loss_e += loss.item()*tr_images.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss_e/len(train_loader.sampler))


# evaluating the model on the testing dataset
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # converting fredicted floats to integers
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # sum of how many integers are the correct labels
        correct += (predicted == labels).sum().item()

print(f'model accuracy: {100 * correct / total}%')