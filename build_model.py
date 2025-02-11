
import torch
import torch.nn as nn
import torchvision
from sngp_wrapper.covert_utils import replace_layer_with_gaussian, convert_to_sn_my
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for conv2
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)  # Batch normalization for fc1
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(128, num_classes)
        self.dropout_rate = 0.1

    def forward(self, x, return_hidden=False, **kwargs):
        # Convolutional layer 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Convolutional layer 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Flatten the tensor
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layer
        x = self.fc1(x)
        x = self.bn3(x)  # Apply batch normalization
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Output layer
        logits = self.classifier(x, **kwargs)
        if return_hidden:
            return F.log_softmax(logits, dim=1), x
        else:
            return F.log_softmax(logits, dim=1)

def Build_MNISTClassifier(num_classes):
    return MNISTClassifier(num_classes=num_classes)




