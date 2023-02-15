import torch
import torch.nn as nn


class mCNN(nn.Module):
    def __init__(self,
                 window_sizes=[8, 16, 24, 32, 40, 48],
                 max_length=5000,
                 num_feature=20,
                 num_filters=256,
                 num_hidden=1024,
                 num_class=2):
        super(mCNN, self).__init__()

        input_shape = (1, max_length, num_feature)
        self.input_layer = nn.Parameter(torch.Tensor(*input_shape))
        self.window_sizes = window_sizes
        self.conv2d = nn.ModuleList()
        self.maxpool = nn.ModuleList()
        self.flatten = nn.ModuleList()
        for window_size in self.window_sizes:
            self.conv2d.append(nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(1, window_size),
                padding=(0, 0)
            ))
            self.maxpool.append(nn.MaxPool2d(
                kernel_size=(1, max_length - window_size + 1),
                stride=(1, max_length),
                padding=(0, 0)
            ))
            self.flatten.append(nn.Flatten())
        self.dropout = nn.Dropout(p=0.7)
        self.fc1 = nn.Linear(
            in_features=num_filters * len(self.window_sizes),
            out_features=num_hidden
        )
        self.fc2 = nn.Linear(
            in_features=num_hidden,
            out_features=num_class
        )

    def forward(self, x, training=False):
        _x = []
        for i in range(len(self.window_sizes)):
            x_conv = self.conv2d[i](x)
            x_maxp = self.maxpool[i](x_conv)
            x_flat = self.flatten[i](x_maxp)
            _x.append(x_flat)

        x = torch.cat(_x, dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x