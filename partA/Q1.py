import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(
        self,
        input_channels=3,
        filters_per_layer=[32, 64, 128, 256, 512],
        kernel_size=3,
        pool_sizes=2,
        conv_activation='relu',
        dense_units=256,
        dense_activation='relu',
        num_classes=10,
        dropout_rate=0.5,
        use_batch_norm=True
    ):
        
        super().__init__()
        
        self.kernel_size = kernel_size
        self.pool_sizes = pool_sizes
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation

        # Initializing Convolutional, Batch Norm, and pooling layers
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = input_channels
        
        # Create 5 convolutional blocks
        for filters in filters_per_layer:
            # Convolutional layer
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            
            # Batch normalization layer
            if use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm2d(filters))
            else:
                self.batch_norm_layers.append(None)
            
            # Max pooling layer
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            # Update in_channels for next layer
            in_channels = filters
        
        # Calculate the output size after conv layers
        # Assuming input is 224x224, after 5 max-pooling layers it will be 7x7
        conv_output_size = 7 * 7 * filters_per_layer[-1]
        
        # First flatten the image to pass it to the dense layer
        self.flatten = nn.Flatten()

        # Dense layer
        self.fc1 = nn.Linear(conv_output_size, dense_units)
        self.fc_bn = nn.BatchNorm1d(dense_units) if use_batch_norm else None
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc2 = nn.Linear(dense_units, num_classes)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def activation_func(self, activation, x):
        """Apply the selected activation function"""
        if activation.lower() == 'relu':
            return F.relu(x)
        elif activation.lower() == 'gelu':
            return F.gelu(x)
        elif activation.lower() == 'silu' or activation.lower() == 'swish':
            return F.silu(x)
        elif activation.lower() == 'mish':
            return x * torch.tanh(F.softplus(x))
            return F.sigmoid(x)
        elif activation.lower() == 'leakyrelu':
            return F.leaky_relu(x, negative_slope=0.01)
        else:
            # Default to ReLU
            return F.relu(x)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Convolutional blocks
        for i, (conv, bn, pool) in enumerate(zip(self.conv_layers, self.batch_norm_layers, self.pool_layers)):
            x = conv(x)
            if bn is not None:
                x = bn(x)
            x = self.activation_func(self.conv_activation, x)
            x = pool(x)
        
        # Flatten
        x = self.flatten(x)
        # x = torch.flatten(x, 1)
        
        # Dense layer
        x = self.fc1(x)
        if self.fc_bn is not None:
            x = self.fc_bn(x)
        x = self.activation_func(self.dense_activation, x)
        x = self.dropout1(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x
