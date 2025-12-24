import mlx.nn as nn
import mlx.core as mx

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            padding = 1
        )

        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x):
        
        return nn.relu(
            self.bn(
                self.conv(x)
            )
        )
    
class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(channels)
        

    def __call__(self, x):
        
        residual = x

        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual 

        return nn.relu(out)
    
class ChessNet(nn.Module):

    def __init__(self):
        super(ChessNet, self).__init__()

        # Backbone
        self.conv_input = ConvBlock(14, 128)
        self.res_tower = nn.Sequential(*[ResBlock(128) for _ in range(5)])

        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

        # Value head
        self.value_conv = nn.Conv2d(128, 3, kernel_size=1)
        self.value_bn = nn.BatchNorm(3)
        self.value_fc1 = nn.Linear(3 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def __call__(self, x):
        
        # MLX Conv2d expects (N, H, W, C ) (channels last)
        # PyTorch format provides (N, C, H, W)
        # The next line for code makes PyTorch format compatible
        x = mx.transpose(x, (0, 2, 3, 1))

        # Backbone
        x = self.conv_input(x)
        
        x = self.res_tower(x)

        # Policy
        p = nn.relu(
            self.policy_bn(
                self.policy_conv(x)
            )
        )
        p = mx.reshape(p, (p.shape[0], -1))
        p = self.policy_fc(p)

        # Value
        v = nn.relu(
            self.value_bn(
                self.value_conv(x)
            )
        )
        v = mx.reshape(v, (v.shape[0], -1))
        v = nn.relu(self.value_fc1(v))
        v = mx.tanh(self.value_fc2(v))

        return p, v