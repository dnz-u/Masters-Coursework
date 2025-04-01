import torch
import torch.nn as nn


class SimpleConvPack(nn.Module):
    def __init__(self, in_channels, out_channels, 
                       kernel_size, stride, padding
                ):
        super(SimpleConvPack, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print(f"Simple Conv: {x.shape=}, {out.shape=}")
        return out

def down_sample(in_channels, out_channels):
    """
    in should be smaller block size
    out should be bigger block size
    """
    # TO DO: check default conv padding val
    downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
        nn.BatchNorm2d(out_channels, eps=1e-05, affine=True, track_running_stats=True)
    )
    #print(f"Downsample: {in_channels=}, {out_channels=}")
    return downsample
        

class myResnetStyleModel(nn.Module):
    def __init__(self, *, keepdims):
        super(myResnetStyleModel, self).__init__()
        self.relu = nn.ReLU() 
        self.input_block_1 = SimpleConvPack(3, 64, 3, 2, 2)
        self.input_block_2 = SimpleConvPack(64, 64, 3, 1, 1)
        
        self.res_block_1_1 = SimpleConvPack(64, 128, 3, 1, 1)
        self.res_block_1_2 = SimpleConvPack(128, 128, 3, 2, 1)
        self.down_sample_1 = down_sample(64, 128)
        
        self.res_block_2_1 = SimpleConvPack(128, 256, 3, 1, 1)
        self.res_block_2_2 = SimpleConvPack(256, 256, 3, 2, 1)
        self.down_sample_2 = down_sample(128, 256)
        
        self.res_block_3_1 = SimpleConvPack(256, 512, 3, 1, 1)
        self.res_block_3_2 = SimpleConvPack(512, 512, 3, 2, 1)
        self.down_sample_3 = down_sample(256, 512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(keepdims)
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):

        out = self.input_block_1(x)
        out = self.input_block_2(out)

        #residual block 1
        identity = out # does this carry the object or copy ?
        out = self.res_block_1_1(out)
        out = self.res_block_1_2(out)
        identity = self.down_sample_1(identity)
        out = out + identity
        out = self.relu(out)
        
        # residual block 2
        identity = out
        out = self.res_block_2_1(out)
        out = self.res_block_2_2(out)
        identity = self.down_sample_2(identity)
        out = out + identity
        out = self.relu(out)
        
        # residual block 3
        identity = out
        out = self.res_block_3_1(out)
        out = self.res_block_3_2(out)
        identity = self.down_sample_3(identity)
        out = out + identity
        out = self.relu(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def initialize_weights(self):
        import torch.nn.init as init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Apply He initialization to weights of Conv and FC layers
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Initialize biases to zero
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm scaling parameter to 1 and shift parameter to 0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
