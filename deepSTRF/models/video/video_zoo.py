import torch
import torch.nn as nn
import torch.nn.functional as F

# from spikingjelly.activation_based import neuron, surrogate, functional

# class VideoModel1(nn.Module):
#     def __init__(self, spatial_resolution, out_neurons, init_mean=0.0, init_std=0.2):
#         super(VideoModel1, self).__init__()
        
#         self.init_mean = init_mean
#         self.init_std = init_std

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=5, stride=4, padding=2),
#             nn.BatchNorm2d(8),
#         )
#         self.lif1 = neuron.LIFNode(
#                 tau=2., 
#                 v_threshold=1., 
#                 surrogate_function=surrogate.ATan(alpha = 5.0),
#                 detach_reset=True, 
#                 step_mode='m',
#                 decay_input=False,
#                 store_v_seq = True,
#                 )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(8, 8, kernel_size=7, stride=2, padding=3, groups=8),
#             nn.Conv2d(8, 16, kernel_size=1),
#             nn.BatchNorm2d(16),
#         )
#         self.lif2 = neuron.LIFNode(
#                 tau=2., 
#                 v_threshold=1., 
#                 surrogate_function=surrogate.ATan(alpha = 5.0),
#                 detach_reset=True, 
#                 step_mode='m',
#                 decay_input=False,
#                 store_v_seq = True,
#                 )
        
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(16, 16, kernel_size=7, stride=2, padding=3, groups=16),
#             nn.Conv2d(16, 32, kernel_size=1),
#             nn.BatchNorm2d(32),
#         )
#         self.lif3 = neuron.LIFNode(
#                 tau=2., 
#                 v_threshold=1., 
#                 surrogate_function=surrogate.ATan(alpha = 5.0),
#                 detach_reset=True, 
#                 step_mode='m',
#                 decay_input=False,
#                 store_v_seq = True,
#                 )
        
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=7, stride=2, padding=3, groups=32),
#             nn.Conv2d(32, 64, kernel_size=1),
#             nn.BatchNorm2d(64),
#         )
#         self.lif4 = neuron.LIFNode(
#                 tau=2., 
#                 v_threshold=1., 
#                 surrogate_function=surrogate.ATan(alpha = 5.0),
#                 detach_reset=True, 
#                 step_mode='m',
#                 decay_input=False,
#                 store_v_seq = True,
#                 )
        
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, groups=64),
#             nn.Conv2d(64, 128, kernel_size=1),
#             nn.BatchNorm2d(128),
#         )
#         self.lif5 = neuron.LIFNode(
#                 tau=2., 
#                 v_threshold=1., 
#                 surrogate_function=surrogate.ATan(alpha = 5.0),
#                 detach_reset=True, 
#                 step_mode='m',
#                 decay_input=False,
#                 store_v_seq = True,
#                 )

#         self.global_pool = nn.AdaptiveAvgPool2d((2, 2)) 
#         self.fc = nn.Linear(128 * 2 * 2, out_neurons)
        
#         self.initialize_weights()

#     def forward(self, x):
#         # x shape: (B, C=1, H, W, T)
#         B, C, H, W, T = x.shape
#         x = x.permute(0, 4, 1, 2, 3)  # (B, T, C=1, H, W)
#         x = x.reshape(-1, 1, H, W)    # (B*T, 1, H, W)

#         x = self.lif1(self.conv1(x).unflatten(0, (B, T)).permute(1, 0, 2, 3, 4)).flatten(0, 1) # (T*B, C, H, W)
#         x = self.lif2(self.conv2(x).unflatten(0, (B, T)).permute(1, 0, 2, 3, 4)).flatten(0, 1) # (T*B, C, H, W)
#         x = self.lif3(self.conv3(x).unflatten(0, (B, T)).permute(1, 0, 2, 3, 4)).flatten(0, 1) # (T*B, C, H, W)
#         x = self.lif4(self.conv4(x).unflatten(0, (B, T)).permute(1, 0, 2, 3, 4)).flatten(0, 1) # (T*B, C, H, W)
#         x = self.lif5(self.conv5(x).unflatten(0, (B, T)).permute(1, 0, 2, 3, 4)).flatten(0, 1) # (T*B, C, H, W)

#         x = self.global_pool(x)                 # (T*B, 128, 2, 2)
#         x = x.view(x.size(0), -1)               # (T*B, 128*2*2)
#         x = self.fc(x)                          # (T*B, out_neurons)
#         x = x.view(T, B, -1).permute(1, 2, 0)   # (B, out_neurons, T)

#         return x
    
#     def reset_model(self):
#         functional.reset_net(self)
        
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=self.init_mean, std=self.init_std)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
                    
class SimpleConv3DModel(nn.Module):
    def __init__(self, out_neurons):
        super(SimpleConv3DModel, self).__init__()

        self.conv = nn.Conv3d(1, 32, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.fc = nn.Linear(32, out_neurons)

        self.initialize_weights()

    def forward(self, x):
        # Input shape: (B, 1, H, W, T)
        x = x.permute(0, 1, 4, 2, 3)  # (B, 1, T, H, W)
        x = self.conv(x)              # (B, 32, T', H', W')
        x = x.mean(dim=[3, 4])        # average over H and W -> (B, 32, T')
        x = x.permute(0, 2, 1)        # (B, T', 32)
        x = self.fc(x)                # (B, T', out_neurons)
        x = x.permute(0, 2, 1)        # (B, out_neurons, T')
        return x

    def initialize_weights(self, mean=0.0, std=1e-2):
        nn.init.normal_(self.conv.weight, mean=mean, std=std)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.normal_(self.fc.weight, mean=mean, std=std)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)