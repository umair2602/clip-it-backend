"""
TalkNet Model Components
Based on: https://github.com/TaoRuijie/TalkNet-ASD

This is a simplified version for inference only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class visualFrontend(nn.Module):
    """Visual frontend using ResNet-like architecture."""
    
    def __init__(self):
        super(visualFrontend, self).__init__()
        
        # 3D Conv for temporal modeling
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # ResNet blocks
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2])
        
    def forward(self, x):
        # x: (B*T, 1, 1, H, W)
        x = self.frontend3D(x)
        # x: (B*T, 64, 1, H/4, W/4)
        x = x.squeeze(2)  # Remove temporal dim
        x = self.resnet(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class visualTCN(nn.Module):
    """Visual Temporal Convolutional Network."""
    
    def __init__(self):
        super(visualTCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        # x: (B, C, T)
        x = self.tcn(x)
        return x


class visualConv1D(nn.Module):
    """Visual 1D Convolution for final features."""
    
    def __init__(self):
        super(visualConv1D, self).__init__()
        self.conv = nn.Conv1d(128, 128, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x


class audioEncoder(nn.Module):
    """Audio encoder using ResNet-like architecture on spectrograms."""
    
    def __init__(self, layers=[3, 4, 6, 3], num_filters=[16, 32, 64, 128]):
        super(audioEncoder, self).__init__()
        
        self.in_planes = num_filters[0]
        
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        
        self.layer1 = self._make_layer(BasicBlock, num_filters[0], layers[0])
        self.layer2 = self._make_layer(BasicBlock, num_filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, num_filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, num_filters[3], layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, 1, T, F) - spectrogram
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x


class attentionLayer(nn.Module):
    """Cross-attention layer for audio-visual fusion."""
    
    def __init__(self, d_model=128, nhead=8):
        super(attentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value):
        # query, key, value: (T, B, C)
        attn_output, _ = self.self_attn(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output


class talkNetModel(nn.Module):
    """TalkNet Model for Active Speaker Detection."""
    
    def __init__(self):
        super(talkNetModel, self).__init__()
        
        # Visual Temporal Encoder
        self.visualFrontend = visualFrontend()
        self.visualTCN = visualTCN()
        self.visualConv1D = visualConv1D()
        
        # Audio Temporal Encoder
        self.audioEncoder = audioEncoder(layers=[3, 4, 6, 3], num_filters=[16, 32, 64, 128])
        
        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model=128, nhead=8)
        self.crossV2A = attentionLayer(d_model=128, nhead=8)
        
        # Backend classifiers
        self.fcAV = nn.Linear(256, 2)
        self.fcA = nn.Linear(128, 2)
        self.fcV = nn.Linear(128, 2)
        
    def forward_visual_frontend(self, x):
        """Process visual features.
        
        Args:
            x: (B, T, H, W) - grayscale face images
            
        Returns:
            Visual embeddings (B, T, 128)
        """
        B, T, W, H = x.shape
        x = x.view(B * T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688  # Normalize
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)
        x = x.transpose(1, 2)  # (B, 512, T)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1, 2)  # (B, T, 128)
        return x
    
    def forward_audio_frontend(self, x):
        """Process audio features.
        
        Args:
            x: (B, T, F) - MFCC features
            
        Returns:
            Audio embeddings (B, T', 128)
        """
        # Validate input shape
        if x.dim() < 2:
            # Invalid input, return zeros
            B = 1 if x.dim() == 0 else x.size(0)
            return torch.zeros(B, 1, 128, device=x.device)
        
        # Ensure we have at least 3 dimensions (B, T, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        x = x.unsqueeze(1).transpose(2, 3)  # (B, 1, F, T)
        
        B, C, F, T = x.shape
        
        # Handle empty or very short audio
        if T == 0:
            return torch.zeros(B, 1, 128, device=x.device)
        
        # Process in chunks for temporal modeling
        # Audio is 100fps, we need to downsample to match video (25fps)
        chunk_size = 4  # 100fps / 25fps = 4
        
        embeddings = []
        for i in range(0, T, chunk_size):
            chunk = x[:, :, :, i:i+chunk_size]
            actual_size = chunk.size(3)
            if actual_size < chunk_size and actual_size > 0:
                # Pad if necessary
                pad_size = chunk_size - actual_size
                chunk = F.pad(chunk, (0, pad_size))
            elif actual_size == 0:
                continue
            embed = self.audioEncoder(chunk)
            embeddings.append(embed)
        
        if embeddings:
            x = torch.stack(embeddings, dim=1)  # (B, T', 128)
        else:
            x = torch.zeros(B, 1, 128, device=x.device)
        
        return x
    
    def forward_cross_attention(self, x1, x2):
        """Cross-attention between audio and visual.
        
        Args:
            x1: Audio embeddings (B, T, 128)
            x2: Visual embeddings (B, T, 128)
            
        Returns:
            Attended audio and visual embeddings
        """
        # Align temporal dimensions
        min_len = min(x1.size(1), x2.size(1))
        x1 = x1[:, :min_len, :]
        x2 = x2[:, :min_len, :]
        
        # Transpose for attention (T, B, C)
        x1 = x1.transpose(0, 1)
        x2 = x2.transpose(0, 1)
        
        # Cross attention
        x1_attended = self.crossA2V(x1, x2, x2)
        x2_attended = self.crossV2A(x2, x1, x1)
        
        # Transpose back (B, T, C)
        x1_attended = x1_attended.transpose(0, 1)
        x2_attended = x2_attended.transpose(0, 1)
        
        return x1_attended, x2_attended
    
    def forward_audio_visual_backend(self, x1, x2):
        """Backend fusion and classification.
        
        Args:
            x1: Attended audio embeddings (B, T, 128)
            x2: Attended visual embeddings (B, T, 128)
            
        Returns:
            Classification logits (B*T, 2)
        """
        # Concatenate audio and visual
        x = torch.cat([x1, x2], dim=2)  # (B, T, 256)
        
        B, T, C = x.shape
        x = x.view(B * T, C)
        
        x = self.fcAV(x)
        
        return x
    
    def forward_audio_backend(self, x):
        """Audio-only backend."""
        B, T, C = x.shape
        x = x.view(B * T, C)
        x = self.fcA(x)
        return x
    
    def forward_visual_backend(self, x):
        """Visual-only backend."""
        B, T, C = x.shape
        x = x.view(B * T, C)
        x = self.fcV(x)
        return x
