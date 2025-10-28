import torch
from torch import nn
import torch.nn.functional as F

def position_coding(x):
    num_token, num_dims = x.size(-2), x.size(-1)
    p = torch.zeros((1, num_token, num_dims))
    t = torch.arange(num_token, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(1e4, torch.arange(0, num_dims, 2, dtype=torch.float32) / num_dims)
    p[:, :, 0::2] = torch.sin(t)
    p[:, :, 1::2] = torch.cos(t)
    return p


class NET(nn.Module):
    def __init__(self, in_channels, input_size, semantic_dim, num_class, device):
        super(NET, self).__init__()
        self.input_size = input_size                     # [T, W]
        self.semantic_dim = semantic_dim
        self.num_class = num_class
        self.device = device
        self.in_channels = in_channels
        self.SA_1 = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dim_feedforward=256)
        self.SA_1 = nn.TransformerEncoder(self.SA_1, num_layers=3)
        self.SA_2 = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dim_feedforward=256)
        self.SA_2 = nn.TransformerEncoder(self.SA_2, num_layers=3)

        self.encoding_to_sa1 = nn.Sequential(
            nn.Linear(self.input_size[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.encoding_to_sa2 = nn.Sequential(
            nn.Linear(self.input_size[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.sa1_to_semantic = nn.Sequential(
            nn.Linear(int(self.input_size[0] * 64), 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )

        self.sa2_to_semantic = nn.Sequential(
            nn.Linear(int(self.input_size[1] * 64), 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )

        self.total_semantic = nn.Sequential(
            nn.Linear(self.semantic_dim * 3, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU(),
            nn.Linear(self.semantic_dim, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )
        self.encoder_d1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.encoder_d3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.encoder_d5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.encoder_to_semantic = nn.Sequential(
            nn.Linear(128*3, self.semantic_dim*2),
            nn.BatchNorm1d(self.semantic_dim*2),
            nn.ReLU(),
            nn.Linear(self.semantic_dim*2, self.semantic_dim),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )
        self.semantic_to_classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, self.num_class)
        )

    def forward(self, x):
        y = x.squeeze(1)
        z = y.permute(0, 2, 1)

        encoder1 = self.encoder_d1(x)# [32, 1, 512, 512]
        encoder2 = self.encoder_d3(x)
        encoder3 = self.encoder_d5(x)
        encoder_output = torch.cat([encoder1, encoder2, encoder3], dim=1)
        x = F.adaptive_avg_pool2d(encoder_output, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.encoder_to_semantic(x)
        expand_x = F.adaptive_avg_pool2d(encoder_output, (1, 1))
        expand_x = expand_x.view(expand_x.shape[0], -1)
        y = self.encoding_to_sa1(y)                 # [B, T, 64]
        z = self.encoding_to_sa2(z)                 # [B, W, 64]
        y = y + position_coding(y).to(self.device)
        z = z + position_coding(z).to(self.device)
        y = self.SA_1(y)                            # [B, T, 64]
        z = self.SA_2(z)                            # [B, W, 64]
        y = y.view(y.shape[0], -1)                  # [B, T*64]
        z = z.view(z.shape[0], -1)                  # [B, W*64]
        y = self.sa1_to_semantic(y)                 # [B, T*64] -> [B, semantic dim]
        z = self.sa2_to_semantic(z)                 # [B, W*64] -> [B, semantic dim]
        semantic = torch.cat([x, y], dim=1)
        semantic = torch.cat([semantic, z], dim=1)
        semantic = self.total_semantic(semantic)  # [B, 128*3] -> [B, 128]
        predict = self.semantic_to_classifier(semantic)

        return predict

if True:
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=10, device=device).to(device)

    x = torch.randn(16, 1, 512, 512).to(device)  # batch_size=16
    out = model(x)
    print(out.shape)

    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    # summary(model, input_size=(1, 512))
    inputs = torch.randn(1, 1, 512, 512).to(device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)