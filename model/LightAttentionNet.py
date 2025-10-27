import torch
import torch.nn as nn
import torch.nn.functional as F

# def position_coding(x):
#     """
#     x: [B, T, D]
#     生成正弦位置编码
#     """
#     B, T, D = x.shape
#     pos = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, D, 2, device=x.device).float() * (-torch.log(torch.tensor(10000.0)) / D))
#     pe = torch.zeros(T, D, device=x.device)
#     pe[:, 0::2] = torch.sin(pos * div_term)
#     pe[:, 1::2] = torch.cos(pos * div_term)
#     return pe.unsqueeze(0).expand(B, -1, -1)  # [B, T, D]

def position_coding(x):
    num_token, num_dims = x.size(-2), x.size(-1)
    p = torch.zeros((1, num_token, num_dims))
    t = torch.arange(num_token, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(1e4, torch.arange(0, num_dims, 2, dtype=torch.float32) / num_dims)
    p[:, :, 0::2] = torch.sin(t)
    p[:, :, 1::2] = torch.cos(t)
    return p

class LightAttentionNet(nn.Module):
    def __init__(self, num_classes=10, device='cuda', include_top=True):
        super(LightAttentionNet, self).__init__()
        self.device = device
        self.embedding_dim = 64  # Transformer的d_model
        self.include_top = include_top

        # 把输入128维映射到Transformer的维度
        self.encoding1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU()
        )
        self.encoding2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU()
        )
        
        # 轻量Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=8,
            batch_first=True,
            dim_feedforward=256,
            dropout=0.1
        )

        self.transformer1 = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer2 = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 将Transformer输出映射到语义空间
        self.semantic1 = nn.Sequential(
            nn.Linear(self.embedding_dim * 128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.semantic2 = nn.Sequential(
            nn.Linear(self.embedding_dim * 128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 分类头
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        B, C, F, T = x.shape
        x = x.view(B, F, T)
        y = x.permute(0, 2, 1)  # [B, F, T]


        x = self.encoding1(x)           # [B, 128, 128]
        y = self.encoding2(y)

        x = x + position_coding(x).to(self.device)
        y = y + position_coding(y).to(self.device)

        x = self.transformer1(x)        # [B, 1, 64]
        y = self.transformer2(y)        # [B, 1, 64]

        x = x.view(B, -1)
        y = y.view(B, -1)              # [B, 64]

        x = self.semantic1(x)
        y = self.semantic2(y)           # [B, 64]

        semantic = torch.cat([x, y], dim=1)
        if self.include_top:
            predict = self.classifier(semantic)       # [B, num_classes]
        return predict


if False:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LightAttentionNet(num_classes=25, device=device, include_top=True).to(device)

    x = torch.randn(16, 1, 128, 128).to(device)  # batch_size=16
    out = model(x)
    print(out.shape)

    from torchsummary import summary
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LightAttentionNet(num_classes=25, include_top=True)
    model = model.to(device)

    # summary(model, input_size=(1, 512))
    inputs = torch.randn(1, 1, 128, 128).to(device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)
