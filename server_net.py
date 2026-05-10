import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------
# Squash activation — قلب Capsule Network
# ---------------------------------------------------------------
def squash(x, dim=-1):
    norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm = norm_sq.sqrt()
    scale = norm_sq / (1.0 + norm_sq)
    return scale * x / (norm + 1e-8)


# ---------------------------------------------------------------
# Primary Capsule Layer
# ---------------------------------------------------------------
class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_features, num_capsules, capsule_dim):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.capsules = nn.Linear(in_features, num_capsules * capsule_dim)

    def forward(self, x):
        out = self.capsules(x)
        out = out.view(x.size(0), self.num_capsules, self.capsule_dim)
        return squash(out)


# ---------------------------------------------------------------
# Dynamic Routing
# ---------------------------------------------------------------
class RoutingLayer(nn.Module):
    def __init__(self, num_input_caps, input_dim, num_output_caps, output_dim, num_routing=3):
        super().__init__()
        self.num_routing = num_routing
        self.num_output_caps = num_output_caps
        self.output_dim = output_dim
        self.W = nn.Parameter(
            torch.randn(1, num_input_caps, num_output_caps, output_dim, input_dim) * 0.01
        )

    def forward(self, x):
        batch = x.size(0)
        x_ = x.unsqueeze(2).unsqueeze(4)
        W_ = self.W.expand(batch, -1, -1, -1, -1)
        u_hat = torch.matmul(W_, x_).squeeze(-1)
        b = torch.zeros(batch, x.size(1), self.num_output_caps, device=x.device)
        for _ in range(self.num_routing):
            c = F.softmax(b, dim=2)
            c_ = c.unsqueeze(-1)
            s = (c_ * u_hat).sum(dim=1)
            v = squash(s)
            b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)
        return v


# ---------------------------------------------------------------
# Prediction Network با Capsule
# ---------------------------------------------------------------
class prediction_net(nn.Module):
    def __init__(
        self,
        d_model=64,
        lr=0.01,
        device=None,
        num_primary_caps=8,
        primary_dim=8,
        num_output_caps=4,
        output_dim=16,
        num_routing=3,
        fc_hidden1=128,
        fc_hidden2=64,
    ):
        super().__init__()
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        capsule_in = fc_hidden2

        self.fc_in = nn.Sequential(
            nn.Linear(d_model, fc_hidden1),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.LeakyReLU(),
        ).to(self.device)

        self.primary_caps = PrimaryCapsLayer(
            in_features=capsule_in,
            num_capsules=num_primary_caps,
            capsule_dim=primary_dim
        ).to(self.device)

        self.routing = RoutingLayer(
            num_input_caps=num_primary_caps,
            input_dim=primary_dim,
            num_output_caps=num_output_caps,
            output_dim=output_dim,
            num_routing=num_routing
        ).to(self.device)

        self.fc_out = nn.Linear(num_output_caps * output_dim, 1).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward_direct(self, cls_tensor):
        """
        ورودی مستقیم tensor با graph سالم — بدون JSON، بدون torch.tensor جدید.
        گراف محاسباتی از client تا اینجا کاملاً حفظ میشه.
        خروجی: (batch,)
        """
        h = self.fc_in(cls_tensor)
        h = self.primary_caps(h)
        h = self.routing(h)
        h = h.view(h.size(0), -1)
        output = self.fc_out(h)
        return output.squeeze(1)  # (batch,)

    def forward(self, cls_vector, label=None, status='test'):
        """
        متد قدیمی — فقط برای سازگاری با transmitter_simulation نگه داشته شده.
        در مسیر جدید (CT_HTTPS) از این استفاده نمیشه.
        """
        x = torch.tensor(cls_vector, dtype=torch.float, device=self.device)
        x.requires_grad_(True)

        h = self.fc_in(x)
        h = self.primary_caps(h)
        h = self.routing(h)
        h = h.view(h.size(0), -1)
        output = self.fc_out(h)

        if status == 'train':
            label = torch.tensor(label, dtype=torch.float, device=self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(output.squeeze(1), label)
            loss.backward()
            input_grad = x.grad.detach().cpu().tolist()
            self.optimizer.step()
            return {'grad': input_grad}
        else:
            return {'prediction': output.squeeze(1).detach().cpu().tolist()}
