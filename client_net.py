import torch
import torch.nn as nn
import torch.optim as optim
import math


# ---------------------------------------------------------------
# Positional Encoding
# به هر توکن اطلاعات موقعیت زمانی میده
# ---------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------
# یک سلول AutoEncoder Transformer
# شامل: Multi-Head Attention + FeedForward
# ---------------------------------------------------------------
class TransformerAutoEncoderCell(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        # x: (batch, seq_len, d_model)
        # key_padding_mask: (batch, seq_len), True where position must be ignored in keys (PyTorch convention)
        # --- Attention ---
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        # --- FeedForward ---
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x  # (batch, seq_len, d_model)


# ---------------------------------------------------------------
# Client Transformer Network
#
# ورودی: x با شکل (batch, W, N)
#   W = طول پنجره زمانی
#   N = تعداد فیچرها (4 برای metavision، 5 برای carevue)
#
# مراحل:
#   1. Input Projection: (batch, W, N) → (batch, W, d_model)
#      یه ماتریس قابل آموزش که ابعاد کم N رو به d_model بالا میبره
#   2. اضافه کردن CLS token به ابتدا: seq_len = W+1
#   3. Positional Encoding
#   4. سه سلول AutoEncoder Transformer پشت سر هم
#   5. استخراج CLS از خروجی آخرین لایه → (batch, d_model)
# ---------------------------------------------------------------
class client_network(nn.Module):
    def __init__(
        self,
        w,                    # طول پنجره زمانی
        n_features_input,     # N = تعداد فیچرها
        d_model=64,           # بعد فضای مدل بعد از projection
        nhead=4,              # تعداد attention head
        dim_feedforward=128,  # بعد لایه feedforward داخل هر سلول
        num_cells=3,          # تعداد سلول‌های AutoEncoder
        dropout=0.1,
        lr=0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.w = w
        self.n_features = n_features_input

        # --- Input Projection ---
        # N معمولاً 4 یا 5 هست — خیلی کوچیکه برای transformer
        # این ماتریس قابل آموزش ابعاد رو بالا میبره
        self.input_proj = nn.Linear(n_features_input, d_model)

        # --- CLS Token ---
        # یه بردار قابل آموزش که همیشه اول سکوانس قرار میگیره
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- Positional Encoding ---
        # max_len = W + 1 (برای CLS)
        self.pos_encoding = PositionalEncoding(d_model, max_len=w + 2, dropout=dropout)

        # --- سلول‌های AutoEncoder Transformer ---
        self.cells = nn.ModuleList([
            TransformerAutoEncoderCell(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_cells)
        ])

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    @staticmethod
    def _prepend_cls_key_padding(pad_mask):
        """
        pad_mask: (batch, W) — 1 برای پنجرهٔ معتبر، 0 برای پَد خالی (قبل از CLS)
        خروجی: key_padding برای سکانس به طول W+1 (توکن اول = CLS هرگز ماسک نمی‌شود)
        """
        # True = آن موقعیت در attention نادیده گرفته شود
        kpm = torch.zeros(
            pad_mask.size(0), pad_mask.size(1) + 1,
            dtype=torch.bool,
            device=pad_mask.device,
        )
        kpm[:, 1:] = pad_mask < 0.5
        return kpm

    @staticmethod
    def _zero_pad_timesteps(x, pad_mask):
        """حداقل رسانی خروجی روی timestepهای نامعتبر تا گرادیان و نمایش تمیز بماند."""
        cls_h = x[:, :1, :]
        ts = x[:, 1:, :] * pad_mask.unsqueeze(-1)
        return torch.cat([cls_h, ts], dim=1)

    def forward(self, x, pad_mask=None):
        """
        x: (batch, W, N)
        pad_mask: (batch, W) — مثل خروجی new_dataset؛ اگر None باشد همه timestepها معتبر فرض می‌شوند.
        خروجی: cls_out با شکل (batch, d_model) — توکن CLS برای شبکهٔ کپسولی سرور
        """
        batch = x.size(0)

        if pad_mask is None:
            pad_mask = torch.ones(batch, self.w, device=x.device, dtype=x.dtype)

        # 1. Input Projection: (batch, W, N) → (batch, W, d_model)
        x = self.input_proj(x)

        # 2. CLS token در ابتدای سکانس
        cls = self.cls_token.expand(batch, -1, -1)

        # 3. Concat CLS: (batch, W+1, d_model)
        x = torch.cat([cls, x], dim=1)

        # 4. Positional Encoding
        x = self.pos_encoding(x)
        x = self._zero_pad_timesteps(x, pad_mask)

        key_padding_mask = self._prepend_cls_key_padding(pad_mask)

        # 5. سلول‌های ترنسفورمر با رعایت ماسک کلیدها و صفرکردن خروجی پَد بین لایه‌ها
        for cell in self.cells:
            x = cell(x, key_padding_mask=key_padding_mask)
            x = self._zero_pad_timesteps(x, pad_mask)

        # 6. بردار CLS برای split point
        cls_out = x[:, 0, :]

        return cls_out

    def train_one_batch(self, cls_out, grad):
        """
        cls_out: خروجی forward — (batch, d_model)
                 باید قبل از ارسال به سرور retain_grad() زده شده باشه
                 تا computational graph سالم بمونه
        grad: gradient برگشتی از سرور — (batch, d_model)
        """
        self.optimizer.zero_grad()
        # backward از طریق همون cls_out اصلی که graph داره
        cls_out.backward(grad)
        self.optimizer.step()
