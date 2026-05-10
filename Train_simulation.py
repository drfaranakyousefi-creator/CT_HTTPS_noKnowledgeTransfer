import pandas as pd
import torch
import torch.nn as nn

from client_net import client_network
from new_dataset import data_preparing
from transmitter_simulation import Transmitter
from server_net import prediction_net


def lr_at_epoch(epoch_1based, lr_spec):
    """
    lr_spec: عدد ثابت یا dict مثل {10: 0.01, 20: 0.001, 30: 0.0001}
    معنی dict (اپک از 1 شروع می‌شود):
      اپک‌های 1..10   → همان LR زیر کلید 10
      اپک‌های 11..20 → LR کلید 20
      بعد از آخرین milestone → آخرین LR در دیکشنری
    اگر lr_spec عدد باشد، همان مقدار برای هر اپک برمی‌گردد.
    """
    if lr_spec is None:
        raise ValueError("lr / lr_spec نمی‌تواند None باشد.")
    if not isinstance(lr_spec, dict):
        return float(lr_spec)
    if not lr_spec:
        raise ValueError("دیکشنری lr خالی است.")
    milestones = sorted(lr_spec.keys(), key=lambda x: int(x))
    for m in milestones:
        if epoch_1based <= int(m):
            return float(lr_spec[m])
    return float(lr_spec[milestones[-1]])


def _initial_scalar_lr(lr, override):
    if override is not None:
        return float(override)
    return lr_at_epoch(1, lr)


def _parse_CT_HTTPS_positional(args, chartevents_kw):
    n = len(args)
    if n < 3:
        raise TypeError(
            "حداقل ۳ آرگومان positional لازم است: "
            "جدید (w, dataset_name, batch_size) یا قدیمی (csv_path, w, dataset_name, batch_size …)."
        )

    if isinstance(args[0], str):
        if n < 4:
            raise TypeError("تماس قدیمی: بعد از مسیر CSV باید w, dataset_name, batch_size بیاید.")
        if not (
            isinstance(args[1], int)
            and isinstance(args[2], str)
            and isinstance(args[3], int)
        ):
            raise TypeError("تماس قدیمی نامعتبر. انتظار: str مسیر، int w، str dataset، int batch.")
        chartevents_path = chartevents_kw if chartevents_kw is not None else args[0]
        w, dataset_name, batch_size = args[1], args[2], args[3]
        server_url = args[4] if n >= 5 else None
        return chartevents_path, w, dataset_name, batch_size, server_url

    if isinstance(args[0], int):
        if not (isinstance(args[1], str) and isinstance(args[2], int)):
            raise TypeError("تماس جدید نامعتبر. انتظار: int w، str dataset_name، int batch_size.")
        chartevents_path = chartevents_kw if chartevents_kw is not None else "./CHARTEVENTS.csv"
        w, dataset_name, batch_size = args[0], args[1], args[2]
        server_url = args[3] if n >= 4 else None
        return chartevents_path, w, dataset_name, batch_size, server_url

    raise TypeError("نوع آرگومان اول نامعتبر.")


class CT_HTTPS(nn.Module):
    """
    Client Transformer + شبکهٔ کپسولی.
    گراف محاسباتی یکپارچه — بدون شکستن در split point.
    یک optimizer مشترک برای هر دو شبکه.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        kw = dict(kwargs)
        chartevents_kw = kw.pop("chartevents_path", None)
        server_kw = kw.pop("server_url", None)

        chartevents_path, w, dataset_name, batch_size, server_url_pos = (
            _parse_CT_HTTPS_positional(args, chartevents_kw)
        )
        _ = server_kw or server_url_pos  # نگه داشته شده برای سازگاری، استفاده نمیشه

        target          = kw.pop("target", "spO2")
        lr              = kw.pop("lr", 0.01)
        test_size       = kw.pop("test_size", 0.2)
        normalize_data  = kw.pop("normalize_data", True)
        client_lr       = kw.pop("client_lr", None)
        server_lr       = kw.pop("server_lr", None)
        d_model         = kw.pop("d_model", 64)
        nhead           = kw.pop("nhead", 4)
        dim_feedforward = kw.pop("dim_feedforward", 128)
        num_cells       = kw.pop("num_cells", 3)
        dropout         = kw.pop("dropout", 0.1)
        num_primary_caps  = kw.pop("num_primary_caps", 8)
        primary_dim       = kw.pop("primary_dim", 8)
        num_output_caps   = kw.pop("num_output_caps", 4)
        output_dim        = kw.pop("output_dim", 16)
        num_routing       = kw.pop("num_routing", 3)
        server_fc_hidden1 = kw.pop("server_fc_hidden1", 128)
        server_fc_hidden2 = kw.pop("server_fc_hidden2", 64)
        device            = kw.pop("device", None)

        if kw:
            raise TypeError(f"کلیدهای نامعتبر به CT_HTTPS: {', '.join(sorted(kw.keys()))}")

        # ── lr اولیه ──────────────────────────────────────────────
        self._lr_spec = lr
        init_lr = _initial_scalar_lr(lr, client_lr)
        # client_lr و server_lr جداگانه دیگه نیازی نیستن (optimizer مشترک داریم)
        # اما اگه کسی server_lr جداگانه داد، میانگین می‌گیریم
        if server_lr is not None:
            init_lr = (init_lr + float(server_lr)) / 2

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dataset_name == "metavision":
            n_features = 4
        elif dataset_name == "carevue":
            n_features = 5
        else:
            raise ValueError("dataset_name باید 'metavision' یا 'carevue' باشد")

        # ── Client Network ─────────────────────────────────────────
        self.network = client_network(
            w=w,
            n_features_input=n_features,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_cells=num_cells,
            dropout=dropout,
            lr=init_lr,          # optimizer داخلی client دیگه استفاده نمیشه
        ).to(self.device)

        # ── Server Network (مستقیم، بدون Transmitter) ──────────────
        self.prediction = prediction_net(
            d_model=d_model,
            lr=init_lr,          # optimizer داخلی server هم دیگه استفاده نمیشه
            device=self.device,
            num_primary_caps=num_primary_caps,
            primary_dim=primary_dim,
            num_output_caps=num_output_caps,
            output_dim=output_dim,
            num_routing=num_routing,
            fc_hidden1=server_fc_hidden1,
            fc_hidden2=server_fc_hidden2,
        ).to(self.device)

        # ── Optimizer مشترک — یه بار ساخته میشه، state حفظ میشه ───
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.prediction.parameters()),
            lr=init_lr,
        )

        # ── Data ───────────────────────────────────────────────────
        df_chartevents = pd.read_csv(chartevents_path)
        self.data = data_preparing(
            df_chartevents,
            dataset_name,
            w,
            test_size=test_size,
            target=target,
            batch_size=batch_size,
            normalize=normalize_data,
        )

        self.loss_fn = nn.MSELoss()

    def _apply_learning_rate(self, lr_value):
        """به‌روزرسانی lr روی optimizer مشترک."""
        v = float(lr_value)
        for g in self.optimizer.param_groups:
            g["lr"] = v

    def fit(self, epochs):
        history = {"loss_train": [], "loss_test": []}
        for epoch in range(epochs):
            ep = epoch + 1
            lr_now = lr_at_epoch(ep, self._lr_spec)
            self._apply_learning_rate(lr_now)
            self.train_one_epoch()
            loss_train, loss_test = self.evaluate_one_epoch()
            print(
                f"[epoch {ep}/{epochs}  lr={lr_now:.6f}"
                f"  train_loss={loss_train:.4f}  test_loss={loss_test:.4f}]"
            )
            history["loss_train"].append(loss_train.item())
            history["loss_test"].append(loss_test.item())
        return history

    def train_one_epoch(self):
        self.network.train()
        self.prediction.train()

        for x, labels, pad_mask in self.data.train_loader:
            x        = x.to(self.device)
            labels   = labels.to(self.device)
            pad_mask = pad_mask.to(self.device)

            self.optimizer.zero_grad()

            # ── Forward pass یکپارچه — گراف نمیشکنه ──────────────
            cls_out = self.network(x, pad_mask=pad_mask)   # (batch, d_model)
            pred    = self.prediction.forward_direct(cls_out)  # (batch,)

            loss = self.loss_fn(pred, labels)
            loss.backward()
            self.optimizer.step()

    def evaluate_one_epoch(self):
        self.network.eval()
        self.prediction.eval()
        with torch.no_grad():
            loss_train = self._eval_loader(self.data.train_loader)
            loss_test  = self._eval_loader(self.data.test_loader)
        return loss_train, loss_test

    def _eval_loader(self, loader):
        total_loss = 0
        total_n    = 0
        for x, labels, pad_mask in loader:
            x        = x.to(self.device)
            labels   = labels.to(self.device)
            pad_mask = pad_mask.to(self.device)

            cls_out = self.network(x, pad_mask=pad_mask)
            pred    = self.prediction.forward_direct(cls_out)

            total_loss += x.shape[0] * self.loss_fn(pred, labels)
            total_n    += x.shape[0]
        return total_loss / total_n
