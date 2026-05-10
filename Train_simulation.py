import pandas as pd
import torch
import torch.nn as nn

from client_net import client_network
from new_dataset import data_preparing
from transmitter_simulation import Transmitter


def _parse_CT_HTTPS_positional(args, chartevents_kw):
    """
    قدیمی: (csv_path, w, dataset_name, batch_size [, server_url])
    جدید: (w, dataset_name, batch_size [, server_url])

    تشخیص: آرگومان اول str → قدیمی، int → جدید.
    server_url در positional اختیاری است؛ اگر نبود None برمی‌گردد (بعداً از kwargs یا './' پر می‌شود).
    """
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
            raise TypeError(
                "تماس قدیمی نامعتبر. انتظار: str مسیر، int w، str dataset، int batch."
            )
        chartevents_path = chartevents_kw if chartevents_kw is not None else args[0]
        w, dataset_name, batch_size = args[1], args[2], args[3]
        if n >= 5:
            if n > 5:
                raise TypeError(
                    "تماس قدیمی: حداکثر ۵ آرگومان positional (مسیر، w، dataset، batch، server_url)."
                )
            server_url = args[4]
        else:
            server_url = None
        return chartevents_path, w, dataset_name, batch_size, server_url

    if isinstance(args[0], int):
        if not (isinstance(args[1], str) and isinstance(args[2], int)):
            raise TypeError(
                "تماس جدید نامعتبر. انتظار: int w، str dataset_name، int batch_size."
            )
        chartevents_path = (
            chartevents_kw if chartevents_kw is not None else "./CHARTEVENTS.csv"
        )
        w, dataset_name, batch_size = args[0], args[1], args[2]
        if n >= 4:
            if n > 4:
                raise TypeError(
                    "تماس جدید: حداکثر ۴ آرگومان positional (w، dataset، batch، server_url)."
                )
            if not isinstance(args[3], str):
                raise TypeError(
                    "چهارمین آرگومان positional باید server_url (رشته مثل './') باشد."
                )
            server_url = args[3]
        else:
            server_url = None
        return chartevents_path, w, dataset_name, batch_size, server_url

    raise TypeError(
        "نوع آرگومان اول نامعتبر (باید int برای تماس جدید یا str مسیر CSV برای تماس قدیمی باشد)."
    )


class CT_HTTPS(nn.Module):
    """
    Client Transformer + شبکهٔ کپسولی؛ split learning محلی (بدون HTTP واقعی).

    تماس پیشنهادی جدید (کولَب):
        CT_HTTPS(w, 'carevue', batch_size, server_url,
                 chartevents_path='./CHARTEVENTS.csv', ...)
    تماس قدیمی هنوز پشتیبانی می‌شود:
        CT_HTTPS(csv_path, w, dataset_name, batch_size, ...)

    chartevents_path فقط یک‌بار از طریق kwargs خوانده می‌شود؛ با امضای تابع تکراری نمی‌شود.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        kw = dict(kwargs)
        chartevents_kw = kw.pop("chartevents_path", None)
        server_kw = kw.pop("server_url", None)

        chartevents_path, w, dataset_name, batch_size, server_url_pos = (
            _parse_CT_HTTPS_positional(args, chartevents_kw)
        )
        server_url = server_kw if server_kw is not None else (server_url_pos or "./")

        target = kw.pop("target", "spO2")
        lr = kw.pop("lr", 0.01)
        test_size = kw.pop("test_size", 0.2)
        normalize_data = kw.pop("normalize_data", True)
        client_lr = kw.pop("client_lr", None)
        server_lr = kw.pop("server_lr", None)
        d_model = kw.pop("d_model", 64)
        nhead = kw.pop("nhead", 4)
        dim_feedforward = kw.pop("dim_feedforward", 128)
        num_cells = kw.pop("num_cells", 3)
        dropout = kw.pop("dropout", 0.1)
        num_primary_caps = kw.pop("num_primary_caps", 8)
        primary_dim = kw.pop("primary_dim", 8)
        num_output_caps = kw.pop("num_output_caps", 4)
        output_dim = kw.pop("output_dim", 16)
        num_routing = kw.pop("num_routing", 3)
        server_fc_hidden1 = kw.pop("server_fc_hidden1", 128)
        server_fc_hidden2 = kw.pop("server_fc_hidden2", 64)
        device = kw.pop("device", None)

        if kw:
            unexpected = ", ".join(sorted(kw.keys()))
            raise TypeError(f"کلیدهای نامعتبر به CT_HTTPS: {unexpected}")

        _ = server_url
        if client_lr is None:
            client_lr = lr
        if server_lr is None:
            server_lr = lr
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if dataset_name == "metavision":
            n_features = 4
        elif dataset_name == "carevue":
            n_features = 5
        else:
            raise ValueError("dataset_name باید 'metavision' یا 'carevue' باشد")

        self.network = client_network(
            w=w,
            n_features_input=n_features,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_cells=num_cells,
            dropout=dropout,
            lr=client_lr,
        ).to(self.device)

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

        self.transmittion = Transmitter(
            device=self.device,
            d_model=d_model,
            server_lr=server_lr,
            num_primary_caps=num_primary_caps,
            primary_dim=primary_dim,
            num_output_caps=num_output_caps,
            output_dim=output_dim,
            num_routing=num_routing,
            fc_hidden1=server_fc_hidden1,
            fc_hidden2=server_fc_hidden2,
        )

        self.loss_fn = nn.MSELoss()

    def fit(self, epochs):
        history = {"loss_train": [], "loss_test": []}
        for epoch in range(epochs):
            self.train_one_epoch()
            loss_train, loss_test = self.evaluate_one_epoch()
            print(
                f"[epoch {epoch + 1}/{epochs}  train_loss={loss_train:.4f}  "
                f"test_loss={loss_test:.4f}]"
            )
            history["loss_train"].append(loss_train.item())
            history["loss_test"].append(loss_test.item())
        return history

    def train_one_epoch(self):
        self.network.train()
        for x, labels, pad_mask in self.data.train_loader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            pad_mask = pad_mask.to(self.device)

            cls_out = self.network(x, pad_mask=pad_mask)
            cls_out.retain_grad()
            grad = self.transmittion.send_data(cls_out, labels, status="train")
            self.network.train_one_batch(cls_out, grad.clone())

    def evaluate_one_epoch(self):
        self.network.eval()
        with torch.no_grad():
            loss_train = self._eval_loader(self.data.train_loader)
            loss_test = self._eval_loader(self.data.test_loader)
        return loss_train, loss_test

    def _eval_loader(self, loader):
        total_loss = 0
        total_n = 0
        for x, labels, pad_mask in loader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            pad_mask = pad_mask.to(self.device)

            cls_out = self.network(x, pad_mask=pad_mask)
            prediction = self.transmittion.send_data(cls_out, labels, status="test")
            total_loss += x.shape[0] * self.loss_fn(prediction, labels)
            total_n += x.shape[0]
        return total_loss / total_n
