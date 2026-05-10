import pandas as pd
import torch
import torch.nn as nn

from client_net import client_network
from new_dataset import data_preparing
from transmitter_simulation import Transmitter


class CT_HTTPS(nn.Module):
    """
    Client Transformer + Head (split learning محلی، بدون درخواست HTTP واقعی).

    CT: ترنسفورمر کلاینت (چند بلوک encoder و توکن CLS در ابتدای سکانس).
    شبیه‌سازی «HTTPS» اینجا نام معماری است (نه پروتکل شبکه): مسیر جدا + گراف قطع
    با شبیه‌سازی JSON، سپس شبکهٔ کپسولی روی بردار CLS.

    جریان: new_dataset → client_network → (detach/شبیه‌سازی) → prediction_net (کپسول).
    """

    def __init__(
        self,
        chartevents_path,
        w,
        dataset_name,
        batch_size,
        target='spO2',
        test_size=0.2,
        normalize_data=True,
        client_lr=0.01,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        num_cells=3,
        dropout=0.1,
        server_lr=0.01,
        num_primary_caps=8,
        primary_dim=8,
        num_output_caps=4,
        output_dim=16,
        num_routing=3,
        server_fc_hidden1=128,
        server_fc_hidden2=64,
        device=None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        if dataset_name == 'metavision':
            n_features = 4
        elif dataset_name == 'carevue':
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
        history = {'loss_train': [], 'loss_test': []}
        for epoch in range(epochs):
            self.train_one_epoch()
            loss_train, loss_test = self.evaluate_one_epoch()
            print(
                f'[epoch {epoch + 1}/{epochs}  train_loss={loss_train:.4f}  '
                f'test_loss={loss_test:.4f}]'
            )
            history['loss_train'].append(loss_train.item())
            history['loss_test'].append(loss_test.item())
        return history

    def train_one_epoch(self):
        self.network.train()
        for x, labels, pad_mask in self.data.train_loader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            pad_mask = pad_mask.to(self.device)

            cls_out = self.network(x, pad_mask=pad_mask)
            cls_out.retain_grad()
            grad = self.transmittion.send_data(cls_out, labels, status='train')
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
            prediction = self.transmittion.send_data(cls_out, labels, status='test')
            total_loss += x.shape[0] * self.loss_fn(prediction, labels)
            total_n += x.shape[0]
        return total_loss / total_n
