import json
import torch
from server_net import prediction_net
import torch.nn as nn 

class Transmitter(nn.Module):
    """
    شبیه‌سازی کانال شبکه بین کلاینت و سروربدون درخواست HTTP:
    detach → لیست/json → شبکهٔ کپسولی (prediction_net).
    """

    def __init__(
        self,
        device,
        d_model=64,
        server_lr=0.01,
        num_primary_caps=8,
        primary_dim=8,
        num_output_caps=4,
        output_dim=16,
        num_routing=3,
        fc_hidden1=128,
        fc_hidden2=64,
    ):
        super().__init__()
        self.device = device
        self.model = prediction_net(
            d_model=d_model,
            lr=server_lr,
            device=device,
            num_primary_caps=num_primary_caps,
            primary_dim=primary_dim,
            num_output_caps=num_output_caps,
            output_dim=output_dim,
            num_routing=num_routing,
            fc_hidden1=fc_hidden1,
            fc_hidden2=fc_hidden2,
        )

    @staticmethod
    def data_to_json(x, label, status):
        x_copy = x.detach().cpu().tolist()
        label_copy = (
            label.detach().cpu().tolist() if hasattr(label, 'detach') else list(label)
        )
        data = {
            'cls_vector': x_copy,
            'label': label_copy if status == 'train' else [],
            'status': status,
        }
        return json.dumps(data)

    def send_data(self, x, label, status):
        data_json = self.data_to_json(x, label, status)
        received = json.loads(data_json)

        result = self.model(
            received['cls_vector'],
            received['label'],
            received['status'],
        )

        result_json = json.loads(json.dumps(result))

        if status == 'train':
            return torch.tensor(result_json['grad'], dtype=torch.float32, device=self.device)
        return torch.tensor(
            result_json['prediction'], dtype=torch.float32, device=self.device
        )
