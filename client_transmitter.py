import requests
import json
import time
import torch


class Transmitter:
    def __init__(self, server_url, device):
        self.server_url = server_url
        self.device = device

    def send_data(self, x, label, status):
        x_copy = x.detach().cpu().tolist()
        l = label.detach().cpu().tolist()

        if status == 'train':
            data = {
                'prediction_iput': x_copy,
                'label': l,
                'status': status
            }
        elif status == 'test':
            data = {
                'prediction_iput': x_copy,
                'label': [],
                'status': status
            }

        headers = {'Content-Type': 'application/json'}

        # FIX: retry منطقی اضافه شد — قبلاً هیچ retry نداشت و crash میکرد
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.server_url,
                    data=json.dumps(data),
                    headers=headers,
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()

                if status == 'train':
                    grad = result['grad']
                    return torch.tensor(grad).to(self.device)
                elif status == 'test':
                    prediction = torch.tensor(result['prediction']).to(self.device)
                    return prediction

            except requests.exceptions.Timeout:
                print(f'[Transmitter] Timeout — تلاش {attempt+1}/{max_retries}')
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # exponential backoff

            except requests.exceptions.RequestException as e:
                print(f'[Transmitter] خطای شبکه: {e} — تلاش {attempt+1}/{max_retries}')
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)