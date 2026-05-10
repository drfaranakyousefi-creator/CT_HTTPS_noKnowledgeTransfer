import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import math


def filter_noisy_data(x, dataset_name):
    item_id = {
        'metavision': [
            220045,          # Heart Rate
            220210,          # Respiratory Rate
            220179, 220180,  # Non-invasive BP Mean
            220052,          # Arterial BP Mean
            220277           # SpO2
        ],
        'carevue': [
            211,        # Heart Rate
            618,        # Respiratory Rate
            52,         # Arterial BP Mean
            456,        # NBP Mean
            676, 678,   # Temperature
            646         # SpO2
        ]
    }
    filtered_df = x[x['itemid'].isin(item_id[dataset_name])].copy()
    return filtered_df


def extract_data_from_person(dataframe, W, dataset_name, target):
    if dataset_name == 'metavision':
        N = 4
    else:
        N = 5

    data = []
    label = []
    mask = []

    e = torch.zeros(N)
    x = torch.zeros(W, N)
    m = torch.zeros(W)
    s = 0

    for index, row in dataframe.iterrows():
        item_id = row['itemid']
        value = row['value']

        try:
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                continue
        except (ValueError, TypeError):
            continue

        # data order metavision: (HR, RR, NIBP Mean, ABP Mean)
        # data order carevue:    (HR, RR, ABP Mean, NBP Mean, Temp)

        if s >= W:
            s = 0
            m = torch.zeros(W)
            x = torch.zeros(W, N)
            e = torch.zeros(N)  # FIX: e هم reset بشه

        if (item_id == 646) or (item_id == 220277):  # SpO2
            if target == 'spO2':
                if s > 0:
                    data.append(x.clone())   # FIX: clone تا reference مشترک نداشته باشن
                    label.append(value)
                    mask.append(m.clone())
                    m = torch.zeros(W)
                    x = torch.zeros(W, N)
                    e = torch.zeros(N)
                    s = 0
            elif target == 'BP':
                e[0] = value
                x[s, :] = e.clone()
                m[s] = 1
                s += 1
            elif target == 'RR':
                e[1] = value
                x[s, :] = e.clone()
                m[s] = 1
                s += 1

        elif (item_id == 52) or (item_id == 220052):  # ABP Mean
            if target == 'BP':
                if s > 0:
                    data.append(x.clone())
                    label.append(value)
                    mask.append(m.clone())
                    m = torch.zeros(W)
                    x = torch.zeros(W, N)
                    e = torch.zeros(N)
                    s = 0
            else:
                e[0] = value
                x[s, :] = e.clone()
                m[s] = 1
                s += 1

        elif (item_id == 618) or (item_id == 220210):  # RR
            if target == 'RR':
                if s > 0:
                    data.append(x.clone())
                    label.append(value)
                    mask.append(m.clone())
                    m = torch.zeros(W)
                    x = torch.zeros(W, N)
                    e = torch.zeros(N)
                    s = 0
            else:
                e[1] = value
                x[s, :] = e.clone()
                m[s] = 1
                s += 1

        elif (item_id == 211) or (item_id == 220045):  # HR
            e[2] = value
            x[s, :] = e.clone()
            m[s] = 1
            s += 1

        elif (item_id == 220179) or (item_id == 220180):  # NIBP metavision
            e[3] = value
            x[s, :] = e.clone()
            m[s] = 1
            s += 1

        elif item_id == 456:  # NBP carevue
            e[3] = value
            x[s, :] = e.clone()
            m[s] = 1
            s += 1

        elif (item_id == 678) or (item_id == 676):  # Temperature
            e[4] = value
            x[s, :] = e.clone()
            m[s] = 1
            s += 1

    if len(data) > 0:
        data = torch.stack(data, dim=0)   # (samples, W, N)
        label = torch.tensor(label, dtype=torch.float)
        mask = torch.stack(mask, dim=0)   # (samples, W)
        return data, label, mask
    else:
        return None, None, None


def extract_data(dataset_name, df_chartevents, w, target, normalize=True):
    total_subject_ids = df_chartevents['subject_id'].unique()
    all_user_data = []
    all_labels = []
    all_mask = []

    for subject_id in total_subject_ids:
        subject_data = df_chartevents[df_chartevents['subject_id'] == subject_id]
        filtered_df = filter_noisy_data(subject_data, dataset_name)
        data, label, mask = extract_data_from_person(filtered_df, w, dataset_name, target)

        # FIX: قبلاً label != None بود که برای tensor کار نمیکنه
        if label is not None:
            all_labels.append(label)
            all_user_data.append(data)
            all_mask.append(mask)

    # FIX: اگه هیچ داده‌ای پیدا نشد crash نکنه
    if len(all_user_data) == 0:
        raise ValueError(f"هیچ داده‌ای برای dataset={dataset_name} و target={target} پیدا نشد.")

    data = torch.cat(all_user_data, dim=0)      # (total_samples, W, N)
    labels = torch.cat(all_labels, dim=0)        # (total_samples,)
    masks = torch.cat(all_mask, dim=0)           # (total_samples, W)

    if normalize:
        mean = data.mean()
        std = data.std()
        data = (data - mean) / (std + 1e-4)

    return data, labels, masks


class ICUDataset(Dataset):
    def __init__(self, data_in, label, mask):
        super().__init__()
        self.data = data_in
        self.label = label
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index, :, :]     # (W, N)
        y = self.label[index]           # scalar
        m = self.mask[index, :]         # (W,)
        return x, y, m


class data_preparing:
    def __init__(
        self, data_frame, dataset_name, w, test_size, target, batch_size, normalize=True
    ):
        x, y, mask = extract_data(
            dataset_name, data_frame, w, target, normalize=normalize
        )

        train_n = int((1 - test_size) * x.shape[0])

        train_dataset = ICUDataset(x[:train_n], y[:train_n], mask[:train_n])
        test_dataset  = ICUDataset(x[train_n:], y[train_n:], mask[train_n:])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)
