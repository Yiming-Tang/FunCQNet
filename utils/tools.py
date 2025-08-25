import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import sys
import os
from datetime import datetime

class NewDataset(Dataset):
    def __init__(self, data_list, Y_pred_all, num_tau):
        self.data_list = data_list
        self.Y_pred_all = Y_pred_all
        self.num_tau = num_tau

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        X = torch.tensor(item['X'], dtype=torch.float32)      # (t, p)
        t = torch.tensor(item['t'], dtype=torch.float32)      # (t,)
        Z = torch.tensor(item['Z'], dtype=torch.float32)      # (q,)
        Y = torch.tensor(item['Y'], dtype=torch.float32)      # (1,)
        C = torch.tensor(item['C'], dtype=torch.float32)      # (1,)
        y_pred = self.Y_pred_all[idx, :self.num_tau]
        interc = torch.ones(1)                   # intercept

        return {
            'Y': Y,
            'C': C,
            'X': X,
            't': t,
            'Z': Z,
            'I': interc,
            'Yp' : y_pred
        }



def pad_and_convert_to_dict(data_list, target_len=None):
    """
    Convert variable-length list data:
    [{'X': (Ti, p), 't': (Ti,), 'Z': (q,), 'Y': (1,), 'C': (1,)}]
    into padded uniform-length dictionary format:
    {
        'X': (n, L, p),
        't': (n, L),
        'Z': (n, q),
        'Y': (n, 1),
        'C': (n, 1),
        'length': (n,)
    }
    target_len: target padding length; if None or smaller than the max length, 
                use the maximum sequence length.
    """
    # 1. Find the maximum length of t
    max_len = max(item['t'].shape[0] for item in data_list)
    final_len = max_len if target_len is None or target_len < max_len else target_len

    padded_X, padded_t, padded_Z, padded_Y, padded_C, lengths = [], [], [], [], [], []

    # 2. Iterate over data and pad each sample
    for item in data_list:
        # Convert to tensors
        X = torch.as_tensor(item['X'], dtype=torch.float32)
        t = torch.as_tensor(item['t'], dtype=torch.float32)
        Z = torch.as_tensor(item['Z'], dtype=torch.float32)
        Y = torch.as_tensor(item['Y'], dtype=torch.float32)
        C = torch.as_tensor(item['C'], dtype=torch.float32)

        # If X is 1D, expand it to 2D
        if X.dim() == 1:
            X = X.unsqueeze(-1)  # (Ti,) -> (Ti, 1)

        length = X.shape[0]
        pad_len = final_len - length

        # Pad X -> (final_len, p)
        if pad_len > 0:
            pad_X = torch.cat([X, torch.zeros(pad_len, X.shape[1], device=X.device)], dim=0)
            pad_t = torch.cat([t, t[-1].repeat(pad_len).to(t.device)], dim=0)
        else:
            pad_X = X
            pad_t = t

        padded_X.append(pad_X)
        padded_t.append(pad_t)
        padded_Z.append(Z)
        padded_Y.append(Y)
        padded_C.append(C)
        lengths.append(length)

    # 3. Convert to dictionary format
    return {
        'X_all': torch.stack(padded_X, dim=0),   # (n, L, p)
        't_all': torch.stack(padded_t, dim=0),   # (n, L)
        'Z_all': torch.stack(padded_Z, dim=0),   # (n, q)
        'Y_all': torch.stack(padded_Y, dim=0),   # (n, 1)
        'C_all': torch.stack(padded_C, dim=0),   # (n, 1)
        'length': torch.tensor(lengths)          # (n,)
    }, final_len

    

class Loss_DFQR(nn.Module):
    def __init__(self, taugrd):
        super().__init__()
        self.taugrd = taugrd

    def forward(self, Y, y_pred, y_pred0, C):
        Y = torch.log(Y)
        n,_,_ = y_pred.shape

        taugrd0 = torch.cat([torch.zeros(1, dtype=self.taugrd.dtype, device=self.taugrd.device), self.taugrd])
        dh = -torch.log(1 - taugrd0[1:]) + torch.log(1 - taugrd0[:-1])
        e1 = Y.squeeze() - y_pred.squeeze()
        indicator = ((e1 <= 0) & (C.squeeze() == 1)).float()

        for j in range(1, len(taugrd0)):
            mask = (Y.view(-1, 1).squeeze() >= y_pred0[:,(j-1)].view(-1, 1).squeeze()).float()
            u = mask * dh[j-1]
            indicator = indicator - u

        loss = torch.dot(e1, -indicator)
        return loss / n


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 30))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss




def save_print(base_dir, setting):
    safe_setting = setting.replace("/", "_").replace("\\", "_")
    log_dir = os.path.join(base_dir, safe_setting)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"log_{timestamp}.txt")

    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)
    print(f"[Logger initialized] Logs will be saved to {log_path}")