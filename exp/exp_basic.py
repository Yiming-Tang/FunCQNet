import os
import torch
from models import DFQR, DFQRwoI, DFQR_new, DFQRadd


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'DFQR': DFQR,
            'DFQRwoI': DFQRwoI,
            'DFQR_new': DFQR_new,
            'DFQRadd': DFQRadd
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            device = self.args.device
            print('Use GPU: {}'.format(self.args.device))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
