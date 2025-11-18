from exp.exp_basic import Exp_Basic
from utils.tools import Loss_DFQR, EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import time
import os
import numpy as np



class Exp_main(Exp_Basic):
    def __init__(self, args, train_loader, val_loader, test_loader, 
                 train_loader_pred, val_loader_pred, test_loader_pred):
        super(Exp_main, self).__init__(args)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.train_loader_pred = train_loader_pred
        self.val_loader_pred   = val_loader_pred
        self.test_loader_pred  = test_loader_pred
        self.tau_grid = torch.tensor([round(i * 1 / (args.num_quantiles+1), 6) for i in range(1, int(args.tau * (args.num_quantiles+1)) + 1)])
        self.device = args.device
        self.tau = args.tau

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model.to(self.device)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return Loss_DFQR(taugrd=self.tau_grid)

    def vali(self, val_loader, val_loader_pred, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, ((Ib, Xb, tb, Zb, Yb, Cb), (Ypb,)) in enumerate(zip(val_loader, val_loader_pred)):
                Ib, Xb, Zb = Ib.to(self.device), Xb.to(self.device), Zb.to(self.device)
                tb = tb.to(self.device)
                Yb = Yb.to(self.device).unsqueeze(-1)
                Cb = Cb.to(self.device).unsqueeze(-1)
                Ypb = Ypb.to(self.device).unsqueeze(-1)

                y_pred, beta_pred,_ = self.model(Ib, Xb, tb, Zb)
                loss = criterion(Yb,y_pred,Ypb,Cb).cpu()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        self.model = self._build_model()
        path = os.path.join(self.args.save_path, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, ((Ib, Xb, tb, Zb, Yb, Cb), (Ypb,)) in enumerate(zip(self.train_loader, self.train_loader_pred)):
                iter_count += 1
                optimizer.zero_grad()
                Ib, Xb, Zb = Ib.to(self.device), Xb.to(self.device), Zb.to(self.device)
                tb = tb.to(self.device)
                Yb = Yb.to(self.device).unsqueeze(-1)
                Cb = Cb.to(self.device).unsqueeze(-1)
                Ypb = Ypb.to(self.device).unsqueeze(-1)
                    
                y_pred, beta_pred,_ = self.model(Ib, Xb, tb, Zb)

                loss = criterion(Yb,y_pred,Ypb,Cb)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.val_loader, self.val_loader_pred, criterion)
            test_loss = self.vali(self.test_loader, self.test_loader_pred, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path + '/' + f"{self.tau:.2f}" + '.pth')
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + f"{self.tau:.2f}" + '.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
        

    def test(self, setting):

        self.model.load_state_dict(torch.load(os.path.join('./save_models/', setting + '/' + f"{self.tau:.2f}" + '.pth')))

        test_loss = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for i, ((Ib, Xb, tb, Zb, Yb, Cb), (Ypb,)) in enumerate(zip(self.test_loader, self.test_loader_pred)):
                Ib, Xb, Zb = Ib.to(self.device), Xb.to(self.device), Zb.to(self.device)
                tb = tb.to(self.device)
                Yb = Yb.to(self.device).unsqueeze(-1)
                Cb = Cb.to(self.device).unsqueeze(-1)
                Ypb = Ypb.to(self.device).unsqueeze(-1)

                y_pred, beta_pred,_ = self.model(Ib, Xb, tb, Zb)
                loss = criterion(Yb,y_pred,Ypb,Cb)
                test_loss.append(loss.item())
        test_loss = np.average(test_loss)

        print(f"Test Loss: {test_loss:.6f}")
        test_loss_path = os.path.join(folder_path, 'test_loss_all.npy')
        if abs(self.tau - 1 / (self.args.num_quantiles+1)) < 1e-8:  
            np.save(test_loss_path, np.array([]))
            
        if os.path.exists(test_loss_path):
            existing_losses = np.load(test_loss_path)
            updated_losses = np.append(existing_losses, test_loss)
        else:
            updated_losses = np.array([test_loss])
        np.save(test_loss_path, updated_losses)

        return

    def updateY_pred(self, setting, I_all, X_all, t_all, Z_all):
        self.model.load_state_dict(torch.load(os.path.join('./save_models/', setting + '/' + f"{self.tau:.2f}" + '.pth')))
        self.model.eval()
        with torch.no_grad():
            I_all = I_all.to(self.device)
            X_all = X_all.to(self.device)
            Z_all = Z_all.to(self.device)
            t_all = t_all.to(self.device)
    
            y_pred_new, beta_pred, _ = self.model(I_all, X_all, t_all, Z_all)
            y_pred_new = y_pred_new.squeeze(2).detach().cpu()
        
        return y_pred_new
