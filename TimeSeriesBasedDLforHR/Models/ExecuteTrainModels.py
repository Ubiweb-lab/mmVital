import os

import torch
import torch.optim as optim
from tqdm import tqdm

import sys

sys.path.append('/home/syt0722/Weichun/60pts')
from Dataset.PrepareTrainData import PrepareTrainData
from Models.ModelNames import ModelNames

import matplotlib.pyplot as plt
from datetime import datetime

seed=42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子


class ExecuteTrainModels:
    def __init__(self, model, model_name, participant_id, is_shuffle=False, epochs=500):
        self.transformer_base = False
        self.participant_id = participant_id
        self.epochs = epochs
        self.model = model
        self.model_name = model_name
        self.lr = self.model.lr
        self.loss_fun = self.model.loss_fun
        self.optimizer = self.initialize_model()

        self.train_loader, self.val_loader = self.initialize_dataloader(participant_id, is_shuffle=is_shuffle)

        self.formatted_time = datetime.now().strftime("%m%d%H%M")

    def initialize_dataloader(self, participant_id=-1, is_shuffle=False):

        if (self.model_name == ModelNames.Transformer.value or
                self.model_name == ModelNames.Informer.value or
                self.model_name == ModelNames.Reformer.value):
            self.transformer_base = True

        if self.transformer_base is True:
            ptd = PrepareTrainData(is_date=True, is_shuffle=is_shuffle)
            return ptd.get_cross_dataloaders_add_date(participant_id)

        else:
            ptd = PrepareTrainData(is_shuffle=is_shuffle)
            return ptd.get_cross_dataloaders(participant_id)
        # return ptd.get_cross_with_date_dataloaders(participant_id)

    def initialize_model(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fun = self.loss_fun.cuda()
        optimizer = optim.ASGD(self.model.parameters(), lr=self.lr)
        if self.transformer_base is True:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        return optimizer

    def start_training(self):
        train_losses = []
        validate_losses = []
        epoch_counter = []
        best_v_loss = float('inf')
        best_t_loss = float('inf')

        current_dir = os.path.dirname(__file__)
        if self.participant_id == -1:
            save_path = os.path.join(current_dir, self.model_name, "trained_models", "idea_noise")
        else:
            save_path = os.path.join(current_dir, self.model_name, "trained_models")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_file_name = self.model_name + "_model_" + self.formatted_time + "_val_" + str(self.participant_id) + ".tar"
        save_model_path = os.path.join(save_path, model_file_name)

        best_t_epoch = 0
        best_v_epoch = 0

        for epoch in tqdm(range(self.epochs)):
            t_loss = self.train_per_epoch()
            v_loss = self.validate_per_epoch()

            if v_loss < best_v_loss:
                best_v_loss = v_loss
                best_v_epoch = epoch
            if t_loss < best_t_loss:
                best_t_loss = t_loss
                best_t_epoch = epoch
                torch.save(self.model.state_dict(), save_model_path)  # 保存训练后的模型

            if (epoch + 1) % 1 == 0:
                print("t_loss: " + str(round(t_loss, 4)) + ", v_loss: " + str(round(v_loss, 4)))

            if t_loss > 20000:
                t_loss = 20000

            train_losses.append(t_loss)
            validate_losses.append(v_loss)
            epoch_counter.append(epoch)
        print(f"best_t_epoch = {best_t_epoch}, best_v_epoch = {best_v_epoch}")
        return train_losses, validate_losses, epoch_counter

    def train_per_epoch(self):
        self.model.train()
        loss_batch_sum = 0.

        for index, batch in enumerate(self.train_loader):
            # if index == 113:
            #     a = 2;
            if self.transformer_base is True:
                X_batch, y_batch, X_batch_mask, y_batch_mask = batch[0], batch[1], batch[2], batch[3]
                if torch.cuda.is_available():
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                    X_batch_mask = X_batch_mask.cuda()
                    y_batch_mask = y_batch_mask.cuda()

                # decoder input
                dec_inp = torch.zeros_like(y_batch[:, -1:, :]).float()
                dec_inp = torch.cat([y_batch[:, :self.model.label_len, :], dec_inp], dim=1).float()

                preds_batch = self.model(X_batch, X_batch_mask, dec_inp, y_batch_mask, y_batch)

                f_dim = -1 if self.model.features == 'MS' else 0
                preds_batch = preds_batch[:, -self.model.pred_len:, f_dim:]
                y_batch = y_batch[:, -self.model.pred_len:, f_dim:]
            else:
                X_batch, y_batch = batch[0], batch[1]
                if torch.cuda.is_available():
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                preds_batch = self.model(X_batch)  # 2. 预测

            loss = self.loss_fun(preds_batch, y_batch)  # 3. 计算 loss
            self.optimizer.zero_grad()  # 4. 每一次 loop, 都重置 gradient
            loss.backward()  # 5. 反向传播，计算并更新 gradient 为 True 的参数值

            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=50)  # 在更新权重之前对梯度进行裁剪，使其不超过 50
            self.optimizer.step()  # 6. 更新 参数值

            loss_batch_sum += loss.item()

        return loss_batch_sum

    def validate_per_epoch(self):
        self.model.eval()
        loss_batch_sum = 0.
        with torch.no_grad():
            for index, batch in enumerate(self.val_loader):
                if self.transformer_base is True:
                    X_batch, y_batch, X_batch_mask, y_batch_mask = batch[0], batch[1], batch[2], batch[3]
                    if torch.cuda.is_available():
                        X_batch = X_batch.cuda()
                        y_batch = y_batch.cuda()
                        X_batch_mask = X_batch_mask.cuda()
                        y_batch_mask = y_batch_mask.cuda()

                    # decoder input
                    dec_inp = torch.zeros_like(y_batch[:, -1:, :]).float()
                    dec_inp = torch.cat([y_batch[:, :self.model.label_len, :], dec_inp], dim=1).float()

                    preds_batch = self.model(X_batch, X_batch_mask, dec_inp, y_batch_mask, y_batch)

                    f_dim = -1 if self.model.features == 'MS' else 0
                    preds_batch = preds_batch[:, -self.model.pred_len:, f_dim:]
                    y_batch = y_batch[:, -self.model.pred_len:, f_dim:]
                else:
                    X_batch, y_batch = batch[0], batch[1]

                    if torch.cuda.is_available():
                        X_batch = X_batch.cuda()
                        y_batch = y_batch.cuda()

                    with torch.inference_mode():  # 关闭 gradient tracking
                        preds_batch = self.model(X_batch)  # 2. 预测

                loss = self.loss_fun(preds_batch, y_batch)  # 3. 计算 loss
                loss_batch_sum += loss.item()

        return loss_batch_sum

    def visualize_loss(self, train_loss, validation_loss):
        plt.figure(figsize=(10, 3))
        plt.plot(train_loss, color='green', label='train_loss')
        plt.plot(validation_loss, color='blue', label='validation_loss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig('loss_plot' + self.formatted_time + '.png')
        plt.show()

    def __find_batch_mark(self, X_batch, y_batch):
        X_batch_mark = X_batch[:, :, 0:2]
        y_batch_mark = X_batch[:, 0:1, 0:1]
        return X_batch_mark, y_batch_mark

# et = ExecuteTrainLSTM()
# t_loss, v_loss, _ = et.start_training()
# et.visualize_loss(t_loss, v_loss)
# et.evaluate_preds()

# X_data, y_data = PrepareTrainData(is_shuffle=True).load_data(isEval=False)
