import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.nn import functional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from kornia.losses import focal_loss
import seaborn as sns
from tqdm import tqdm


class RandomDatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.array = [i for i in range(len(dataset))]

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.array) == 0:
            self.array = [i for i in range(len(self.dataset))]
            raise StopIteration
        else:
            i = random.randint(0, len(self.array) - 1)
            idx = self.array[i]
            self.array.remove(idx)
            return self.dataset[idx]

class LSTMDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index()
        self.x1_col = ['HR_series', 'Resp_series', 'MAP_series', 'O2Sat_series', 'SBP_series']
        self.x2_col = ['age', 'gender', 'unit1', 'unit2', 'unknown unit','HospAdmTime', 'Final ICULOS', 'Temp_var', 'Temp_mean', 'WBC_not_null','WBC_mean', 'Lactate_not_null', 'BaseExcess_not_null']
        self.label_col = ['SepsisLabel']
        self.id_col = 'index'
    
    def __len__(self):
        return len(self.df)

    def stack_x1(self, x1_list):
        return torch.from_numpy(np.stack([i.to_numpy() for i in x1_list], axis=1)).type(torch.float)

    def __getitem__(self, idx):
        x1_val = self.df.loc[idx][self.x1_col]
        x2_val = torch.from_numpy(self.df.loc[idx][self.x2_col].values.astype(float)).type(torch.float)
        x1_len = self.df.loc[idx]['Final ICULOS']
        label = self.df.loc[idx][self.label_col]
        id = int(self.df.loc[idx][self.id_col])
        return x1_len, self.stack_x1(x1_val), x2_val, torch.from_numpy(label.values.astype(int)), id
    


class LSTMNet(torch.nn.Module):
    def __init__(self,
                type,
                lstm_input_size=5,
                lstm_hidden_size=30,
                lstm_num_layers=2,
                ind_feat_input_size=13,
                meeting_size=15,
                ):
        """
        t_input_size: time serires dim
        t_hidden_size: lstm hidden state
        num_layers: number of layers of LSTM

        """

        super(LSTMNet, self).__init__()

        self.type = type

        self.lstm = nn.LSTM(
                            input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            dropout=0.,
                            bidirectional=True
                            )

        self.lstm_nn = nn.Sequential(
                                    nn.Linear(2*lstm_hidden_size, 2*meeting_size),
                                    nn.ReLU(),
                                    nn.Linear(2*meeting_size, meeting_size),
                                    nn.ReLU(),
                                    )
        
        self.ind_feat_embedding = nn.Sequential(
                                        nn.Linear(ind_feat_input_size, meeting_size),
                                        nn.ReLU(),
                                        )
        
        if type == 'concat':
            self.meeting_nn = nn.Sequential(
                                nn.Linear(2*meeting_size, 2),
                                        )
        elif type == 'add' or type == 'ignore_ts':
                self.meeting_nn = nn.Sequential(
                                nn.Linear(meeting_size, 2),
                                        )
                
        self.x2_mean = torch.tensor([[61.668052, 0.555500, 0.303850, 0.311850, 0.384300, -50.975196, 38.235300, 0.264711, 36.839020, 0.064669, 11.010088,  0.025039, 0.053017]])
        self.x2_var = torch.tensor([[271.786744, 0.246932, 0.211536, 0.214610, 0.236625, 19621.742764, 472.696169, 0.163510, 0.291444, 0.002099, 37.194200, 0.003781, 0.011140]])


    def forward(self, x1, x2):
        # masked the null values and pass
        x2_norm = (x2 - self.x2_mean) *(1 / self.x2_var)
        ind_meet_vec = self.ind_feat_embedding(x2_norm)
        if torch.isnan(ind_meet_vec).any():
            print(f'x2: {x2}')


        # pass to the lstm
        lstm_out, _ = self.lstm(torch.nan_to_num(x1))
        lstm_out = torch.unsqueeze(lstm_out[0][-1], dim=0)
        lstm_meet_vec = self.lstm_nn(lstm_out)
        if torch.isnan(lstm_meet_vec).any():
            print(f'x1: {x1}')

        # concat/add/ignore two vectors and pass
        if self.type == 'concat':
            x = torch.concat([lstm_meet_vec, ind_meet_vec], dim=1)
        elif self.type == 'add': 
            x = lstm_meet_vec + ind_meet_vec
        elif self.type == 'ignore_ts':
            x = ind_meet_vec

        return self.meeting_nn(x)

###### TRAINING ######
def train_epoch(model, device, optimizer, training_dl, val_dl, batch_size=64):
    train_val_loss = []
    train_val_f1 = []
    optimizer.zero_grad()

    for phase in ["train", "val"]:
        y_pred_list = []
        y_gt_list = []
        total_loss = [0, 0]

        if phase == "train":
            model.train(True)
        else:
            model.eval()

        dl  = training_dl if phase == "train" else val_dl
        t_bar = tqdm(dl, total=len(dl))
        
        for i, (l, x1, x2, y, _) in enumerate(t_bar):

            x1 = torch.unsqueeze(x1.to(device), dim=0)
            x2 = torch.unsqueeze(x2.to(device), dim=0)
            y_gt = y.to(device)

            if phase == "train":
                
                model_output = model(x1, x2)
                y_prob = functional.softmax(model_output, dim=1)
                y_pred = torch.argmax(y_prob)
                loss = focal_loss(model_output, y_gt, alpha=0.5, gamma=2.0,) / batch_size
                loss.backward()

                if (i + 1) % batch_size == 0 or i == len(dl):
                    optimizer.step()
                    optimizer.zero_grad()

                y_pred_list.append(y_pred.detach().cpu().item())
                y_gt_list.append(y_gt.detach().cpu().item())
                total_loss[0] += loss.detach().cpu().item() 
                total_loss[1] += 1

            else:
                with torch.no_grad():

                    model_output = model(x1, x2)
                    y_prob = functional.softmax(model_output, dim=1)
                    y_pred = torch.argmax(y_prob)

                    loss = focal_loss(model_output, y_gt, alpha=0.5, gamma=2.0,)  / batch_size

                    y_pred_list.append(y_pred.detach().cpu().item())
                    y_gt_list.append(y_gt.detach().cpu().item())
                    total_loss[0] += loss.detach().cpu().item()
                    total_loss[1] += 1

            if len(y_pred_list) > 100:
                f1 = f1_score(y_gt_list, y_pred_list)
            else:
                f1 = 0

            t_bar.set_description(f"{phase}, loss: {total_loss[0] / total_loss[1]:.4f} f1: {f1:.2f}")

        train_val_loss.append(total_loss[0] / total_loss[1])
        train_val_f1.append(f1_score(y_gt_list, y_pred_list))

    return train_val_loss,train_val_f1

def save_graph(train_loss, test_loss, train_uas, test_uas, graph_name, epoch=None):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 4))
    ax[0].plot(train_loss, label="train")
    ax[0].plot(test_loss, label="test")
    ax[0].legend()
    if epoch is not None:
        ax[0].set_title("Model  Loss - Epoch {}".format(epoch))
    else:
        ax[0].set_title("Model  Loss")

    ax[1].plot(train_uas, label="train")
    ax[1].plot(test_uas, label="test")
    ax[1].legend()
    if epoch is not None:
        ax[1].set_title("Model F1 - Epoch {}".format(epoch))
    else:
        ax[1].set_title("Model F1")
    fig.savefig(f'/home/student/Data-analysis-and-presentation/HW1/graphs/{graph_name}.png')
    plt.close(fig)


def training(training_ds,
             val_ds,
             model,
             save_checkpoint_path,
             graph_name,
             epochs=20,
             early_stop=5,
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    training_dl = RandomDatasetIterator(training_ds)
    val_dl = RandomDatasetIterator(val_ds)

    best_f1 = 0
    best_epoch = 0
    train_val_loss_epochs = {"train": [], "val": []}
    train_val_uas_epochs = {"train": [], "val": []}
    for epoch in range(epochs):
        print(f"\n -- Epoch {epoch} --")
        train_val_loss, train_val_f1 = train_epoch(model, device, optimizer, training_dl, val_dl)
        if train_val_f1[1] > best_f1:
            torch.save(model.state_dict(), f'/home/student/Data-analysis-and-presentation/HW1/weights/{save_checkpoint_path}.pt')
            best_uas = train_val_f1[1]
            best_epoch = epoch

        train_val_loss_epochs['train'].append(train_val_loss[0])
        train_val_loss_epochs['val'].append(train_val_loss[1])
        train_val_uas_epochs['train'].append(train_val_f1[0])
        train_val_uas_epochs['val'].append(train_val_f1[1])

        if epoch - best_epoch >= early_stop:
            print(f"Early stop, best UAS on validation: {best_uas:.2f}, in epoch: {best_epoch}")
            break


    save_graph(train_val_loss_epochs["train"],
                         train_val_loss_epochs["val"],
                         train_val_uas_epochs["train"],
                         train_val_uas_epochs["val"],
                         graph_name,
                         best_epoch
                         )
    
def predict(model, ds):
    """
    Predict the ds 
    returning: y_pred_list, y_gt_list, id_list
    """
    y_pred_list = []
    y_gt_list = []
    id_list = []

    model.eval()
    
    for i in range(len(ds)):
            
            l, x1, x2, y_gt, id = ds[i]
            x1 = torch.unsqueeze(x1, dim=0)
            x2 = torch.unsqueeze(x2, dim=0)
            model_output = model(x1, x2)
            y_prob = functional.softmax(model_output, dim=1)
            y_pred = torch.argmax(y_prob)

            y_pred_list.append(y_pred.detach().cpu().item())
            y_gt_list.append(y_gt.detach().cpu().item())
            id_list.append(id)


    return y_pred_list, y_gt_list, id_list


def post_analysis(ds, col, model, change_val_fun, graph_name):
    df = ds.df.copy()
    ds = LSTMDataset(df)

    f1_vals = {'standart': None, 'after': None}
    y_pred_list, y_gt_list, id_list = predict(model, ds)

    cm = confusion_matrix(y_gt_list, y_pred_list)
    f = sns.heatmap(cm, annot=True, fmt='d')
    fig = f.get_figure()
    fig.savefig(f'/home/student/Data-analysis-and-presentation/HW1/graphs/{graph_name}_standart.png')
    plt.close(fig)

    f1_vals['standart'] = f1_score(y_pred_list, y_gt_list)

    ds.df.loc[:, col] = ds.df[col].apply(lambda v: change_val_fun(v))

    y_pred_list, y_gt_list, id_list = predict(model, ds)

    f1_vals['after'] = f1_score(y_pred_list, y_gt_list)

    cm = confusion_matrix(y_gt_list, y_pred_list)
    f = sns.heatmap(cm, annot=True, fmt='d')
    fig = f.get_figure()
    fig.savefig(f'/home/student/Data-analysis-and-presentation/HW1/graphs/{graph_name}_after.png')
    plt.close(fig)


    return f1_vals


