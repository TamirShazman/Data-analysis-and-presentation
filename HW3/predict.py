from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
import pandas as pd
from dataset import HW3Dataset


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(128, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, int(hidden_channels/2))
        self.conv3 = GCNConv(int(hidden_channels/2), int(hidden_channels/4))
        self.conv4 = GCNConv(int(hidden_channels/4), int(hidden_channels/8))
        self.conv5 = GCNConv(int(hidden_channels/8), 40)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv5(x, edge_index)
        return x


def predict(model, dataset):
    model.eval()   
    out = model(dataset.x, dataset.edge_index)
    pred = out.argmax(dim=1)
    return [i for i in range(len(pred))], pred


def main():
    test_ds = dataset = HW3Dataset(root='data/hw3/')

    model = GCN(hidden_channels=1024)
    try:
        model.load_state_dict(torch.load('../HW3/exp/GNC/models/trained_model.pt')) #../Data-analysis-and-presentation/HW3/exp/MLP/models/trained_model.pt
    except:
        raise("Please change current working directory to HW3")

    idx_list, preds = predict(model, test_ds)

    pred_df = pd.DataFrame(data={'idx': idx_list, 'prediction': preds})
    pred_df.to_csv('prediction.csv', index=False)

    


if __name__ == "__main__":
    main()