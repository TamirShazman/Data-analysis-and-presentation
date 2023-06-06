from sklearn.model_selection import KFold
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import seaborn as sns
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import shutil

from aug_run_train_eval import train_fold


MAP_LABELS = {
    'i': 0,
    'ii': 1,
    'iii': 2,
    'iv': 3,
    'ix': 4,
    'v': 5,
    'vi': 6,
    'vii': 7,
    'viii': 8,
    'x': 9
}

MAP_NUM_LABELS = {
    0: 'i',
    1: 'ii',
    2: 'iii',
    3: 'iv',
    4: 'ix',
    5: 'v',
    6: 'vi',
    7: 'vii',
    8: 'viii',
    9: 'x'
}

def split_train_val_test(img_paths='/home/student/Data-analysis-and-presentation/HW2/hw2_094295/data', 
                         write_folds_path='/home/student/Data-analysis-and-presentation/HW2/hw2_094295/folds'):
    """
    Split the data into 3-cross validation: 56% train 10%val 33%test.
    Assuming the folders hierarchy are like the data directory
    """

    df_img = pd.DataFrame({'path': [], 'label': []})

    for root, dirs, files in os.walk(img_paths, topdown=True):

        if len(files) == 0:
            continue

        for name in files:
            if not name.endswith('.png'):
                continue

            img_path = os.path.join(root, name)
            label = MAP_LABELS[root.split('/')[-1]]
            df_img.loc[len(df_img.index)] = [img_path, label]

    if not os.path.exists(write_folds_path):
        os.makedirs(write_folds_path)

    kf = KFold(n_splits=3, random_state=None, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(df_img.index)):
        fold_path = os.path.join(write_folds_path, str(i))
        os.mkdir(fold_path)

        train_df = df_img.iloc[train_index]
        test_df = df_img.iloc[test_index]

        val_df = train_df.sample(frac=0.1)
        train_df = train_df.drop(val_df.index)

        train_df.to_csv(fold_path + '/train.csv', index=False)
        val_df.to_csv(fold_path + '/val.csv', index=False)
        test_df.to_csv(fold_path + '/test.csv', index=False)


class MyDataset(Dataset):
    def __init__(self, df, tran):
        self.df = df
        self.tran = tran
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx]['label']
        img = self.load_img(idx)
        return img, label
    
    def load_img(self, idx):
        with open(self.df.iloc[idx]['path'], "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.tran:
                img = self.tran(img)
            return img
    
def test_augm(exp_name, num_folds, trans, folds_path, epoch, lr, batch_size, num_of_classes):
    cur_path = '/home/student/Data-analysis-and-presentation/HW2/hw2_094295/experiment'
    exp_path = os.path.join(cur_path, exp_name)
    os.mkdir(exp_path)

    acc_fold = []
    loss_fold = []

    for i in range(num_folds):
        exp_fold_path = os.path.join(exp_path, str(i))
        os.mkdir(exp_fold_path)
        fold_path = os.path.join(folds_path, str(i))

        train_df = pd.read_csv(fold_path + '/train.csv')
        val_df = pd.read_csv(fold_path + '/val.csv')
        test_df = pd.read_csv(fold_path + '/test.csv')

        val_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
        train_ds = MyDataset(train_df, trans)
        val_ds = MyDataset(val_df, val_transforms)
        test_ds = MyDataset(test_df, val_transforms)

        train_sampler = torch.utils.data.RandomSampler(data_source=train_ds, replacement=True, num_samples=8000)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        test_loss, test_acc = train_fold(train_dl, val_dl, test_dl, epoch, lr, num_of_classes, exp_fold_path)

        acc_fold.append(test_acc.item())
        loss_fold.append(test_loss)

    df_acc = pd.DataFrame({'Acc': acc_fold})
    df_loss = pd.DataFrame({'loss': loss_fold})

    acc_boxplot = sns.boxplot(x=df_acc['Acc'])
    fig = acc_boxplot.get_figure()
    fig.savefig(exp_path + "/Fold Acc.png") 
    fig.clear()
    loss_boxplot = sns.boxplot(x=df_loss['loss'])
    fig = loss_boxplot.get_figure()
    fig.savefig(exp_path + "/Fold Loss.png") 
    fig.clear()

    with open(exp_path + '/AVG_result.txt', 'w') as f:
        f.write(f'AVG loss: {sum(loss_fold) / len(loss_fold)}, AVG Acc: {sum(acc_fold) / len(acc_fold)}')




def run(
    exp_name = 'baseline',
    folds_path = '/home/student/Data-analysis-and-presentation/HW2/hw2_094295/folds',
    num_folds= 3,
    epoch = 15,
    lr = 0.001,
    batch_size = 16,
    num_of_classes = 10,
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
):

    # creating the folds
    # split_train_val_test(img_paths='/home/student/Data-analysis-and-presentation/HW2/hw2_094295/fixed_data1')
    # change the transformation, make sure you add the crop as is and first in the tranformation pipline

    test_augm(exp_name, num_folds, data_transforms, folds_path, epoch, lr, batch_size, num_of_classes)

    
def produce_img(load_path, save_path, num_img_to_produce=8000, tran=transforms.Compose([transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN), transforms.ToTensor()])):
    """
    load all the images under load_path directory, apply the given transformation for a given number on the image
    and save them under save_path according to the accepted format (i.e. a directory for each class).
    """
    num_img_class = {}

    # load all the images
    df_img = pd.DataFrame({'path': [], 'label': []})
    for root, dirs, files in os.walk(load_path, topdown=True):

        if len(files) == 0:
            continue

        for name in files:
            if not name.endswith('.png'):
                continue

            img_path = os.path.join(root, name)
            label = MAP_LABELS[root.split('/')[-1]]
            df_img.loc[len(df_img.index)] = [img_path, label]

    ds = MyDataset(df_img, tran)
    sampler = torch.utils.data.RandomSampler(data_source=ds, replacement=True, num_samples=num_img_to_produce)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, sampler=sampler)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for x, y in dl:
        label_str = MAP_NUM_LABELS[y.item()]

        if label_str not in num_img_class.keys():
            num_img_class[label_str] = 0
        
        image_path = os.path.join(save_path, label_str)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_path = os.path.join(image_path, f'{str(num_img_class[label_str])}.png')
        save_image(x, image_path)
        num_img_class[label_str] += 1

def split_train_val_test(img_paths='/home/student/Data-analysis-and-presentation/HW2/hw2_094295/fixed_data1', 
                         write_path='/home/student/Data-analysis-and-presentation/HW2/hw2_094295/final_before_aug_data',
                         test_size=1000):
    """
    Create stratified train-val
    """
    df_img = pd.DataFrame({'path': [], 'label': []})

    for root, dirs, files in os.walk(img_paths, topdown=True):

        if len(files) == 0:
            continue

        for name in files:
            if not name.endswith('.png'):
                continue
            img_path = os.path.join(root, name)
            label = MAP_LABELS[root.split('/')[-1]]
            df_img.loc[len(df_img.index)] = [img_path, label]
            
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    x_train, x_val, y_train, y_val = train_test_split(df_img['path'].values, df_img['label'].values, test_size=test_size, stratify=df_img['label'].values, random_state=123)

    for x, y in zip(x_train, y_train):
        train_directory = os.path.join(write_path, 'train')
        target_directory = os.path.join(train_directory, MAP_NUM_LABELS[y])
        if not os.path.exists(train_directory):
            os.makedirs(train_directory)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        
        source = x
        # Destination path
        destination = os.path.join(target_directory, x.split('/')[-1])
        try:
            shutil.copy(source, destination)
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        except PermissionError:
            print("Permission denied.")
        except e:
            print(e)

    for x, y in zip(x_val, y_val):
        val_directory = os.path.join(write_path, 'val')
        target_directory = os.path.join(val_directory, MAP_NUM_LABELS[y])
        if not os.path.exists(val_directory):
            os.makedirs(val_directory)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        
        source = x
        # Destination path
        destination = os.path.join(target_directory, x.split('/')[-1])
        try:
            shutil.copy(source, destination)
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        except PermissionError:
            print("Permission denied.")
        except:
            print("Error occurred while copying file.")

    