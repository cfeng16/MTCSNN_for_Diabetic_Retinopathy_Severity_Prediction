import numpy as np
from tqdm import tqdm 
import torch
import torch.nn.functional as F
from sklearn import metrics
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import medmnist
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import confusion_matrix, roc_auc_score
from resnet18 import resnet18
from resnet50 import resnet50
from resnet34 import resnet34
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import argparse
import matplotlib.pyplot as plt 
import torch.utils.data as data
from medmnist import INFO, Evaluator
parser = argparse.ArgumentParser()
parser.add_argument('--train_directory', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\train_upscale_2.h5')
parser.add_argument('--test_directory', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\test_upscale_2.pkl')
parser.add_argument('--scale_factor', type=int, default=2)
parser.add_argument('--mapping_layer', type=int, default=4)
parser.add_argument('--lr_dimension', type=int, default=56)
parser.add_argument('--hr_dimension', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=36)
parser.add_argument('--val_batch_size', type=int, default=60)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--deconv_lr', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--output_dir', type=str, default=r'C:\Users\33602\Desktop\598_mini_project')
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()
data_flag = 'retinamnist'
download = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    model = resnet34(num_classes=5, grayscale=False)
    model.to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    info = INFO[data_flag]
    task = info['task']
    n_channels  = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
    ])
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform = data_transform, download=download)
    pil_dataset = DataClass(split='train', download=download)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    best_acc = 0
    best_auc = 0
    train_loss_history = []
    val_loss_history = []
    for i in range(args.num_epochs):
        train_loss = AverageMeter()
        val_loss = AverageMeter()
        model.train()
        for inputs, targets in tqdm(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred1, pred2 = model(inputs)
            targets = targets.long()
            loss1 = criterion1(pred1, targets.squeeze(1))
            pred3 = torch.abs(pred2[:args.batch_size//2] - pred2[args.batch_size//2:])
            #pred3 = pred2[:args.batch_size//2] - pred2[args.batch_size//2:]
            targets = targets.float()
            target2 = torch.abs(targets[:args.batch_size//2] - targets[args.batch_size//2:])
            #target2 = targets[:args.batch_size//2] - targets[args.batch_size//2:]
            loss2 = criterion2(pred3, target2)
            loss3 = criterion2(pred2, targets)
            loss = loss1  + 0.5*loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), n=1)
        train_loss_history.append(train_loss.avg)
        lr_scheduler(optimizer, i)
        model.eval()
        acc = 0
        auc = 0
        prob_list = []
        target_list = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                pred1, pred2 = model(inputs)
                targets = targets.long()
                loss1 = criterion1(pred1, targets.squeeze(1))
                pred3 = torch.abs(pred2[:args.val_batch_size//2] - pred2[args.val_batch_size//2:])
                #pred3 = pred2[:args.batch_size//2] - pred2[args.batch_size//2:]
                targets = targets.float()
                target2 = torch.abs(targets[:args.val_batch_size//2] - targets[args.val_batch_size//2:])
                #target2 = targets[:args.batch_size//2] - targets[args.batch_size//2:]
                loss2 = criterion2(pred3, target2)
                #loss3 = criterion2(pred2, targets)
                loss = loss1 + 0.5*loss2
                val_loss.update(loss.item(), n=1)
            val_loss_history.append(val_loss.avg)
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets = targets.squeeze(1)
                pred1, _ = model(inputs)
                prob = F.softmax(pred1, dim=1)
                pred_label = torch.max(pred1, dim=1).indices
                prob_list.append(pred_label)
                target_list.append(targets)
                acc += torch.sum(pred_label == targets)
            #prob = torch.cat(prob_list, dim=0)
            #targets = torch.cat(target_list, dim=0)
            #auc = roc_auc_score(targets.cpu().numpy(), prob.cpu().numpy(), multi_class='ovr')
            prob_list = torch.cat(prob_list, dim=0).cpu().detach().numpy()
            target_list = torch.cat(target_list, axis=0).cpu().detach().numpy()
            acc = acc/ len(test_dataset)
            if acc > best_acc:
                best_acc = acc
                confusion = metrics.confusion_matrix(target_list, prob_list, labels=[0, 1, 2, 3, 4])
                disp = ConfusionMatrixDisplay(confusion_matrix=confusion)
                #torch.save(model.state_dict(), r'C:\Users\33602\Desktop\eecs545_project\resnet18_sia_best.pth')
        print('best acc is {:.4f}'.format(best_acc))
    # plt.plot(train_loss_history, marker='.', label='Train')
    # plt.plot(val_loss_history, marker='.', label='Val')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss history')
    # x_major_locator=MultipleLocator(2)
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.show()
    disp.plot(cmap='Blues')
    plt.show()
def lr_scheduler(optimizer, epoch):
    if epoch == 49 or epoch == 74:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.1
    return optimizer

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    main()




