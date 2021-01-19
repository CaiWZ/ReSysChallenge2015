import torch
from  sklearn.metrics import roc_auc_score
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from recsys_utils import parse_args,save_checkpoint,AverageMeter,clip_gradient,get_logger
from config import device,grad_clip,print_freq,batch_size
from models import Net
from Mydata import YoochooseBinaryDataset

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint =  args.checkpoint
    start_epcoch = 0
    best_acc = 0
    writer = SummaryWriter()
    epoch_since_improvement = 0

    if checkpoint is None:
        model = Net()
        # model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epcoch =checkpoint['epoch'] + 1
        epoch_since_improvement = checkpoint['epoch_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()
    
    model = model.to(device)

    criterion = nn.BCELoss()

    logger.info('init dataset...')
    dataset = YoochooseBinaryDataset(root='data/')
    dataset = dataset.shuffle()
    train_dataset = dataset[:800000]
    val_dataset = dataset[800000:900000]
    test_dataset = dataset[900000:]

    train_loader = DataLoader(train_dataset,batch_size=batch_size)
    val_loader = DataLoader(val_dataset,batch_size=batch_size)
    test_loader = DataLoader(test_dataset,batch_size=batch_size)

    for epoch in range(start_epcoch,args.end_epoch):
        train_loss = train(train_loader = train_loader,
                           model = model,
                           criterion = criterion,
                           optimizer = optimizer,
                           epoch = epoch,
                           logger =logger
        )
        writer.add_scalar('model/train_loss',train_loss,epoch)

        train_acc = evalute(train_loader,model)
        val_acc = evalute(val_loader,model)
        test_acc =  evalute(test_loader,model)
        # print('epoch:{:03d},loss:{:.5f},train acc:{:.5f},val acc:{:.5f},test acc:{:.5f}'.format(epoch,train_acc,val_acc,test_acc))
        print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
              format(epoch, train_loss, train_acc, val_acc, test_acc))

        writer.add_scalar('model/train_acc',train_acc,epoch)
        writer.add_scalar('model/val_acc',val_acc,epoch)
        writer.add_scalar('model/test_acc',test_acc,epoch)

        is_best = val_acc > best_acc
        best_acc = max(val_acc,best_acc)
        if not is_best:
            epoch_since_improvement += 1
            print('\n epcoch since last improvement:%d\n'%(epoch_since_improvement))
        else:
            epoch_since_improvement = 0
        
        save_checkpoint(epoch,epoch_since_improvement,model,optimizer,best_acc,is_best)

def train(train_loader,model,criterion,optimizer,epoch,logger):
    model.train()

    losses = AverageMeter()
  

    for i, data in enumerate(train_loader):
        data = data.to(device)
        label = data.y.to(device)

        y_out = model(data)

        loss = criterion(y_out,label)

        optimizer.zero_grad()
        loss.backward()

        clip_gradient(optimizer,grad_clip)

        optimizer.step()

        losses.update(loss.item())

        if i%print_freq == 0:
            status = ('epcoh:[{0}][{1}/{2}]\t'\
                  'loss:{loss.val:.5f}({loss.avg:.5f})\t'.format(epoch,i,len(train_loader),loss=losses)
                )
            
            logger.info(status)
    return losses.avg

def evalute(loader,model):
    model.eval()

    prediction = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            prediction.append(pred)
            labels.append(label)
    prediction =  np.hstack(prediction)
    labels = np.hstack(labels)

    return roc_auc_score(labels,prediction)

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__=='__main__':
    main()

    



