'''
16-fold Cross Validation
train 16 models
return the average kappa
'''

import argparse
import os
import torch
from torch import optim
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from dataset import dataset
from sklearn.metrics import cohen_kappa_score,accuracy_score
from timm.loss import SoftTargetCrossEntropy


if __name__ == '__main__':
    
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50d', help='model')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--batch-size', default=32, type=int, help='batch-size')
    parser.add_argument('--lr', default=1e-3, type=float, help='lr')
    parser.add_argument('--epochs', default=30, type=int, help='epochs')
    parser.add_argument('--eval-cycle', default=2, type=int, help='eval-cycle')
    parser.add_argument('--save-dir', default='checkpoints', type=str, help='where to save model')
    parser.add_argument('--alpha', default=1.0, type=float, help='weighted loss(1 will cancel mixup)')
    args = parser.parse_args()

    kappaSum = 0
    stateList = []

    for k in range(16):
        print(f'kfold: {k}')
        # backbone network
        if args.model == 'resnet50d':
            net = timm.create_model('resnet50d', pretrained=True, num_classes=3).to(args.gpu)
            
        # dataset
        trainset = dataset(train=True,kfold=k)
        valset = dataset(val=True,kfold=k)
        trainloader = DataLoader(trainset,  shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)
        valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

        # optimizer & criterion
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, amsgrad=True)
        criterion = nn.CrossEntropyLoss()

        # evaluation: find best model
        bestModel = {
            'state': None,
            'kappa': -1,
            'epoch': 0,
        }

        for epoch in range(args.epochs):
            # train
            net.train()
            totalLoss = 0
            predList = []
            gtList = []
            for img, label, name in trainloader:
                img = img.to(args.gpu)
                label = label.to(args.gpu) # bs
                label_pred = net(img) # bs*3
                prediction = torch.max(label_pred, 1)[1] # bs
                loss = args.alpha * criterion(label_pred, label)
                totalLoss += loss.item()
                predList.extend(prediction.detach().cpu())
                gtList.extend(label.cpu())
    
    
                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            kappa = cohen_kappa_score(gtList, predList, weights='quadratic')
            acc = accuracy_score(gtList, predList)
           
            print(f'Train Epoch:{epoch}, Loss:{totalLoss}, Acc: {acc}, Kappa: {kappa}')

            # validation
            if (epoch+1) % args.eval_cycle == 0:
                with torch.no_grad():
                    net.eval()
                    predList = []
                    gtList = []

                    for img, label, name in valloader:
                        img = img.to(args.gpu)
                        label_pred = net(img)
                        predList.extend(label_pred.max(1)[1].cpu())
                        gtList.extend(label)

                    kappa = cohen_kappa_score(gtList, predList, weights='quadratic')
                    acc = accuracy_score(gtList, predList)
                    print(f'Val Epoch: {epoch}, Acc: {acc}, Kappa: {kappa}')

                    # update best model
                    if kappa > bestModel['kappa']:
                        bestModel['epoch'] = epoch
                        bestModel['kappa'] = kappa
                        bestModel['state'] = net.state_dict()
        
        stateList.append(bestModel['state'])
        kappaSum += bestModel["kappa"]
        # save best model
        savePath = os.path.join(args.save_dir, args.model, f'kfold_{k}.pkl')
        print(f'Saving model(epoch={bestModel["epoch"]},kappa={bestModel["kappa"]}) to {savePath}...')
        torch.save(bestModel, savePath)
        print("*" * 100)

    print(f'Average kappa is {kappaSum/16}')
