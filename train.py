import torch
import numpy as np
import torch.nn as nn
import cv2
import torchvision.transforms.functional as F
from utils import *

def train_1(data_loader, model, scheduler, optimizer, epoch, args, iters_per_epoch):
    print("{} epoch: \t start training....".format(epoch))
    model.train()
    optimizer.zero_grad()
    data_iter = iter(data_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = torch.nn.MSELoss().to(device)
    total_loss = []
    for iters in range(iters_per_epoch):
        try:
            data, _, _ = next(data_iter)
        except:
            data_iter = iter(data_loader)
            data, _, _ = next(data_iter)

        image, label = data['img'], data['label']
        
        image = image.cuda().float()
        label = label.cuda().float()
        
        output = model(image)
        
        loss = loss_fn(output,label)
        #loss = -(label * torch.log(output) + (1-label) * torch.log(1-output)).mean()
        #loss = torch.sum((output - label)**2)
        #loss = torch.mean(torch.sqrt((output-label)**2))
        loss.backward()
        loss_value = loss.item()
        total_loss.append(loss_value)

        if(iters % 5 == 0):
            print('{} iteration: training...'.format(iters))

            ans = {
                'epoch': epoch,
                'iteration': iters,
                'loss': loss_value,
                'total_loss': np.mean(total_loss)
            }
            for key, value in ans.items():
                print('    {:15s}: {}'.format(str(key), value))
            print('label:',label)
            print('output:', output)
        if (iters + 1) % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()


    return np.mean(total_loss)

def train_2(data_loader, model_1, model_2, scheduler_2, optimizer_2, epoch, args, iters_per_epoch):
    print("{} epoch: \t start training....".format(epoch))
    model_1.train()
    model_2.train()
    optimizer = optimizer_2
    scheduler = scheduler_2
    
    optimizer.zero_grad()
    data_iter = iter(data_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = torch.nn.MSELoss().to(device)
    total_loss = []
    for iters in range(iters_per_epoch):
        try:
            data, org_img, label_2 = next(data_iter)
        except:
            data_iter = iter(data_loader)
            data, org_img, label_2 = next(data_iter)

        image, label = data['img'], data['label']
        
        image = image.cuda().float()
        label = label.cuda().float()
        with torch.no_grad():
            output_ = model_1(image)
        
            #print(output_)
            #print(label)
        target_position = find_hough_circle(org_img)
        #patch_image_ = get_patch_image(args, output_, target_position, org_img)
        patch_image = get_circle_image(args, output_, target_position, org_img, label)
        patch_image = patch_image.cuda().float()

        label_2 = label_2.cuda().float()

        output_1 = model_2(torch.unsqueeze(patch_image[0], 0))
        output_2 = model_2(torch.unsqueeze(patch_image[1], 0))
        loss_1 = loss_fn(output_1, label_2[0][0])
        loss_2 = loss_fn(output_2, label_2[0][1])
        loss = loss_1 + loss_2
        loss.backward()
        loss_value = loss.item()
        total_loss.append(loss_value)

        if(iters % 5 == 0):
            print('{} iteration: training...'.format(iters))

            ans = {
                'epoch': epoch,
                'iteration': iters,
                'loss': loss_value,
                'total_loss': np.mean(total_loss)
            }
            for key, value in ans.items():
                print('    {:15s}: {}'.format(str(key), value))
            print('label:',label_2)
            print('output:', [round(output_1.item(), 4), round(output_2.item(), 4)])
        if (iters + 1) % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model_2.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()


    return np.mean(total_loss)
