import os
import torch
import torch.optim as optim
import cv2
import numpy as np
import torchvision.transforms.functional as F
from model import *

def weight_load(args):
    checkpoint_1 = []
    checkpoint_2 = []
    if args.resume_1 is None:
            return checkpoint_1, checkpoint_2, args
    if args.resume_1 is not None:
        if os.path.isfile(args.resume_1):
            print("loading checkpoint '{}'".format(args.resume_1))
            if args.gpu is None:
                checkpoint_1 = torch.load(args.resume_1)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint_1 = torch.load(args.resume_1, map_location=loc)
    if args.resume_2 is not None:
        if os.path.isfile(args.resume_2):
            print("loading checkpoint '{}'".format(args.resume_2))
            if args.gpu is None:
                checkpoint_2 = torch.load(args.resume_2)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint_2 = torch.load(args.resume_2, map_location=loc)

        params = checkpoint_2['parser']
        args.start_epoch = checkpoint_2['epoch'] + 1
    else:
        params = checkpoint_1['parser']
        args.start_epoch = checkpoint_1['epoch'] + 1
    del params

    return checkpoint_1, checkpoint_2, args

def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict

def get_model(args):
    
    #model = mymodel(args)
    #model = resnet50(args)
    model_1 = unet(args.num_class_1, 512)
    #model_2 = unet(args.num_class_2, 128)
    model_2 = mymodel(args.num_class_2)
    return model_1, model_2

def get_optimizer(args, model):
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    else:
        raise NotImplementedError('{} is not supported. select one among SGD, Adam, AdamW'.format(args.optim))

    return optimizer

def get_scheduler(args, optimizer):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)    
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False)
    return scheduler

def find_hough_circle(image):
    plot_image = image[0].detach().numpy().copy()
    dst = plot_image.copy()
    gray = cv2.cvtColor(plot_image, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1 = 250, param2 = 10, minRadius = 2, maxRadius = 10)
    #for i in circles[0]:
        #cv2.circle(dst, (i[0].item(),i[1].item()), i[2].item(), (255,255,0), 5)
    #    cv2.circle(dst, (int(i[0]),int(i[1])), int(i[2]), (255,255,0), 2)
    #cv2.imshow("hough", dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if circles is None:
        return None

    return circles[0]

def get_patch_image(args, xy_position, target_position, image):
    """
    xy_position[batch_][0] : x1
    xy_position[batch_][1] : y1
    xy_position[batch_][2] : x2
    xy_position[batch_][3] : y2
    """
    #import cv2
    #cv2.imshow('original_image', image[0].detach().numpy())
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    x1=[[]]
    y1=[[]]
    x2=[[]]
    y2=[[]]
    x=[[]]
    y=[[]]
    x_=[]
    dist_threshold = 100
    crop_size = 64
    
    for i in range(image.shape[0]):
        k = 0
        x_ = 0
        y_ = 0
        x1[i] = int(xy_position[i][0] * 100 * 512 / 1920) # resize: 512, original: 1920, label normalize: /100
        y1[i] = int((1920 - (xy_position[i][1] * 100)) * 512 / 1920)
        x2[i] = int(xy_position[i][2] * 100 * 512 / 1920)
        y2[i] = int((1920 - xy_position[i][3] * 100) * 512 / 1920)
        x[i] = int((x1[i]+x2[i]) / 2)
        y[i] = int((y1[i]+y2[i]) / 2)
        if x[i] < crop_size:
            x[i] = crop_size
        if y[i] < crop_size:
            y[i] = crop_size
        if x[i] + crop_size > image.shape[1]:
            x[i] = image.shape[1] - crop_size - 1
        if y[i] + crop_size > image.shape[2]:
            y[i] = image.shape[2] - crop_size - 1
        if target_position is not None:
            for j in target_position:
                dist = np.sqrt((j[0].item() - x[i]) ** 2 + (j[1] - y[i]) ** 2)
                if dist > dist_threshold:
                    continue
                x_ += j[0]
                y_ += j[1]
                k += 1
        if k < 2:
            x_ = x[i]
            y_ = y[i]
        else:
            x_ = int(x_ / k)
            y_ = int(y_ / k)
        if x_ < crop_size:
            x_ = crop_size
        if y_ < crop_size:
            y_ = crop_size
        if x_ + crop_size > image.shape[1]:
            x_ = image.shape[1] - crop_size - 1
        if y_ + crop_size > image.shape[2]:
            y_ = image.shape[2] - crop_size - 1
        if i == 0:
            patch_image = image[i, y_ - crop_size:y_ + crop_size, x_ - crop_size:x_ + crop_size, :]
            patch_image = torch.unsqueeze(patch_image, 0)
        else:
            torch.stack([patch_image, image[i, y_ - crop_size:y_ + crop_size, x_ - crop_size:x_ + crop_size, :]], axis = 0)
    
    ### figure ###
    if args.patch_show:
        plot_image = patch_image[0].detach().numpy().copy()
        cv2.imshow('patch_image', plot_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        #print(k)
    patch_image = patch_image / 255.0
    patch_image = np.transpose(patch_image, (0, 3, 1, 2))
    patch_image = F.resize(patch_image, crop_size * 2)
    #patch_image = torch.from_numpy(patch_image)


    return patch_image


def get_circle_image(args, xy_position, target_position, image, label):
    """
    xy_position[batch_][0] : x1
    xy_position[batch_][1] : y1
    xy_position[batch_][2] : x2
    xy_position[batch_][3] : y2
    """
    #import cv2
    #cv2.imshow('original_image', image[0].detach().numpy())
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    x1=[[]]
    y1=[[]]
    x2=[[]]
    y2=[[]]
    x=[[]]
    y=[[]]
    dist_threshold = 100
    crop_size_ = 10
    patch_image = None
    for i in range(image.shape[0]):
        x_=[]
        y_=[]
        k = 0
        if args.test:
            x1[i] = int(xy_position[i][0] * 100 * 512 / 1920) # resize: 512, original: 1920, label normalize: /100
            y1[i] = int((1920 - (xy_position[i][1] * 100)) * 512 / 1920)
            x2[i] = int(xy_position[i][2] * 100 * 512 / 1920)
            y2[i] = int((1920 - xy_position[i][3] * 100) * 512 / 1920)
            x[i] = int((x1[i]+x2[i]) / 2)
            y[i] = int((y1[i]+y2[i]) / 2)
        
            if target_position is not None and args.test:
                for j in target_position:
                    dist = np.sqrt((j[0].item() - x[i]) ** 2 + (j[1] - y[i]) ** 2)
                    if dist > dist_threshold:
                        continue
                    #x_ += j[0]
                    #y_ += j[1]
                    x_.append(int(j[0]))
                    y_.append(int(j[1]))
                    k += 1
        if args.test is False:
            x_.append(int(label[i][0]* 100 * 512 / 1920))
            x_.append(int(label[i][2]* 100 * 512 / 1920))
            y_.append(int((1920 - (label[i][1]* 100)) * 512 / 1920))
            y_.append(int((1920 - (label[i][3]* 100)) * 512 / 1920))
            k = 2
        if k == 2:
            if y_[0] >= y_[1]:
                if i == 0:
                    patch_image = image[i, y_[0] - crop_size_:y_[0] + crop_size_, x_[0] - crop_size_:x_[0] + crop_size_, :]
                    patch_image = torch.stack([patch_image, image[i, y_[1] - crop_size_:y_[1] + crop_size_, x_[1] - crop_size_:x_[1] + crop_size_ :]], axis = 0)
                else:
                    patch_image = torch.stack([patch_image, image[i, y_[0] - crop_size_:y_[0] + crop_size_, x_[0] - crop_size_:x_[0] + crop_size_ :]], axis = 0)
                    patch_image = torch.stack([patch_image, image[i, y_[1] - crop_size_:y_[1] + crop_size_, x_[1] - crop_size_:x_[1] + crop_size_ :]], axis = 0)
            else:
                if i == 0:
                    patch_image = image[i, y_[1] - crop_size_:y_[1] + crop_size_, x_[1] - crop_size_:x_[1] + crop_size_, :]
                    patch_image = torch.stack([patch_image, image[i, y_[0] - crop_size_:y_[0] + crop_size_, x_[0] - crop_size_:x_[0] + crop_size_ :]], axis = 0)
                else:
                    patch_image = torch.stack([patch_image, image[i, y_[1] - crop_size_:y_[1] + crop_size_, x_[1] - crop_size_:x_[1] + crop_size_ :]], axis = 0)
                    patch_image = torch.stack([patch_image, image[i, y_[0] - crop_size_:y_[0] + crop_size_, x_[0] - crop_size_:x_[0] + crop_size_ :]], axis = 0)

            if args.patch_show:
                plot_image_0 = patch_image[0].detach().numpy().copy()
                plot_image_1 = patch_image[1].detach().numpy().copy()
                cv2.imshow('patch_image', plot_image_0)
                cv2.waitKey()
                cv2.destroyAllWindows()
                cv2.imshow('patch_image', plot_image_1)
                cv2.waitKey()
                cv2.destroyAllWindows()
            patch_image = patch_image / 255.0
            patch_image = np.transpose(patch_image, (0, 3, 1, 2))
            #patch_image = F.resize(patch_image, crop_size * 2)
        
        if i == image.shape[0] - 1:
            if patch_image is not None:
                return patch_image
            else:
                return None
        