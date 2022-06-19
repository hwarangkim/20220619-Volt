import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from utils import *
import math

def xy_plot(data_loader, model, iters_per_epoch, args):
    print('x,y position plotting')
    data_iter = iter(data_loader)
    model.eval()
    font = cv2.FONT_HERSHEY_SIMPLEX
    with torch.no_grad():
        for iters in range(iters_per_epoch):
            try:
                data, _, label_2 = next(data_iter)
            except:
                data_iter = iter(data_loader)
                data, _, label_2 = next(data_iter)
            image, label = data['img'], data['label']
            image = image.cuda().float()

            output = model(image)
            
            for i in range(args.batch_size):
                #print((label[i]-output[i].detach().cpu())*512)
                plot_img = image[i] * 255.0
                plot_img = plot_img.detach().cpu().numpy()
                plot_img = np.transpose(plot_img, (1,2,0)).astype(np.uint8)
                plot_img = plot_img.copy()
                #print('output:', output*100)
                #print('label:', label*100)
                #print('output/512:', output)
                print('label:', label_2)
                #output[i][0] = label[i][0]
                #output[i][2] = label[i][2]
                resize_ratio = 100*512.0/1920.0
                """
                #cv2.circle(plot_img, (int(output[i][0]*resize_ratio), 512-int(output[i][1]*resize_ratio)), 3, (0,0,255), -1)
                #cv2.circle(plot_img, (int(output[i][2]*resize_ratio), 512-int(output[i][3]*resize_ratio)), 3, (255,0,0), -1)
                #cv2.putText(plot_img, "x: {}, y: {}".format(int((output[i][0]*resize_ratio)), (512-int(output[i][1]*resize_ratio))) , (10,500), font, 1, (255,0,0), 2)
                #cv2.putText(plot_img, "x: {}, y: {}".format(int((output[i][2]*resize_ratio)),(512-int(output[i][3]*resize_ratio))) , (10,450), font, 1, (0,0,255), 2)
                """
                #cv2.putText(plot_img, "x: {}, y: {}".format(str(int(label[i][0].item()*resize_ratio)), str(int(label[i][1].item()*resize_ratio))) , (10,400), font, 1, (0,255,0), 2)
                #cv2.putText(plot_img, "x: {}, y: {}".format(str(int(label[i][2].item()*resize_ratio)), str(int(label[i][3].item()*resize_ratio))) , (10,300), font, 1, (255,255,0), 2)

                cv2.putText(plot_img, "a1: {}, a2: {}".format(label_2[i][0], label_2[i][1]) , (10,450), font, 1, (0,0,255), 2)
                cv2.imshow('xy_plot', plot_img)
                cv2.waitKey()
                cv2.destroyAllWindows()

def line_plot(data_loader, model_1, model_2, iters_per_epoch, args):
    print('first-ordered function plotting')
    data_iter = iter(data_loader)
    model_1.eval()
    model_2.eval()
    
    with torch.no_grad():
        for iters in range(iters_per_epoch):
            try:
                data, org_img, label_2 = next(data_iter)
            except:
                data_iter = iter(data_loader)
                data, org_img, label_2 = next(data_iter)
            image, label = data['img'], data['label']
            image = image.cuda().float()
            
            output_ = model_1(image)
        
            #print(output_)
            #print(label)
            target_position = find_hough_circle(org_img)
            patch_image = get_circle_image(args, output_, target_position, org_img, label)
            if patch_image is None:
                continue
            patch_image = patch_image.cuda().float()

            label_2 = label_2.cuda().float()

            output_1 = model_2(torch.unsqueeze(patch_image[0], 0))
            output_2 = model_2(torch.unsqueeze(patch_image[1], 0))
            #patch_image = get_patch_image(args, output_, target_position, org_img)
            
            label_2 = label_2.cuda().float()
            #output = model_2(patch_image)

            
            x = np.array(range(0, image.shape[2]))
            for i in range(args.batch_size):
                #print((label[i]-output[i].detach().cpu())*512)
                #plot_img = image[i] * 255.0
                #plot_img = plot_img.detach().cpu().numpy()
                #plot_img = np.transpose(plot_img, (1,2,0)).astype(np.uint8)
                #plot_img = plot_img.copy()
                
                
                #label_2[i][0] = math.tan(label_2[i][0])
                #label_2[i][1] = math.tan(label_2[i][1])
                print('output:', [output_1[i].item(), output_2[i].item()])
                print('label:', label_2)
                #print('output/512:', output)
                #print('label/512:', label)
                #output = output.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                label_2 = label_2.detach().cpu().numpy()
                a1 = math.tan(output_1[i])
                b1 = label[i][1] - (a1 * label[i][0])
                a2 = math.tan(output_2[i])
                b2 = label[i][3] - (a2 * label[i][2])
                
                plt.grid(color = "gray", alpha = .5, linestyle = '--')
                plt.plot(x, a1 * x + b1, label = 'prediction 1')
                plt.plot(x, label_2[i][0] * x + label[i][1] - (label_2[i][0] * label[i][0]), label = 'label 1')
                plt.plot(x, a2 * x + b2, label = 'prediction 2')
                plt.plot(x, label_2[i][1] * x + label[i][3] - (label_2[i][1] * label[i][2]), label = 'label 2')
                
                plt.legend()
                plt.show()
                