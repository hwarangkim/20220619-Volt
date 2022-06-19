import os
import argparse
import torch
from data import ScrewholeDataset
from torch.utils.data import DataLoader
from utils import *
from train import train_1, train_2
from test import *

def argparser(parser):
    #parser.add_argument('--image_path', default='.\\test_images', type=str)
    parser.add_argument('--image_path', default='.\\train_images\\withBG', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_class_1', default=4, type=int)
    parser.add_argument('--num_class_2', default=1, type=int)
    parser.add_argument('--resume_1', default=".\\weights_1\\checkpoint_17.pth", type=str)
    parser.add_argument('--resume_2', default=None, type=str)
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--train_1', default=False, type=bool)
    parser.add_argument('--train_2', default=True, type=bool)
    parser.add_argument('--num_epoch', default=5000, type=int,
                        help='Epoch number for training')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_folder_1', default='.\\weights_1', type=str)
    parser.add_argument('--weight_folder_2', default='.\\weights_2', type=str)
    parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use. If you want to use only CPU, set None.')
    parser.add_argument('--optim', default='AdamW', type=str,
                        help='SGD or Adam or AdamW')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--grad_accumulation_steps', default=1, type=int,
                    help='Number of gradient accumulation steps')
    parser.add_argument('--equation_find', default=False, type=bool)
    parser.add_argument('--xyplot', default=True, type=bool)
    parser.add_argument('--patch_show', default=False, type=bool)
    parser.add_argument('--lineplot', default=False, type=bool)
    args = parser.parse_args()
    return args

def main(args):
    ### Data parsing ###
    print('data loading...')

    dataset = ScrewholeDataset(args.image_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle = True,
                                drop_last = False)
    print('data loaded.')
    
    
    # Load model, optimizer, scheduler, checkpoint
    model_1, model_2 = get_model(args)
    
    checkpoint_1, checkpoint_2, args = weight_load(args)


    if len(checkpoint_1) != 0:
        model_1.load_state_dict(checkpoint_1['state_dict'])
        #print('###### model architecture #######')
        #print(model)
        del checkpoint_1
    if len(checkpoint_2) != 0:
        model_2.load_state_dict(checkpoint_2['state_dict'])
        del checkpoint_2

    if args.gpu is not None:
        print('Use GPU: {}'.format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model_1.cuda(args.gpu)
        model_2.cuda(args.gpu)

    iters_per_epoch = int(len(dataset)/args.batch_size)
    if args.xyplot:
        xy_plot(data_loader,model_1,iters_per_epoch, args)
    if args.lineplot:
        line_plot(data_loader,model_1, model_2, iters_per_epoch, args)
    
    if args.train_1:
        optimizer_1 = get_optimizer(args, model_1)
        scheduler_1 = get_scheduler(args, optimizer_1)
        for epoch in range(args.start_epoch, args.num_epoch):
        
            loss = train_1(data_loader, model_1, scheduler_1, optimizer_1, epoch, args, iters_per_epoch)
            if (epoch + 1) % 5 == 0:
                pass #validation is not developed yet
            state = {
                'epoch': epoch,
                'parser': args,
                'state_dict':get_state_dict(model_1)
                }
            torch.save(
                state,
                os.path.join(
                    args.weight_folder_1,
                    "checkpoint_{}.pth".format(epoch)
                    )
                )
    
    elif args.train_2:
        optimizer_2 = get_optimizer(args, model_2)
        scheduler_2 = get_scheduler(args, optimizer_2)
        for epoch in range(args.start_epoch, args.num_epoch):
            loss = train_2(data_loader, model_1, model_2, scheduler_2, optimizer_2, epoch, args, iters_per_epoch)
            if (epoch + 1) % 3 == 0:
                pass #validation is not developed yet
                state = {
                    'epoch': epoch,
                    'parser': args,
                    'state_dict':get_state_dict(model_2)
                    }
                torch.save(
                    state,
                    os.path.join(
                        args.weight_folder_2,
                        "checkpoint_{}.pth".format(epoch)
                        )
                    )
    else:
        pass #test is not developed yet

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fiding normal vector equations in screw hole images')
    args = argparser(parser) #set argument
    torch.cuda.empty_cache()

    if(not os.path.exists(args.weight_folder_1)):
        os.makedirs(args.weight_folder_1)
    if(not os.path.exists(args.weight_folder_2)):
        os.makedirs(args.weight_folder_2)
    print('start epoch: {}'.format(args.start_epoch))
    print('total epoch: {}'.format(args.num_epoch))
    print('batch size: {}'.format(args.batch_size))
    main(args)