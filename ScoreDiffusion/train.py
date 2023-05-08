import os
import warnings
import scipy.io as sio
from absl import app, flags
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import gridspec
import functools
import numpy as np

import torch
# from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from diffusion import loss_fn, marginal_prob_std, diffusion_coeff, EMA, euler_sampler, pc_sampler, ode_sampler
# from model import ScoreNet
from model import UNet
from Dataset.dataset import Train_Data, Valid_Data


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', True, help='train from scratch')
flags.DEFINE_bool('continue_train', False, help='train from scratch')

# UNet
flags.DEFINE_integer('ch', 64, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 4, 4], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0., help='dropout rate of resblock')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')

# Training
flags.DEFINE_float('lr', 1e-4, help='target learning rate')
flags.DEFINE_integer('img_size', 512, help='image size')
flags.DEFINE_integer('batch_size', 4, help='batch size')
flags.DEFINE_integer('num_workers', 1, help='workers of Dataloader')
# Logging & Sampling
flags.DEFINE_string('DIREC', 'score-unet', help='name of your project')
flags.DEFINE_integer('sample_size', 2, "sampling size of images")
# Evaluation
flags.DEFINE_integer('max_epoch', 5000, help='frequency of saving checkpoints, 0 to disable during training')

device = torch.device('cuda:0')


def train():
    sigma = 25.
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma) # construc function without parameters
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma) # construc function without parameters
    
    # dataset
    tr_train = Train_Data()
    trainloader = DataLoader(tr_train, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers, 
                             pin_memory=True, shuffle=True)
    va_train = Valid_Data()
    validloader = DataLoader(va_train, batch_size=FLAGS.sample_size, num_workers=FLAGS.num_workers, 
                             pin_memory=True, shuffle=False)

    # model setup
    score_model = UNet(marginal_prob_std=marginal_prob_std_fn,
                       T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
                       num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    
    ema_model = EMA(score_model).to(device)

    optim = torch.optim.Adam(score_model.parameters(), lr=FLAGS.lr)
    
    # sampler setup
    sampler_od = ode_sampler
    sampler_eu = euler_sampler
    sampler_pc = pc_sampler
    
    # show model size
    model_size = 0
    for param in score_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    if FLAGS.continue_train:
        checkpoint = torch.load('./Save/' + FLAGS.DIREC + '/model_latest.pkl')
        score_model.load_state_dict(checkpoint['score_model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optim.load_state_dict(checkpoint['optim'])
        restore_epoch = checkpoint['epoch']
        print('Finish loading model')
    else:
        restore_epoch = 0
               
    if not os.path.exists('Loss'):
        os.makedirs('Loss')

    tr_ls = []
    if FLAGS.continue_train:        
        readmat = sio.loadmat('./Loss/' + FLAGS.DIREC)
        load_tr_ls = readmat['loss']
        for i in range(restore_epoch):
            tr_ls.append(load_tr_ls[0][i])
        print('Finish loading loss!')
        
    for epoch in range(restore_epoch, FLAGS.max_epoch):
        with tqdm(trainloader, unit="batch") as tepoch:
            tmp_tr_loss = 0
            tr_sample = 0
            score_model.train()        
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")            

                # train
                condition = data.to(device)
                x_0 = target.to(device)
                
                loss = loss_fn(score_model, condition, x_0, marginal_prob_std_fn)
                
                tmp_tr_loss += loss.item()
                tr_sample += len(data)
                
                optim.zero_grad()
                loss.backward()                
                optim.step()
                ema_model.update(score_model)
                
                tepoch.set_postfix({'Loss': loss.item()})
                            
        tr_ls.append(tmp_tr_loss / tr_sample)   
        sio.savemat('./Loss/' + FLAGS.DIREC +'.mat', {'loss': tr_ls})

        if not os.path.exists('Train_Output/' + FLAGS.DIREC):
            os.makedirs('Train_Output/' + FLAGS.DIREC)
            
            
        score_model.eval()
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(validloader):
                    if batch_idx == 4:
                        condition = data.to(device)  
                        
                        samples1 = sampler_od(score_model, condition, marginal_prob_std_fn, diffusion_coeff_fn, FLAGS.sample_size)
                        samples2 = sampler_eu(score_model, condition, marginal_prob_std_fn, diffusion_coeff_fn, FLAGS.sample_size)
                        samples3 = sampler_pc(score_model, condition, marginal_prob_std_fn, diffusion_coeff_fn, FLAGS.sample_size)
                        
                        # sample visulization
                        samples1 = samples1.clamp(0., 1.)
                        samples2 = samples2.clamp(0., 1.)
                        samples3 = samples3.clamp(0., 1.)               

                        fig = plt.figure() 
                        fig.set_figheight(8) 
                        fig.set_figwidth(20)
                        spec = gridspec.GridSpec(ncols=5, nrows=2,
                                              width_ratios=[1,1,1,1,1], wspace=0.01,
                                              hspace=0.01, height_ratios=[1,1],left=0,right=1,top=1,bottom=0)
                        ax = fig.add_subplot(spec[0])
                        ax.imshow(data[0].data.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
                        ax = fig.add_subplot(spec[1])
                        ax.imshow(samples1[0].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
                        ax = fig.add_subplot(spec[2])
                        ax.imshow(samples2[0].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
                        ax = fig.add_subplot(spec[3])
                        ax.imshow(samples3[0].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
                        ax = fig.add_subplot(spec[4])
                        ax.imshow(target[1].data.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
						
                        ax = fig.add_subplot(spec[5])
                        ax.imshow(data[1].data.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
                        ax = fig.add_subplot(spec[6])
                        ax.imshow(samples1[1].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
                        ax = fig.add_subplot(spec[7])
                        ax.imshow(samples2[1].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
                        ax = fig.add_subplot(spec[8])
                        ax.imshow(samples3[0].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
                        ax = fig.add_subplot(spec[9])
                        ax.imshow(target[1].data.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax.axis('off')
    
                        plt.savefig('./Train_Output/'+ FLAGS.DIREC + '/Epoch_' + str(epoch+1) + '.png', 
                                    bbox_inches='tight', pad_inches=0)

        # save
        if not os.path.exists('Save/' + FLAGS.DIREC):
            os.makedirs('Save/' + FLAGS.DIREC)
        ckpt = {
            'score_model': score_model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optim': optim.state_dict(),
            'epoch': epoch+1,
        }
        if (epoch+1) % 20 == 0:
            torch.save(ckpt, './Save/' + FLAGS.DIREC + '/model_epoch_'+str(epoch+1)+'.pkl')
        torch.save(ckpt, './Save/' + FLAGS.DIREC + '/model_latest.pkl')



def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore')
    if FLAGS.train:
        train()


if __name__ == '__main__':
    app.run(main)
