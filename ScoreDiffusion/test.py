import copy
import warnings
from absl import app, flags
import os
import functools
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
from diffusion import loss_fn, marginal_prob_std, diffusion_coeff, EMA, euler_sampler, pc_sampler, ode_sampler
from model import UNet
from Dataset.dataset import Valid_Data


FLAGS = flags.FLAGS

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
flags.DEFINE_integer('batch_size', 2, help='batch size')
flags.DEFINE_integer('num_workers', 1, help='workers of Dataloader')

# Logging & Sampling
flags.DEFINE_string('DIREC', 'score-unet', help='name of your project')
flags.DEFINE_integer('sample_size', 9, "sampling size of images")

device = torch.device('cuda:0')


def test():
    sigma = 25.
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma) # construc function without parameters
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma) # construc function without parameters

    # dataset
    va_train = Valid_Data()
    validloader = DataLoader(va_train, batch_size=FLAGS.sample_size, num_workers=FLAGS.num_workers, 
                             pin_memory=True, shuffle=False)

    # model setup
    # score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    score_model = UNet(marginal_prob_std=marginal_prob_std_fn,
                       T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
                       num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    
    ema_model = EMA(score_model).to(device)
    
    # sampler setup
    sampler_od = ode_sampler
    sampler_eu = euler_sampler
    sampler_pc = pc_sampler
        
    # show model size
    model_size = 0
    for param in score_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    checkpoint = torch.load('./Save/' + FLAGS.DIREC + '/model_latest.pkl')
    score_model.load_state_dict(checkpoint['score_model'])
    ema_model.load_state_dict(checkpoint['ema_model'])
    restore_epoch = checkpoint['epoch']
    print('Finish loading model')
                        
    output1 = np.zeros((27,512,512))  # example size, please change based on your data
    output2 = np.zeros((27,512,512))
    output3 = np.zeros((27,512,512))
    lr = np.zeros((27,512,512))
    hr = np.zeros((27,512,512))
    if not os.path.exists('Output/' + FLAGS.DIREC):
        os.makedirs('Output/' + FLAGS.DIREC)
    score_model.eval()
    count = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validloader):
            condition = data.to(device)                   
            length = data.shape[0]
            
			samples1 = sampler_od(score_model, condition, marginal_prob_std_fn, diffusion_coeff_fn, FLAGS.sample_size)
			samples2 = sampler_eu(score_model, condition, marginal_prob_std_fn, diffusion_coeff_fn, FLAGS.sample_size)
			samples3 = sampler_pc(score_model, condition, marginal_prob_std_fn, diffusion_coeff_fn, FLAGS.sample_size)
			samples1 = samples1.clamp(0., 1.)
			samples2 = samples2.clamp(0., 1.)
			samples3 = samples3.clamp(0., 1.)

			output1[count:count+length,:,:] = samples1.squeeze().cpu()        
			output2[count:count+length,:,:] = samples2.squeeze().cpu()        
			output3[count:count+length,:,:] = samples3.squeeze().cpu()
			lr[count:count+length,:,:] = data.squeeze().cpu()
			hr[count:count+length,:,:] = target.squeeze().cpu()
            
            count += length
            
    path = 'Output/' + FLAGS.DIREC + '/result_epoch_' + str(restore_epoch) + '.hdf5'
    f = h5py.File(path, 'w')
    f.create_dataset('ode', data=output1)
    f.create_dataset('em', data=output2)
    f.create_dataset('pc', data=output3)
    f.create_dataset('lr', data=lr)
    f.create_dataset('hr', data=hr)
    f.close()



def main(argv):
    warnings.simplefilter(action='ignore')
    test()


if __name__ == '__main__':
    app.run(main)
