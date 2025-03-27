import os
import yaml
import random
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader

from models import FNO1d

from train_utils.losses import LpLoss, DI_loss 
from train_utils.datasets import DI, DIIC, sample_data
from train_utils.utils import save_ckpt, count_params, dict2str

try:
    import wandb
except ImportError:
    wandb = None


def get_molifier(mesh, device):
    mollifier = 0.001 * torch.sin(np.pi * mesh[...])
    return mollifier.to(device)


@torch.no_grad()
def eval_darcy(model, val_loader, criterion, device='cpu'):
    mollifier = get_molifier(val_loader.dataset.mesh, device)
    model.eval()
    val_err = []
    for a in val_loader:
        a = a.to(device)
        out = model(a).squeeze(dim=-1)
        out = out * mollifier
        val_loss = criterion(out)
        val_err.append(val_loss.item())
    N = len(val_loader)
    avg_err = np.mean(val_err)
    std_err = np.std(val_err, ddof=1) / np.sqrt(N)
    return avg_err, std_err

def train(model, 
          ic_loader,   # 초기 조건 데이터셋만 사용
          val_loader,  # 검증 데이터셋
          optimizer, 
          scheduler,
          device, config, args):
    save_step = config['train']['save_step']
    eval_step = config['train']['eval_step']
    f_weight = config['train']['f_loss']

    base_dir = os.path.join('exp', config['log']['logdir'])
    ckpt_dir = os.path.join(base_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # loss function
    lploss = LpLoss(size_average=True)
    ic_mol = get_molifier(ic_loader.dataset.mesh, device)

    if wandb and args.log:
        run = wandb.init(project=config['log']['project'], 
                         entity=config['log']['entity'], 
                         group=config['log']['group'], 
                         config=config, reinit=True, 
                         settings=wandb.Settings(start_method='fork'))
    
    pbar = range(config['train']['num_iter'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)

    ic_iter = sample_data(ic_loader)
    
    for e in pbar:
        log_dict = {}
        optimizer.zero_grad()
        
        # PDE loss 계산만 수행 (unpacking 없이 단일 변수로 받음)
        ic = next(ic_iter)
        ic = ic.to(device)
        out = model(ic).squeeze(dim=-1)
        out = out * ic_mol
        u0 = ic[...]
        f_loss = DI_loss(out, u0)
        log_dict['PDE'] = f_loss.item()

        loss = f_loss * f_weight
        loss.backward()
        optimizer.step()
        scheduler.step()

        log_dict['train loss'] = loss.item()
        if e % eval_step == 0:
            eval_err, std_err = eval_darcy(model, val_loader, DI_loss, device)
            log_dict['val error'] = eval_err
        pbar.set_description(dict2str(log_dict))
        if wandb and args.log:
            wandb.log(log_dict)
        if e % save_step == 0 and e > 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            save_ckpt(ckpt_path, model, optimizer, scheduler)

    if wandb and args.log:
        run.finish()


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 난수 시드 설정
    config['seed'] = args.seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 모델 생성
    model = FNO1d(modes=config['model']['modes'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'], 
                  act=config['model']['act'], 
                  out_dim=config['model']['out_dim']).to(device)
    num_params = count_params(model)
    config['num_params'] = num_params
    print(f'Number of parameters: {num_params}')
    
    # checkpoint 불러오기 (있는 경우)
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    
    if args.test:
        batchsize = config['test']['batchsize']
        testset = DI(datapath=config['test']['path'], # data
                            nx=config['test']['nx'],  # nx가 increment
                            sub=config['test']['sub'], # sub이무지
                            offset=config['test']['offset'],  # 일단 0?
                            num=config['test']['n_sample']) 
        testloader = DataLoader(testset, batch_size=batchsize, num_workers=4)
        criterion = DI_loss()
        test_err, std_err = eval_darcy(model, testloader, criterion, device)
        print(f'Averaged test relative L2 error: {test_err}; Standard error: {std_err}')
    else:
        # 학습 데이터셋 구성 (PDE loss에 사용할 초기조건 데이터셋)
        batchsize = config['train']['batchsize']
        ic_set = DIIC(datapath=config['data']['path'], 
                         nx=config['data']['nx'], 
                         sub=config['data']['pde_sub'], 
                         offset=config['data']['offset'], 
                         num=config['data']['n_sample'])
        ic_loader = DataLoader(ic_set, batch_size=batchsize, num_workers=4, shuffle=True)
        
        # 검증 데이터셋 구성
        valset = DI(datapath=config['test']['path'], 
                           nx=config['test']['nx'], 
                           sub=config['test']['sub'], 
                           offset=config['test']['offset'], 
                           num=config['test']['n_sample'])
        val_loader = DataLoader(valset, batch_size=batchsize, num_workers=4)
        print(f'Test set: {len(valset)} samples.')

        optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=config['train']['milestones'], 
                                                         gamma=config['train']['scheduler_gamma'])
        if args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)
            optimizer.load_state_dict(ckpt['optim'])
            scheduler.load_state_dict(ckpt['scheduler'])
        train(model, ic_loader, val_loader, optimizer, scheduler, device, config, args)
              
    print('Done!')
        

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description='Train using only PINN loss (f_loss)')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on wandb logging')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test', action='store_true', help='Test mode')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    subprocess(args)
