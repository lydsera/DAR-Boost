import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead")
import argparse
import sys
import os
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2019_devNeval,Dataset_ASVspoof2021_eval
from model import Model
import importlib
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from evaluation import calculate_tDCF_EER


def evaluate_accuracy(dev_loader, model, device, autoboost=None):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    if autoboost:
        autoboost.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        if autoboost:
            with torch.no_grad():
                batch_x, _ = autoboost(batch_x)
        
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss

def produce_evaluation_file_19(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        batch_out = model(batch_x)
        
        batch_score = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel() 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr, optim, device, autoboost=None, optimizer_boost=None, lambda_div=0.5):
    running_loss = 0
    num_total = 0.0
    
    
    model.train()
    if autoboost:
        autoboost.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        if autoboost:
            # --- Single-Pass Adversarial Training (Decoupled) ---
            
            # 1. Forward (AutoBoost + Task Model)
            # GRL is inside AutoBoost, so we just forward through both.
            x_aug, params = autoboost(batch_x)
            
            batch_out = model(x_aug)
            
            batch_loss = criterion(batch_out, batch_y)
            
            running_loss += (batch_loss.item() * batch_size)
            
            # --- Diversity Loss (多样性损失) ---
            div_loss = 0

            total_loss = batch_loss - lambda_div * div_loss
            
            # 2. Backward (Computes gradients for both models)
            # Task Model: gradients are normal (minimize loss)
            # AutoBoost: gradients are reversed by GRL (maximize loss)
            total_loss.backward()
            
            optim.step()                    # Update Task Model
            optim.zero_grad()               
            
            if optimizer_boost is not None:
                optimizer_boost.step()          # Update AutoBoost
                optimizer_boost.zero_grad()
                
        else:
            # Standard Training
            batch_out = model(batch_x)
            
            batch_loss = criterion(batch_out, batch_y)
            
            running_loss += (batch_loss.item() * batch_size)
        
            optim.zero_grad()
            batch_loss.backward()
            optim.step()
    running_loss /= num_total

    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/home/abcd/data/ASVspoof2019/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac

 
 
    '''

    parser.add_argument('--protocols_path', type=str, default='/home/abcd/data/ASVspoof2019/LA/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--models_save_dir', type=str, default='models', help='Directory to save models during training(default: models)')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--year',type=int, default=2019, help='2019 or 2021 for evaluation')
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 

    # AutoBoost
    parser.add_argument('--use_autoboost', action='store_true', default=False,
                        help='Enable AutoBoost adversarial data augmentation') 
    parser.add_argument('--lambda_div', type=float, default=0.5, 
                        help='Weight for diversity loss in AutoBoost')


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    args = parser.parse_args()

    # Dynamically import AutoBoost
    if args.use_autoboost:
        try:
            module_name = 'boost'
            AutoBoost_module = importlib.import_module(module_name)
            AutoBoost = AutoBoost_module.AutoBoost
            print(f"Successfully imported AutoBoost from {module_name}")
        except ImportError as e:
            print(f"Error importing AutoBoost version {args.autoboost_version}: {e}")
            sys.exit(1)

    if not os.path.exists(args.models_save_dir):
        os.mkdir(args.models_save_dir)
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    assert track in ['LA', 'PA','DF'], 'Invalid track given'

    #database
    prefix      = 'ASVspoof2019_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    # AutoBoost Initialization
    autoboost = None
    optimizer_boost = None
    lr_boost = 0.0
    if args.use_autoboost:
        print("Initializing AutoBoost...")
        autoboost = AutoBoost().to(device)
        n_boost_params = sum([param.view(-1).size()[0] for param in autoboost.parameters()])
        print('AutoBoost nb_params:', n_boost_params)
        
        # Adjust AutoBoost LR based on parameter ratio
        if n_boost_params > 0:
            lr_boost = args.lr * (nb_params / n_boost_params)
            print(f"AutoBoost LR adjusted to {lr_boost:.2e} (Model/Boost Ratio: {nb_params/n_boost_params:.2f})")
        else:
            lr_boost = args.lr
            
        optimizer_boost = torch.optim.Adam(autoboost.parameters(), lr=lr_boost)

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.use_autoboost:
        model_tag = model_tag + '_autoboost_lr{:.2e}'.format(lr_boost) # format scientific notation
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join(args.models_save_dir, model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))


    #evaluation 
    if args.eval:
        if args.year==2019:
            from pathlib import Path
            database_path = Path(args.database_path)
            eval_trial_path = (
                database_path /
                "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
                    track, prefix_2019))
            eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)
            _,file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=True,
                              is_eval=False)
            print('no. of eval trials',len(file_eval))
            eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
            eval_loader = DataLoader(eval_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)
            produce_evaluation_file_19(eval_loader, model, device,
                                "score.txt", eval_trial_path)
            eval_eer, eval_tdcf = calculate_tDCF_EER(
                cm_scores_file="score.txt",
                asv_score_file=database_path / "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
                output_file="results.txt")
            print("EER: {:.3f}, min t-DCF: {:.5f}".format(
                eval_eer, eval_tdcf))
            sys.exit(0)
        elif args.year==2021:
            file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix,prefix_2021)),is_train=False,is_eval=True)
            print('no. of eval trials',len(file_eval))
            eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))
            produce_evaluation_file(eval_set, model, device, args.eval_output)
            sys.exit(0)
   
    

     
    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev

    
    

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device,
                                                  autoboost, optimizer_boost, lambda_div=args.lambda_div)
        val_loss = evaluate_accuracy(dev_loader, model, device, autoboost)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch,
                                                   running_loss,val_loss))
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
            if autoboost:
                torch.save(autoboost.state_dict(), os.path.join(model_save_path, 'best_autoboost.pth'))
            print("Best model saved at epoch {}".format(epoch))

    
