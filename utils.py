import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from tqdm import tqdm 
from glob import glob
from torchaudio.transforms import Spectrogram, MelSpectrogram , ComplexNorm
from torchaudio.transforms import TimeStretch, AmplitudeToDB 
from torch.nn import ConstantPad1d
from torch.distributions import Uniform
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

def data_pre():
    if not os.path.isfile('./tensor_dict.pt'):
        print('data need process')
        rate = 48000

        names = [file.split('/')[1][0:5] for file in sorted(glob('audio/*'))]

        for name in tqdm(names):
            waveform_, sample_rate = torchaudio.load('audio/'+name+'.mp3')

            #resample waveform to rate
            if sample_rate != rate:
                waveform_ = torchaudio.compliance.kaldi.resample_waveform(
                                                    waveform=waveform_,
                                                    orig_freq=sample_rate,
                                                    new_freq=rate)
                torchaudio.save('audio/'+name+'.mp3',waveform_,sample_rate=48000)

        rate = 48000
        names = [file.split('/')[1][0:5] for file in sorted(glob('audio/*'))]
        window_size = 6 #sec
        shift_size = 3 #sec
        pad_len = window_size*rate
        typs = ['intro','verse','bridge','outro','break','Refrain','silece','other']
        dic = {k:v for v,k in enumerate(typs)}
        xs = torch.LongTensor([])
        ys = torch.LongTensor([])

        g = 0
        group = np.array([])
        for name in tqdm(names):
            waveform_, sample_rate = torchaudio.load('audio/'+name+'.mp3')
            #pading zero in sizr window_size/2 on start & end 
            pad_len = (window_size/2)*rate
            pad = ConstantPad1d(int(pad_len),0)
            waveform = pad(waveform_[0])

            #make a sliding window 
            x = waveform.unfold(dimension = 0,
                                     size = window_size*rate,
                                     step =shift_size*rate).unsqueeze(1)

            #get labels
            f = open('Labels/'+name+'.txt')
            txts = f.readlines()
            c = []
            i = 0
            for txt in txts:
                tmp = txt.split()
                typs = dic.get(tmp[2].replace('\n',''))
                start = float(tmp[0])
                end = float(tmp[1])
                while start <= i < end:
                    #print(start,end,i,typs)
                    c.append(typs)
                    i += shift_size
            y = torch.LongTensor(c)
            f.close()

            #make label and musiz len is same
            if x.shape[0] > y.shape[0]:
                x = x[:y.shape[0]]
            if y.shape[0] > x.shape[0]:
                y = y[:x.shape[0]]

            #make a group for GroupShuffleSplit
            group = np.append(group,np.repeat(g,y.shape[0]))
            g += 1

            #extand 
            xs = torch.cat((xs,x))
            ys = torch.cat((ys,y))   

            #save
            dic = {'xs':xs,'ys':ys,'group':group}
            torch.save(dic, 'tensor_dict.pt')
        
    print('data do not need process')
    dic = torch.load('tensor_dict.pt')
    xs,ys,group = [dic[c] for c in dic]
    gss = GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)
    for train_index, test_index in gss.split(xs, ys, group):
        X_train, X_test = xs[train_index], xs[test_index]
        y_train, y_test = ys[train_index], ys[test_index]

    dataloader_X_train = DataLoader(X_train,batch_size=64,shuffle=False, num_workers=0,drop_last=True)
    dataloader_y_train= DataLoader(y_train,batch_size=64,shuffle=False, num_workers=0,drop_last=True)

    dataloader_X_test = DataLoader(X_test,batch_size=64,shuffle=False, num_workers=0,drop_last=True)
    dataloader_y_test= DataLoader(y_test,batch_size=64,shuffle=False, num_workers=0,drop_last=True)

    return dataloader_X_train,dataloader_y_train,dataloader_X_test,dataloader_y_test


def _num_stft_bins(lengths, fft_length, hop_length, pad):
    return (lengths + 2 * pad - fft_length + hop_length) // hop_length

class MelspectrogramStretch(MelSpectrogram):

    def __init__(self, hop_length=None, 
                       sample_rate=44100, 
                       num_mels=512, 
                       fft_length=2048, 
                       norm='whiten', 
                       stretch_param=[0.4, 0.4]):

        super(MelspectrogramStretch, self).__init__(sample_rate=sample_rate, 
                                                    n_fft=fft_length, 
                                                    hop_length=hop_length, 
                                                    n_mels=num_mels)

        self.stft = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length, pad=self.pad, 
                                       power=None, normalized=False)

        # Augmentation
        self.prob = stretch_param[0]
        self.random_stretch = RandomTimeStretch(stretch_param[1], 
                                                self.hop_length, 
                                                self.n_fft//2+1, 
                                                fixed_rate=None)
        
        # Normalization (pot spec processing)
        self.complex_norm = ComplexNorm(power=2.) #Compute the norm of complex tensor input. Power of the norm. (Default: to 1.0)
        self.norm = SpecNormalization(norm) #whiten:(z norm) db:(分貝 norm)

    def forward(self, x, lengths=None):
        x = self.stft(x)

        if lengths is not None:
            lengths = _num_stft_bins(lengths, self.n_fft, self.hop_length, self.n_fft//2)
            lengths = lengths.long()
        
            if torch.rand(1)[0] <= self.prob and self.training:
                # Stretch spectrogram in time using Phase Vocoder
                x, rate = self.random_stretch(x)
                # Modify the rate accordingly
                lengths = (lengths.float()/rate).long()+1
        
        x = self.complex_norm(x)
        x = self.mel_scale(x)

        # Normalize melspectrogram
        x = self.norm(x)

        if lengths is not None:
            return x, lengths        
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTimeStretch(TimeStretch):

    def __init__(self, max_perc, hop_length=None, n_freq=201, fixed_rate=None):

        super(RandomTimeStretch, self).__init__(hop_length, n_freq, fixed_rate)
        self._dist = Uniform(1.-max_perc, 1+max_perc)

    def forward(self, x):
        rate = self._dist.sample().item()
        return super(RandomTimeStretch, self).forward(x, rate), rate


class SpecNormalization(nn.Module):

    def __init__(self, norm_type, top_db=80.0):

        super(SpecNormalization, self).__init__()

        if 'db' == norm_type:
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            self._norm = lambda x: self.z_transform(x)
        else:
            self._norm = lambda x: x
        
    
    def z_transform(self, x):
        # Independent mean, std per batch
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean)/std 
        return x

    def forward(self, x):
        return self._norm(x)
    
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        
        self.fft = MelspectrogramStretch() #Fast Fourier Transform featrue = 512   
        
        cnn = nn.Sequential()
        cnn.add_module('conv{0}',   nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]))
        cnn.add_module('norm{0}',   nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        cnn.add_module('relu{0}',   nn.ELU(alpha=1.0))
        cnn.add_module('pooling{0}',nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False))
        cnn.add_module('drop{0}',   nn.Dropout(p=0.1))
                       
        cnn.add_module('conv{1}',   nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]))
        cnn.add_module('norm{1}',   nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        cnn.add_module('relu{1}',   nn.ELU(alpha=1.0))
        cnn.add_module('pooling{1}',nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False))
        cnn.add_module('drop{1}',   nn.Dropout(p=0.1))
     
        cnn.add_module('conv{2}',   nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]))        
        cnn.add_module('norm{2}',   nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        cnn.add_module('relu{2}',   nn.ELU(alpha=1.0))
        cnn.add_module('pooling{2}',nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False))
        cnn.add_module('drop{2}',   nn.Dropout(p=0.1))
        self.cnn=cnn
        
        self.LSTM        = nn.LSTM(input_size = 5,hidden_size = 64,num_layers=2) #input_size change buy windows size
        self.Dropout     = nn.Dropout(p=0.1)
        self.BatchNorm1d = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.Linear      = nn.Linear(in_features=64 , out_features=10, bias=True)
        self.Linear2     = nn.Linear(in_features=6400 , out_features=128, bias=True)
        self.Linear3     = nn.Linear(in_features=128 , out_features=8, bias=True)  #category number 
    
    def forward(self, x):                      #(seq)
        x   = self.fft(x)                      #(batch,1,seq)
        x   = self.cnn(x)                      #(batch,chanel,featrue,seq)
        x   = x.flatten(start_dim=1,end_dim=2) #(batch,chanel*featrue,seq)
        x   = x.transpose(0,1)                 #(seq,batch,chanel*featrue)
        x,_ = self.LSTM(x)                     #(seq,batch,64)
        x   = self.Dropout(x)                  #(seq,batch,64)
        x   = x.transpose(0,1)                 #(batch,seq,64)
        x   = x.transpose(1,2)                 #(batch,64,seq)
        x   = self.BatchNorm1d(x)              #(batch,64,seq)
        x   = x.transpose(1,2)                 #(batch,seq,64)      
        x   = self.Linear(x)                   #(batch,seq,10)
        x   = x.flatten(start_dim=1)           #(batch,seq*10)
        x   = self.Linear2(x)                  #(batch,128)
        x   = self.Linear3(x)                  #(batch,10)
        return F.softmax(x)                    #(batch,10)
    
def adjusting_learning_rate(optimizer, factor=.5, min_lr=0.00001):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = max(old_lr * factor, min_lr)
        param_group['lr'] = new_lr
        print('adjusting learning rate from %.6f to %.6f' % (old_lr, new_lr))
    