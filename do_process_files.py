#!/usr/bin/env python3
import numpy as np
import glob
import re
import librosa
from scipy import stats
import os.path
from scipy.io import wavfile
from os import path
from librosa.filters import mel as librosa_mel_fn
#import matplotlib.pyplot as plt

prm_WAVERNN = {
  'hop_length':275,
  'win_length':1024,
  'filter_length':1024,
  'n_mel_channels':80,
  'fmin':40.0,
  'fmax':8000.0
}
prm_WAVEGLOW = {
  'hop_length':256,
  'win_length':1024,
  'filter_length':1024,
  'n_mel_channels':80,
  'fmin':0.0,
  'fmax':8000.0
}

fe_wav=16000; f0_moy=150; n=55; ind=list(range(18))+[36,37];

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)

lpcnet_demo="~/Documents/LPCNet-master/lpcnet_demo"
lpcnet_demo="/research/crissp//LPCNet-master/lpcnet_demo"

myfiles = glob.glob("_wav_22050/GD*_EG_*.wav")
for nm_wav in myfiles:
  fs, wav = wavfile.read(nm_wav)
  lg_wav_s=len(wav)/fs; mx_wav=max(abs(wav))
  nm=re.search('([^\/]+?)\.wav',nm_wav).group(1)
  print("{}: {}Hz {}s max={}".format(nm,fs,lg_wav_s,mx_wav))
  #~ # Analyse WAVERNN
  #~ nm_WAVERNN = "_WAVERNN/"+nm+".WAVERNN"
  #~ if not path.exists(nm_WAVERNN):
    #~ nm_npy="WaveRNN/WaveRNN-master/data/mel/"+nm+".npy"
    #~ if path.exists(nm_npy):
      #~ MEL=np.load(nm_npy).transpose(); header=MEL.shape; nbt=header[0];
      #~ fp=open(nm_WAVERNN,'wb')
      #~ fp.write(np.asarray(header,dtype=np.int32))
      #~ fp.write(MEL.copy(order='C'))
      #~ fp.close();
    #~ else:
      #~ print("{} inexistant".format(nm_py))
  #~ else:
   #~ header = tuple(np.fromfile(nm_WAVERNN, count=2, dtype=np.int32)); nbt=header[0];
  #~ Hz=nbt/lg_wav_s
  #~ print('WAVERNN {} {:.2f}Hz'.format(header,Hz),flush=True)
  #~ # Analyse LPC_Net
  #~ nm_f32 = "_f32/"+nm+".f32"
  #~ if not path.exists(nm_f32):
    #~ os.system("sox -V "+nm_wav+" -b 16 -r 16000 -t raw /tmp/sig.s16")
    #~ os.system(lpcnet_demo+" -features /tmp/sig.s16 _f32/"+nm+".f32")
  #~ # Compression LPC_Net
  #~ nm_LPCNet = "_LPCNet/"+nm+".LPCNet"
  #~ if not path.exists(nm_LPCNet):
    #~ with open(nm_f32, 'rb') as FH:
      #~ s=np.fromfile(FH,dtype='f4');
    #~ s=s.reshape(-1,n);
    #~ s[:,36]=10*(np.log10(fe_wav/(.1+50*s[:,36]+100))-np.log10(f0_moy))
    #~ nbt=np.size(s,0)
    #~ LPC=s[:,ind]
#~ #    np.save("_LPCNet/"+nm+"_LPCNet.npy",LPC)
    #~ mn=np.min(LPC,axis=0); mx=np.max(LPC,axis=0); mn_f0=mn[18]; mx_f0=mx[18];
    #~ print("Lg({}: {:3f}s {}".format(nm,lg_wav_s,nbt))
    #~ print("F0({}: {:3f} {:3f}".format(nm,mn_f0,mx_f0))
#~ #    print('{}'.format(LPC[0,:]))
    #~ header=LPC.shape; nbt=header[0]
    #~ fp=open(nm_LPCNet,'wb')
    #~ fp.write(np.asarray(header,dtype=np.int32))
    #~ fp.write(LPC.copy(order='C'),dtype=np.float32)
    #~ fp.close();
  #~ else:
   #~ header = tuple(np.fromfile(nm_LPCNet, count=2, dtype=np.int32)); nbt=header[0];
  #~ Hz=nbt/lg_wav_s
  #~ print('LPCNet {} {:.2f}Hz'.format(header,Hz),flush=True)
  # Analyse WAVEGLOW
  nm_WAVEGLOW = "_WAVEGLOW/"+nm+".WAVEGLOW"
  if not path.exists(nm_WAVEGLOW):
    wav=wav/mx_wav
    mel=librosa.feature.melspectrogram(y=np.float32(wav), sr=fs, power=1,
            win_length=prm_WAVEGLOW.get('win_length'), hop_length=prm_WAVEGLOW.get('hop_length'), fmin=prm_WAVEGLOW.get('fmin'), fmax=prm_WAVEGLOW.get('fmax'), n_mels=prm_WAVEGLOW.get('n_mel_channels'))
    mel=np.log(mel.clip(1e-5)/10.0).transpose()+1.23
    header=mel.shape+(fs,prm_WAVEGLOW.get('hop_length')); nbt=header[0]
    fp=open(nm_WAVEGLOW,'wb')
    fp.write(np.asarray(header,dtype=np.int32))
    fp.write(mel.copy(order='C'))
    fp.close();
  else:
    header = tuple(np.fromfile(nm_WAVEGLOW, count=4, dtype=np.int32)); nbt=header[0]
  Hz=nbt/lg_wav_s
  print('WAVEGLOW {} {:.2f}Hz'.format(header,Hz),flush=True)

#    for i in range(n_mel_channels):
#      a = mel[i,:]
#      b = mel_tacotron[i,:]
#      slope, intercept, r_value, p_value, std_err = stats.linregress(a,b)
#      print("{:3f} {:3f} {:3f}".format(slope,intercept,r_value))
#      ax.clear(); ax.plot(a); ax.plot(b);ax.legend(['mel','mel_tacotron']);
#      plt.draw(); plt.pause(0.1)

