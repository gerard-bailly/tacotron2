#!/usr/bin/env python3
import sys
import re
import os
import scipy.io
import numpy as np
import torch
import argparse
from os import path, system
import wave
MAX_WAV_VALUE = 32768.0
import gc
import yaml
import json

from load_csv import load_csv
from def_symbols import init_symbols, text_to_sequence
from model import Tacotron2, to_gpu
from py3nvml.py3nvml import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
#matplotlib.use('Agg')
import scipy.interpolate as interpolate

hps = []
exe = '/research/crissp/LPCNet-master/lpcnet_demo_NEB -synthesis'
spk_imposed=-1
style_imposed=-1

def check_gpu(msg):
    mo=pow(1024.0,2)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    if torch.cuda.is_available():
    	handle = nvmlDeviceGetHandleByIndex(num_gpu)
    	info = nvmlDeviceGetMemoryInfo(handle)
    	print("Device {}: {} Free_memory={}/{}MiB".format(num_gpu, nvmlDeviceGetName(handle), info.free >> 20, info.total >> 20))

def check_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print("TENSOR> {} [{}]".format(type(obj),obj.size()))
        except:
            pass

def synthesis(nm):
	if (hps['ext_data']=='LPCNet'):
		cmd='{} {}.f32 {}.pcm'.format(exe,nm,nm); system(cmd); print(cmd);
		cmd='sox -t raw -b 16 -e signed-integer -r 16000 {}.pcm {}.wav'.format(nm,nm); system(cmd); print(cmd);
		print('{}.wav created'.format(nm), flush=True);
		cmd='/bin/rm -f {}.pcm {}.f32'.format(nm,nm); system(cmd); print(cmd);
	if (hps['ext_data']=='WAVERNN'):
		os.chdir('WaveRNN/WaveRNN-master')
		nm_sd=nm.split('/',1)[1]
		cmd='{} ../../{}.npy'.format(exe,nm); system(cmd); print(cmd);
		cmd='mv model_outputs/ljspeech_mol.wavernn/__{}__797k_steps_gen_batched_target11000_overlap550.wav ../../{}.wav'.format(nm_sd,nm); system(cmd); print(cmd);
		os.chdir('../../')

nms_data=[]

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_directory', type=str, help='directory to save generated files')
parser.add_argument('--config', required=False, type=str, default="tc2.yaml", help='configuration file')
parser.add_argument('--num_gpu', required=False, type=int, default='0', help='number of the gpu')
parser.add_argument('--no_auto_numbering', required=False, action='store_true', default=False, help='no adding of _{:04d} at the end of files')
parser.add_argument('-p', '--prediction', required=False, action='store_true', default=False, help='prediction instead of synthesis')
parser.add_argument('--play_wav', required=False, action='store_true', default=False, help='play sound')
parser.add_argument('--draw', required=False, action='store_true', default=False, help='plot behaviour')
parser.add_argument('--overwrite', required=False, action='store_true', default=False, help='no overwrite')
parser.add_argument('-g', '--ground_truth', required=False, action='store_true', default=False, help='generate ground-truth parameter files')
parser.add_argument('--parameter_files', required=False, action='store_true', default=False, help='generate parameter files')
parser.add_argument('-t', '--tacotron', type=str, default='tacotron2_IT+PHO', required=False, help='Tacotron model')
parser.add_argument('-v', '--vocoder', type=str, default='waveglow_NEB.pt', required=False, help='Vocoder model')
parser.add_argument('--speaker', type=str, default='MT', required=False, help='speaker')
parser.add_argument('--style', type=str, default='NONE', required=False, help='style')
parser.add_argument('-r', '--sampling_rate', type=int, default='22050', required=False, help='sampling rate')
#parser.add_argument('-e', '--exe', type=str, default='/research/crissp/LPCNet-master/lpcnet_demo_DG -synthesis', required=True, help='vocoder exe')
parser.add_argument('-e', '--exe', type=str, default='', required=False, help='vocoder exe')
parser.add_argument('--hparams', type=str, required=False, help='comma separated name:value pairs')
parser.add_argument('--phonetic_only', action='store_true', default=False, help='output only phonetic predictions')
parser.add_argument('--silent', action='store_true', default=False, help='run silently')
args = parser.parse_args()

hps = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
hps['save_embeddings']=''
hparams = args.hparams
if hparams: hps.update(yaml.safe_load(hparams))
init_symbols(hps)
code_PAR=text_to_sequence('§')[0] # symbol for text spliting
code_POINT=text_to_sequence('.')[0] # symbol for end of utterance... to be replaced by end of paragraph at the end of each entry
code_SILENT=0

exe = args.exe
spk = args.speaker
style = args.style
no_auto_numbering = args.no_auto_numbering

play_wav = args.play_wav
if play_wav:
    import sounddevice as sd
nm_tacotron2 = args.tacotron
gen_pf = args.parameter_files
vocoder = args.vocoder
ground_truth = args.ground_truth
overwrite = args.overwrite
prediction = args.prediction
fe_wav = args.sampling_rate
phonetic_only = args.phonetic_only
silent = args.silent
num_gpu = args.num_gpu

if args.draw: # Display Graph
	matfig=plt.figure(figsize=(15,10))
	hps['output_alignments']=True
	
device = torch.device("cuda:%d" % (num_gpu) if torch.cuda.is_available() else "cpu")
if device!="cpu":
	nvmlInit()
	torch.cuda.set_device(num_gpu)
	check_gpu("START")

if spk in hps['speakers']:
  spk_imposed=hps['speakers'].index(args.speaker);
  print('SPK [{}]={}'.format(hps['speakers'],spk_imposed))
else:
  print('SPK {} not in {}'.format(args.speaker,hps['speakers']))

if path.exists(vocoder):
	if vocoder.find('waveglow')>=0:
		sigma=0.6
		sys.path.append('waveglow')
		waveglow = torch.load(vocoder,device)['model']
		waveglow = waveglow.remove_weightnorm(waveglow)
		if torch.cuda.is_available() : waveglow.cuda().eval()
		print('WAVEGLOW: MODEL {} loaded'.format(vocoder),flush=True);
	elif vocoder.find('hifigan')>=0:
		sys.path.append('hifigan')
		from env import AttrDict
		from models import Generator
		with open("hifigan/config.json", "r") as f: config = json.load(f)
		config = AttrDict(config)
		hgan = Generator(config)
		ckpt = torch.load(vocoder,map_location=torch.device('cpu'))
		hgan.load_state_dict(ckpt["generator"])
		hgan.eval()
		hgan.remove_weight_norm()
		param_size=buffer_size=0
		for param in hgan.parameters(): param.requires_grad=False; param_size+=param.nelement()*param.element_size()
		for buffer in hgan.buffers(): buffer_size+=buffer.nelement()*buffer.element_size()
		size_all_mb = (param_size + buffer_size) / 1024**2
		if torch.cuda.is_available() : hgan.to(device)
		if not silent: print('HIFIGAN: MODEL {} loaded: {:.2f} Mo'.format(vocoder,size_all_mb),flush=True);
else:
	print('VOCODER: MODEL {} not found. No synthesis'.format(vocoder),flush=True);
	vocoder=None
	#~ import socket
	#~ import subprocess
	#~ import SharedArray as sa
	#~ a = sa.create("shm://waveglow",shape=(hps['max_decoder_steps'],hps['dim_data']),dtype=np.float32)
	#~ p=subprocess.Popen(['cd waveglow; python3', 'do_inference_waveglow.py'], shell=False)

	#~ s = socket.socket();
	#~ s.connect((socket.gethostname(), 8080))

	#~ msg='MODEL:../waveglow_NEB.pt'; s.send(msg.encode('utf-8'))
	#~ msg=s.recv(1024).decode('utf-8')

hps['mask_padding']=False # only one file at a time!
model = Tacotron2(hps).to(device)
if path.exists(nm_tacotron2):
	model_dict=torch.load(nm_tacotron2, map_location='cpu')['state_dict']
	if not silent:
		print('List of the checkpoint''s modules');
		for k, v in model_dict.items(): print(k,list(v.shape))
	for key in list(model_dict.keys()):
		if re.search(r'decoderVisual',key): model_dict[key.replace('decoderVisual','decoder.1')] = model_dict.pop(key)
		if re.search(r'decoder\.[^\d]',key): model_dict[key.replace('decoder','decoder.0',1)] = model_dict.pop(key)
		if re.search(r'postnet\.[^\d]',key): model_dict[key.replace('postnet','postnet.0',1)] = model_dict.pop(key)
	phonetize=model_dict.get('phonetize.linear_layer.weight') #change of number of phonetic embeddings
	if phonetize!=None:
		nb=len(hps['out_symbols'])-phonetize.shape[0]
		if nb>0:
			phonetize=torch.cat((phonetize,torch.zeros(nb,phonetize.shape[1])))
			model_dict.update({('phonetize.linear_layer.weight',phonetize)})
			phonetize=model_dict.get('phonetize.linear_layer.bias')
			phonetize=torch.cat((phonetize,-1000.0*torch.ones(nb)))
			model_dict.update({('phonetize.linear_layer.bias',phonetize)})
			print('{} phonemes added'.format(nb))
	model.load_state_dict(model_dict)
	print('Tacotron2 model "{}" loaded'.format(nm_tacotron2))
else:
	print('Tacotron2 model "{}" not found'.format(nm_tacotron2))
	sys.exit()
check_gpu("AFTER LOADING MODELS")

(data_test, nms_data)=load_csv(hps['nm_csv_test'], hps, sort_utt=False, check_org=False)
if prediction==False:
	suffix='syn'
else:
	suffix='prd'
model.eval()
torch.set_grad_enabled(False)
if phonetic_only: hps['dim_data']=[]; model.set_dim_data([]) # only phonetic prediction
dim_data=hps['dim_data']; fe_data=hps['fe_data']; nb_out=len(dim_data)

c_prec=code_PAR
i_syn=0
nm_base='INPUT'
while 1:
	i_syn=i_syn+1;
	all_text_in=input("Your input>>");
	if (re.search('^[\[\.§:?!¬;,(§]',all_text_in)==None): all_text_in='§'+all_text_in # if no initial punctuation: use of last from previous text
	if (re.search('[\[\.§:?!¬;,(§]$',all_text_in)==None): all_text_in=all_text_in+'.§' # if no final punctuation: end of chapter
	if spk_imposed>=0: spk_in=spk_imposed
	l_tags=re.findall(r'\<([^\>]*)\>\s*',all_text_in); # tags in text
	if l_tags:
		all_text_in=re.sub(r'\<([^\>]*)\>\s*','',all_text_in);  #remove tags
		for tags in l_tags:
			lt=re.findall("(\w+)=([^;]+)",tags);
			for t in lt:
				if t[0]=='SPK' and t[1] in hps['speakers']: spk_in=hps['speakers'].index(t[1])
	all_text_in = np.array(text_to_sequence(all_text_in)); lg_in=len(all_text_in);
	tensor_spk_in=to_gpu(torch.LongTensor([spk_in])[None, :])
	if hps['nb_styles']:
		if style_imposed>=0: style_in=style_imposed
		tensor_style_in=to_gpu(torch.LongTensor([style_in])[None, :])
	else:
		tensor_style_in=[]; style_in=0
		
	wf_syn, out_par, spe_org = nb_out*[None], nb_out*[None], nb_out*[None]
	if not no_auto_numbering: nm_base+='_{:04d}'.format(i_syn)
	for i_out in range(nb_out):
		nm_syn='_syn_{}/{}_{}'.format(hps['dir_data'][i_out],nm_base,suffix)
		if hps['ext_data'][i_out]=='WAVEGLOW' and vocoder is not None:
			wf_syn[i_out]=wave.open(nm_syn+'.wav','wb'); wf_syn[i_out].setparams((1,2,fe_wav,0,'NONE','not compressed'));
			if not silent: print('WAVEGLOW: {}.wav created'.format(nm_syn),flush=True);
		out_par[i_out] = np.empty((0,hps['dim_data'][i_out]),dtype=np.float32)
	if args.prediction==False: # synthesis
		if '(~:?!§¬«».#;,[])"'.find(hps['symbols'][all_text_in[0]])<0: all_text_in=[c_prec]+all_text_in; lg_in += 1 # prefix first utterance by a chapter onset
		parts = [i for i,val in enumerate(all_text_in) if val==code_PAR]; nb_parts=len(parts);
		parts=[x for x in parts if (x+1) not in parts] # keep first § if succession of §§
		if all_text_in[-1]==code_PAR:
			parts[-1]=lg_in
		else: parts=parts+[lg_in]
		if parts[0]: parts=[0]+parts
	else:
		parts = [0,lg_in]; # un seul bloc
	c_prec=all_text_in[0]
	d_syn=np.zeros(nb_out,dtype=int);
	for ipart_txt in range(len(parts)-1): # splits of text entry
		all_text_in[parts[ipart_txt]]=c_prec; c_prec=all_text_in[parts[ipart_txt+1]-1]; # text split prefixed by last character (punctuation) of previous split
		text_in=all_text_in[parts[ipart_txt]:parts[ipart_txt+1]]; lg_in=len(text_in); tensor_text_in=torch.Tensor(text_in)[None, :]; tensor_text_in=to_gpu(tensor_text_in).long();
		if prediction==False:
			(part_spe_out, part_spe_out_postnet, part_gate_out, pho_out, dur_out, style, part_alignement, part_embeddings) = model.inference((tensor_text_in,tensor_spk_in,tensor_style_in), hps['seed'])
		else:
			tensor_out, lg_out = nb_out*[None], nb_out*[None]
			for i_out in range(nb_out):
			  if spe_org[i_out].any(): tensor_out[i_out]=torch.Tensor(spe_org[i_out].transpose())[None, :].cuda(); lg_out[i_out]=spe_org[i_out].shape[0]
			(part_spe_out, part_spe_out_postnet, part_gate_out, pho_out, dur_out, style, part_alignement, part_embeddings) = model.forward((tensor_text_in, [lg_in], tensor_spk_in, tensor_style_in, tensor_out, lg_out, [lg_in]))
			for i_out in range(nb_out):
				if len(part_alignement[i_out]): part_alignement[i_out]=part_alignement[i_out].transpose(0,1)
		ch_in=[hps['symbols'][p] for p in text_in]
		if not silent: print('synthesis of chunk {:d} [{:d}-{:d},{:d}]: {}'.format(ipart_txt,spk_in,style_in,lg_in,'|'.join(ch_in)),flush=True)
		if len(pho_out)>0:
			pb=torch.sigmoid(pho_out[0,:,:].cpu()).data.numpy()
#			pb=pb/sum(pb)
#			ind1=pb.max(axis=0); ind=ind1.indices
#			ind2=pb[1:,:].max(axis=0); #second best candidate
#			iok=np.where(ind2.values>.45)
#			ind[iok]=ind2.indices[iok]+1
			ind=pho_out[0,:,:].argmax(axis=0).cpu().data.numpy()
			ph_prd='|'.join(["{}".format(hps['out_symbols'][p]) for p in ind]) #predicted aligned phonetic chain
			print('PH_PRD: {}'.format(ph_prd))
# production d'alignement
		if 0:
			ind_aln = np.append(ind,1); ch_in_aln=ch_in+['x']
			part_aln = part_alignement[0].cpu().data.numpy()
			to_s=hps['n_frames_per_step'][0]/hps['fe_data'][0]
			d_seq=data_test[i_syn][1]
			id=0
			for i_in in range(lg_in):
				if ind[i_in]!=code_SILENT:
					res=next(x for x, val in enumerate(part_aln[0,i_in,id:]) if val > 0.4) if (part_aln[0,i_in,id:].max()>0.4) else 1
					id=id+res
					# nb de caractères
					dd=0 if hps['out_symbols'][ind[i_in]]=='__' else 1
					aa=np.where(np.logical_or(ind_aln[i_in+dd:]!=code_SILENT,np.array([v in ' !?~§\'¬«»[]{}(),.' for v in ch_in_aln[i_in+dd:]])))
					if len(aa[0]):
						nb_car=aa[0][0]+dd
					else:
						nb_car=0 if hps['out_symbols'][ind[i_in]]=='__' else 1
					print('{:3f} {}|{}'.format(d_seq+id*to_s,hps['out_symbols'][ind[i_in]],nb_car),end = '')
					if hps['out_symbols'][ind[i_in]]=='__':
						txt=''; i=i_in;
						while i<lg_in and ch_in[i] in '!?~§\'¬«»[]{}(),.': txt=txt+ch_in[i]; i=i+1;
						print('|{}'.format(txt),end='')
					if len(txt)==0:
						i=i_in;
						while i<lg_in and not ch_in[i] in ' !?~§\'¬«»[]{}(),.': txt=txt+ch_in[i]; i=i+1;
						print('|{}'.format(txt),end='')
					print('')
				if ch_in[i_in] in ' !?~§\'¬«»[]{}().': txt='';
								
		if len(dur_out[0,:]):
			ch_dprd=' '.join(["{}".format(int(d)) for d in 100*dur_out[0,:].cpu().detach().numpy()[0,:]])
			if not silent: print('prd_dur: {}'.format(ch_dprd))
		for i_out in range(nb_out):
			lg_part_out=part_spe_out[i_out].shape[2]
			if i_out>0:
				lg_part_ref=int(part_spe_out_postnet[0].shape[2]*hps['fe_data'][i_out]/hps['fe_data'][0])
				if lg_part_out>lg_part_ref:
					part_spe_out_postnet[i_out]=part_spe_out_postnet[i_out][:,:,0:lg_part_ref-1];
					part_spe_out[i_out]=part_spe_out[i_out][:,:,0:lg_part_ref-1];
					lg_part_out=lg_part_ref
			d_syn[i_out]+=lg_part_out
			if not silent: print('{}: {}, {}->{:.2f}\n'.format(i_out, d_syn[i_out], lg_part_out,d_syn[i_out]/hps['fe_data'][i_out]))
			if len(part_alignement[i_out]):
				part_aln = part_alignement[i_out].cpu().data.numpy()[0]
				part_gate = torch.sigmoid(part_gate_out[i_out].cpu()).data.numpy()[0,:]
				ms_from_act = 1000.0*hps['n_frames_per_step'][i_out]*part_aln.sum(axis=1)/hps['fe_data'][i_out]
				print(''.join(['|{:.0f}{}'.format(ms_from_act[x],ch_in[x]) for x in range(0,len(ch_in))]))

			part_out=part_spe_out_postnet[i_out].cpu().data.numpy()
			part_out=part_out[0,:,:].transpose()
			
			if not silent: print('prd_{}: {:.2f}s'.format(hps['ext_data'][i_out],lg_part_out/hps['fe_data'][i_out]));
			
			if ipart_txt: # insert silence 300ms
				nt=int(0.3*hps['fe_data'][i_out])
				d_syn[i_out]+=nt
				if wf_syn[i_out]:
					ne=int(nt*fe_wav/hps['fe_data'][i_out]);
					wf_syn[i_out].writeframes(np.zeros(ne,dtype='int16'));
				out_par[i_out]=np.concatenate((out_par[i_out],np.tile(part_out[0,:],(nt,1))))
			if wf_syn[i_out]:
				with torch.no_grad():
					if vocoder.find('waveglow')>=0:
						audio = MAX_WAV_VALUE * waveglow.infer(part_spe_out_postnet[i_out], sigma=sigma)[0]
					elif vocoder.find('hifigan')>=0:
						audio = MAX_WAV_VALUE * hgan(part_spe_out_postnet[i_out][0])[0]
				audio = audio.cpu().numpy().astype('int16');
				wf_syn[i_out].writeframes(audio)
				if play_wav: sd.play(audio,fe_wav)

			out_par[i_out]=np.concatenate((out_par[i_out],part_out))

			# ----------- Display Attention alignments of each chunk ----------------
			if (args.draw) :
				aln=(1.0+np.arange(lg_in)).dot(part_aln)-1.0

				if (i_out==0): plt.clf()
				aa=plt.subplot(2, nb_out, i_out+1)
				plt.matshow(part_aln,origin='lower',aspect='auto',fignum=0,vmin=0,vmax=1)
				axes = plt.gca()
				axes.get_xaxis().set_visible(False)
				plt.ylabel('Time')
				axes.yaxis.set_ticks_position('left')
				axes.yaxis.set_major_locator(MultipleLocator(1))
				axes.yaxis.set_major_formatter(FormatStrFormatter("%s"))
				axes.set_yticks(1+np.arange(lg_in))
				axes.yaxis.set_ticklabels(ch_in, fontsize=7, rotation=90)
				plt.ylabel('Encoder states')
				nm_syn='_syn_{}/{}_{}'.format(hps['dir_data'][i_out],nm_base,suffix)
				plt.title('{}: "{}"'.format(nm_syn,''.join(ch_in)),fontsize=9,pad=10)
				plt.plot(aln,'w-',linewidth=3)
				plt.draw()
				plt.subplot(2, nb_out, nb_out+i_out+1)
				if wf_syn[i_out]:
					plt.plot(hps['fe_data'][i_out]*np.arange(len(audio))/fe_wav/hps['n_frames_per_step'][i_out],audio/10000,'k',linewidth=0.5)
					if play_wav: sd.play(audio,fe_wav)
				else:
					plt.plot(part_out.mean(axis=1))
					if spe_org[i_out] is not None: plt.plot(spe_org[i_out].mean(axis=1),linestyle='-.')
				plt.plot(part_gate,'r-',linewidth=0.5)
				plt.ylim(-2.1,2.1)
				plt.xlim(0,len(part_gate))
				plt.draw()
		if (args.draw):
			matfig.show()
			plt.waitforbuttonpress()
# resynchronize
		if len(out_par):
			lg=[out_par[i_out].shape[0]/hps['fe_data'][i_out] for i_out in range(nb_out)]; lg_max=max(lg)
			for i_out in range(nb_out):
				nt=round((lg_max-lg[i_out])*hps['fe_data'][i_out])
				if nt:
					d_syn[i_out]+=nt
					out_par[i_out]=np.concatenate((out_par[i_out],np.tile(out_par[i_out][-1,:],(nt,1))))
					if wf_syn[i_out]:
						ne=int(nt*fe_wav/hps['fe_data'][i_out]);
						wf_syn[i_out].writeframes(np.zeros(ne,dtype='int16'));
	for i_out in range(nb_out):
		if not silent: print('dur_syn[{}]={:.3f}s'.format(i_out,d_syn[i_out]/hps['fe_data'][i_out]))
		if wf_syn[i_out]: wf_syn[i_out].close();
		if gen_pf:
			nm_syn='_syn_{}/{}_{}'.format(hps['dir_data'][i_out],nm_base,suffix)
			fp=open(nm_syn+'.'+hps['ext_data'][i_out],'wb')
			if type(hps['fe_data'][i_out]) is int:
				num=hps['fe_data'][i_out]; den=1;
			else:
				(num,den)=(hps['fe_data'][i_out]).as_integer_ratio()
			fp.write(np.asarray(out_par[i_out].shape+(num,den),dtype=np.int32))
			fp.write(out_par[i_out].copy(order='C'))
			fp.close()
			print('{}.{} created [{}, {:d}ms]'.format(nm_syn,hps['ext_data'][i_out],out_par[i_out].shape,int(1000.0*out_par[i_out].shape[0]/hps['fe_data'][i_out])), flush=True)
