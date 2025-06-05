#!/usr/bin/env python3
import sys
import os
import re
import numpy as np
import numpy.ma as ma
import torch
from torch import nn
from torch.utils.data.sampler import Sampler
import torch.optim as optim
import argparse
from os import path
from load_csv import load_csv
#from text import text_to_sequence
#from text.symbols import symbols, out_symbols
from def_symbols import init_symbols, text_to_sequence
from model import Tacotron2, get_mask_from_lengths
import pdb
import yaml
import gc

_symbol_to_id, _id_to_symbol, symbols, out_symbols, _out_symbol_to_id = [], [], [], [], []

torch.autograd.set_detect_anomaly(True)

nnBCE=nn.BCEWithLogitsLoss()
nnBCE_nored=nn.BCEWithLogitsLoss(reduction='none')
nnCE = nn.CrossEntropyLoss(ignore_index=-1)
nnCE_nored=nn.CrossEntropyLoss(reduction='none',ignore_index=-1)
nnKLD=nn.KLDivLoss()
nnKLD_nored=nn.KLDivLoss(reduction='none')
nnMSE=nn.MSELoss()
nnMSE_nored=nn.MSELoss(reduction='none')

import pynvml
pynvml.nvmlInit()

def check_gpu(msg):
  if torch.cuda.is_available():
  	handle = pynvml.nvmlDeviceGetHandleByIndex(num_gpu)
  	info = pynvml.nvmlDeviceGetMemoryInfo(handle)
  	print("Device {}: {} Free_memory={}/{}MiB".format(num_gpu, pynvml.nvmlDeviceGetName(handle), info.free >> 20, info.total >> 20))

hps = []
nms_data=[]
(CSV_num_fic, CSV_start_spe, CSV_lg_spe, CSV_txt_in, CSV_lg_in, CSV_spk_in, CSV_style_in, CSV_pho_out, CSV_dur_out) = range(9)
(BATCH_text_in, BATCH_lg_in, BATCH_spk_in, BATCH_style_in, BATCH_spe_tgt, BATCH_gate_tgt, BATCH_lg_tgt, BATCH_pho_tgt, BATCH_dur_tgt, BATCH_lg_pho_out, BATCH_i_nm)=range(11)
stateful=False

def check_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print("TENSOR> {} [{}]".format(type(obj),obj.size()))
        except:
            pass

class OrderedSampler(Sampler):

    def __init__(self, train_data, batch_size, drop_last=True):
        self.train_data = train_data
        self.batch_size = batch_size
        self.nb_utts = len(data_train)
        self.drop_last = drop_last

    def __iter__(self):
        lst=np.arange(self.nb_utts)
        while len(lst)>=self.batch_size:
            i_first=np.random.randint(len(lst))
            ind=np.argsort(abs(lst-lst[i_first]))
            batch=lst[ind[0:self.batch_size]];
            lst=np.delete(lst,ind[0:self.batch_size])
#            dm=np.mean([e[CSV_lg_spe] for i,e in enumerate(np.array(self.train_data,dtype=object)[batch])]); print('dm={:.2f}s'.format(dm));
            yield batch
        batch=lst; print('LAST:= {}/{} {}'.format(len(batch),len(lst),self.drop_last))
        if len(lst) > 0 and not self.drop_last:
            yield lst

    def __len__(self):
        return self.nb_utts
        
class BatchSampler(Sampler):

    def __init__(self, train_data, batch_size, drop_last=True):
        self.train_data = train_data
        self.batch_size = batch_size
        self.nb_utts = len(data_train)
        self.drop_last = drop_last
        self.nb_batch = self.nb_utts//self.batch_size
        self.nb_last = self.nb_utts-self.nb_batch*self.batch_size

    def __iter__(self):
        batch = []
        for idx in range(0,self.nb_batch) :
            batch = np.arange(idx,self.nb_utts-self.nb_last,self.nb_batch)
            yield batch
        if len(batch) > 0 and not self.drop_last:
        	batch = np.arange(self.nb_batch*self.batch_size,self.nb_utts)
        	yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size

def get_mask_from_lengths(lengths):
  max_lengths = max(lengths); nb_lengths=len(lengths)
  ids=np.arange(0, max_lengths)
  mask=(ids<np.reshape(lengths,(nb_lengths,1)))
  mask=torch.from_numpy(mask).cuda()
  return mask

def collate_batch(batch):
  # sort by increasing input length
  # batch = (num_fic,start_spe,lg_spe,txt_in,lg_in,spk_in,pho_out,dur_out)
  # -> (text_in, lg_in, spk_in, spe_tgt, gate_tgt, lg_tgt, pho_tgt, dur_tgt, lg_pho_out, i_nm)
  def takeLen_in(elem):
    return elem[CSV_lg_in]
    
  if not stateful: batch.sort(key=takeLen_in,reverse = True)
  lg_batch = len(batch)
  nm=lg_batch*[None]
  dim_data=hps['dim_data']; nb_out=len(dim_data)
  if not silent:
    print('collate_batch {}'.format(lg_batch))
  # item
  spk_in = torch.LongTensor([item[CSV_spk_in] for item in batch]);
  style_in = torch.LongTensor([item[CSV_style_in] for item in batch]);
  lg_tgt= np.zeros([lg_batch,nb_out],dtype=int);  spe_tgt= nb_out*[None]; gate_tgt = nb_out*[None];
  for i_out in range(nb_out):
    fe=hps['fe_data'][i_out]; dim=dim_data[i_out]
    for i_batch in range(lg_batch):
      nm_data = '_'+hps['dir_data'][i_out]+'/'+nms_data[batch[i_batch][CSV_num_fic]]+'.'+hps['ext_data'][i_out]
      if path.exists(nm_data): lg_tgt[i_batch,i_out] = int(0.5+batch[i_batch][CSV_lg_spe]*fe/hps['n_frames_per_step'][i_out])*hps['n_frames_per_step'][i_out]; nm[i_batch]=nm_data
    max_lg = max(lg_tgt[:,i_out])
    if max_lg: # valid trajectories
      spe = np.zeros([lg_batch,dim,max_lg])
      gate = np.zeros([lg_batch,max_lg])
      for i_batch in range(lg_batch):
        lg=lg_tgt[i_batch,i_out]
        if lg:
          deb = int(batch[i_batch][CSV_start_spe]*fe)
          spe[i_batch,:,:lg] = np.memmap(nm[i_batch],mode='r',offset=4*(4+deb*dim),dtype=np.float32,shape=(lg,dim)).transpose()
          gate[i_batch,lg-int(1+hps['lgs_sil_add']*fe):] = 1.0
    else: spe, gate = [], []
    spe_tgt[i_out]=torch.FloatTensor(spe); gate_tgt[i_out]=torch.FloatTensor(gate);
  lg_in = [item[CSV_lg_in] for item in batch]; max_lg_in = max(lg_in)
  text_in = torch.zeros([lg_batch, max_lg_in], dtype=torch.long);
  lg_pho_out = [len(item[CSV_pho_out]) for item in batch]
  pho_tgt = -1*torch.ones([lg_batch, max_lg_in],dtype=torch.long) if hps['dim_out_symbols'] and max(lg_pho_out) else []
  dur_tgt = torch.zeros([lg_batch, max_lg_in],dtype=torch.float32) if hps['compute_durations'] and max(lg_pho_out) else []
  for i_batch in range(lg_batch):
    lg_txt = batch[i_batch][CSV_lg_in]
    text_in[i_batch,:lg_txt] = torch.Tensor(batch[i_batch][CSV_txt_in])
    if hps['dim_out_symbols'] and lg_pho_out[i_batch]:
      lg_ph=len(batch[i_batch][CSV_pho_out])
      if lg_ph!=lg_txt:
        ch='|'.join(["{}".format(hps['symbols'][p]) for p in batch[i_batch][CSV_txt_in]])
        ph_tgt='|'.join(["{}".format(hps['out_symbols'][p]) for p in batch[i_batch][CSV_pho_out]])
        print('{} {}[{}] - {}[{}]'.format(nms_data[batch[i_batch][CSV_num_fic]],ch,lg_txt,ph_tgt,lg_ph))
      pho_tgt[i_batch,:lg_txt] = torch.Tensor(batch[i_batch][CSV_pho_out])
    if hps['compute_durations'] and batch[i_batch][CSV_dur_out]: dur_tgt[i_batch,:lg_txt] = torch.Tensor(batch[i_batch][CSV_dur_out])
    if not silent:
      ch=''.join([hps['symbols'][p] for p in batch[i_batch][CSV_txt_in]]).replace('@','')
      print('BATCH{} [{}, {}:{}]: {} -> {} {}'.format(i_batch,nms_data[batch[i_batch][CSV_num_fic]],batch[i_batch][CSV_spk_in],hps['speakers'][batch[i_batch][CSV_spk_in]],len(batch[i_batch][CSV_txt_in]),lg_tgt[i_batch],ch))
  i_nm = [[item[CSV_num_fic],item[CSV_start_spe]] for item in batch] #fileid, starting_frame
  return [text_in, lg_in, spk_in, style_in, spe_tgt, gate_tgt, lg_tgt, pho_tgt, dur_tgt, lg_pho_out, i_nm] # spe_tgt used for teacher forcing

def warm_start_model(nm_mod, model, ignore_layers):
    print('Warm starting model {}'.format(nm_mod), flush=True)
    
    if False:
      checkpoint_dict = torch.load('mellotron_libritts.pt', map_location='cpu') # load a pre-trained convolutional prosodic encoder
      model_dict = checkpoint_dict['state_dict']
      print('mellotron_libritts.pt')
      lst={};
      for k, v in model_dict.items():
        if k.find('gst.encoder.convs')>=0:
          p=re.compile(r'gst\.encoder\.convs\.(\d+)'); k=p.sub(r'style_encoder.convolutions.\1.0.conv',k); lst[k]=v; print(k,list(v.shape))
        if k.find('gst.encoder.bns')>=0:
          p=re.compile(r'gst\.encoder\.bns.(\d+)'); k=p.sub(r'style_encoder.convolutions.\1.1',k); lst[k]=v; print(k,list(v.shape))
      dummy_dict = model.state_dict(); dummy_dict.update(lst); model_dict = dummy_dict
      model.load_state_dict(model_dict,strict=False)

    checkpoint_dict = torch.load(nm_mod, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    for key in list(model_dict.keys()):
        if re.search(r'decoderVisual',key): model_dict[key.replace('decoderVisual','decoder.1')] = model_dict.pop(key)
        if re.search(r'decoder\.[^\d]',key): model_dict[key.replace('decoder','decoder.0')] = model_dict.pop(key)
        if re.search(r'postnet\.[^\d]',key): model_dict[key.replace('postnet','postnet.0')] = model_dict.pop(key)
    print('List of the checkpoint''s modules');
    
    phonetize=model_dict.get('phonetize.linear_layer.weight') #change of number of phonetic embeddings
    if phonetize!=None:
    	nb_phon = phonetize.shape[0]
    	nbo = len(hps['out_symbols'])
    	if nb_phon<nbo:
    		phonetize=torch.cat((phonetize,phonetize[0,:].repeat(nbo-nb_phon,1)))
    		model_dict.update({('phonetize.linear_layer.weight',phonetize)})
    		phonetize=model_dict.get('phonetize.linear_layer.bias')
    		phonetize=torch.cat((phonetize,-1000.0*torch.ones(nbo-nb_phon)))
    		model_dict.update({('phonetize.linear_layer.bias',phonetize)})
    		print('{} phonemes added'.format(nbo-nb_phon))

    speaker_embeddings=model_dict.get('speaker_embedding.weight') #change of number of speakersif speaker_embeddings is None
    if speaker_embeddings!=None:
      nb_spk=speaker_embeddings.shape[0]
      if nb_spk<hps['nb_speakers']:
        speaker_embeddings=torch.cat((speaker_embeddings,speaker_embeddings[id_new_speaker,:].repeat(hps['nb_speakers']-nb_spk,1)))
        model_dict.update({('speaker_embedding.weight',speaker_embeddings)})
        print('{} speakers added'.format(hps['nb_speakers']-nb_spk))

    car_embeddings=model_dict.get('embedding.weight'); nb_car=car_embeddings.shape[0] #change of number of characters
    if (nb_car <len(hps['symbols'])): # extra characters have been added
      car_embeddings=torch.cat((car_embeddings,car_embeddings[0,:].repeat(len(hps['symbols'])-nb_car,1)))
      model_dict.update({('embedding.weight', car_embeddings)})
      print('{} symbols added: {}'.format(len(hps['symbols'])-nb_car, [hps['symbols'][p] for p in range(nb_car,len(hps['symbols']))]))

    if (nb_car>len(hps['symbols'])): # extra characters should be deleted
      car_embeddings=car_embeddings[0:len(hps['symbols']),:]
      model_dict.update({('embedding.weight', car_embeddings)})
      print('{} symbols deleted: {}'.format(len(hps['symbols'])-nb_car, [hps['symbols'][p] for p in range(nb_car,len(hps['symbols']))]))
      nm='phonetize.linear_layer.weight'
      dim=model_dict.get(nm).shape[0]
      nb=len(hps['out_symbols'])
      if (dim>nb):
          weights=model_dict.get('phonetize.linear_layer.weight'); weights=weights[0:nb,:]; model_dict.update({('phonetize.linear_layer.weight', weights)})
          bias=model_dict.get('phonetize.linear_layer.bias'); bias=bias[0:nb]; model_dict.update({('phonetize.linear_layer.bias', bias)})

    for i_out in range(len(hps['dim_data'])): #change of number of output parameters
      nm='decoder.%d.decoder_rnn.weight_hh'%i_out
      if nm in model_dict.keys():
        dim=model_dict.get(nm).shape[1]
        if (dim!=hps['decoder_rnn_dim'][i_out]):
        	print('decoder_rnn[{}] changed: {} -> {}'.format(i_out,dim,hps['decoder_rnn_dim'][i_out]))
        	ll=['decoder_rnn.weight_ih','decoder_rnn.weight_hh','decoder_rnn.bias_ih','decoder_rnn.bias_hh','linear_projection.linear_layer.weight','gate_layer.linear_layer.weight']
        	ignore_layers+=['decoder.%d.'%i_out+k for k in ll]
      nm='decoder.%d.prenet.layers.0.linear_layer.weight'%i_out
      if nm in model_dict.keys():
       	dim=model_dict.get(nm).shape[0]
        if (dim!=hps['prenet_dim'][i_out]):
        	print('prenet output[{}] changed: {} -> {}'.format(i_out,dim,hps['prenet_dim'][i_out]))
        	ll=['prenet.layers.0.linear_layer.weight','prenet.layers.1.linear_layer.weight','attention_rnn.weight_ih']
        	ignore_layers+=['decoder.%d.'%i_out+k for k in ll]    
      nm='decoder.%d.linear_projection.linear_layer.bias' % i_out
      if nm in model_dict.keys():
        dim=model_dict.get(nm).shape[0]
        if (dim!=hps['dim_data'][i_out]*hps['n_frames_per_step'][i_out]):
          print('prenet input[{}] changed: {} -> {}'.format(i_out,dim,hps['dim_data'][i_out]*hps['n_frames_per_step'][i_out]))
          ll=['decoder.prenet.layers.0.linear_layer.weight','decoder.linear_projection.linear_layer.weight','decoder.linear_projection.linear_layer.bias']
          ignore_layers+=[k.replace('decoder','decoder.%d'%i_out) for k in ll]
#          if hps['use_postnet'][i_out]: # get rid of the postnet for transfer learning
#            hps['use_postnet'][i_out]=False
#            ll=['postnet.convolutions.0.0.conv.weight','postnet.convolutions.4.0.conv.weight','postnet.convolutions.4.0.conv.bias','postnet.convolutions.4.1.weight','postnet.convolutions.4.1.bias','postnet.convolutions.4.1.running_mean','postnet.convolutions.4.1.running_var'] # do not consider free parameters that differ in dim
#            ignore_layers+=[k.replace('postnet','postnet.%d'%i_out) for k in ll]
        
    if ignore_layers is not None:
      mdict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
      for k, v in mdict.items():
        print(k)
      model_dict = mdict

#    nm='tacotron2_MULTI_STYLE_WAVEGLOW_1.27mars2021'
#    print('Load style_encoder.convolutions from {}'.format(nm))
#    checkpoint_dict = torch.load(nm, map_location='cpu') # load a pre-trained convolutional prosodic encoder
#    mdict = checkpoint_dict['state_dict']
#    lst = {k: v for k, v in mdict.items() if k.find('style_encoder.convolutions')>=0}
#    model_dict.update(lst);
    model.load_state_dict(model_dict,strict=False)
    print("Model '{}' loaded".format(nm_mod), flush=True)
    return model

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
  print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
  torch.save({'iteration': iteration, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'learning_rate': learning_rate}, filepath)

def mse_loss_with_nans(input, target):
    # Missing data are inf's
    mask = ~torch.isinf(target)

    out = (input[mask]-target[mask])**2
    loss = out.mean().sqrt()
    return loss

def process(model, is_train, device, loader, optimizer, epoch, by_utt=False, nt=0, T_Perf=np.empty((0,3))):
	dim_data=hps['dim_data']; nb_out=len(dim_data)
	check_gpu("PROCESS"); # check_tensors();
	if is_train:
		model.train(); phase='train'; torch.set_grad_enabled(True)
		torch.backends.cudnn.allow_tf32=False
	else:
		model.eval(); phase='validation'; torch.set_grad_enabled(False); T_Perf = np.empty([0,3])
	print('{} on {:3d} sequences'.format(phase,len(loader.dataset)))
	mean_loss=0.0; mean_err_lg = 0.0; lg_prd=np.empty([0,nb_out]); nm_ficERR_prec=['']*nb_out;
	for i_batch, batch in enumerate(loader):
		lg_batch=len(batch[CSV_num_fic])
		lg=np.zeros([lg_batch,nb_out],dtype=int); err_lg=np.zeros(nb_out)
		if is_train: optimizer.zero_grad()
		(T_in, T_tgt, i_nm) = model.parse_batch(batch)
		if hps['compute_durations']: 
			(spe_out, spe_out_postnet, gate_out, pho_out, dur_out, style, aln_out, emb_out) = model(T_in)
		else:
			(spe_out, spe_out_postnet, gate_out, pho_out, dur_out, style, _, emb_out) = model(T_in)
		(spe_tgt, gate_tgt, pho_tgt, dur_tgt) = T_tgt
		ind_pho_ok = np.nonzero(batch[BATCH_lg_pho_out])[0]
		pho_ok = hps['dim_out_symbols'] and (len(ind_pho_ok))
		dur_ok = hps['compute_durations'] and (len(ind_pho_ok))
		for i_out in range(nb_out):
			if gate_out[i_out]!=None:
#				gate_out_clone = gate_out[i_out].clone()
				ind=np.where(torch.sigmoid(gate_out[i_out]).cpu()>hps['gate_threshold'][i_out]);
				ind_ok=np.unique(ind[0],return_index=True)[1]; lg[ind[0][ind_ok],i_out] = ind[1][ind_ok]*hps['n_frames_per_step'][i_out]
#				for i_b in range(lg_batch): gate_out_clone[i_b,lg[i_b][i_out]:]=torch.clamp(gate_out[i_out][i_b,lg[i_b][i_out]:],pb_min_gate) # 
				err_lg[i_out]=np.mean(abs(lg[:,i_out]-batch[BATCH_lg_tgt][:,i_out]))/hps['fe_data'][i_out]-hps['lgs_sil_add']
#				gate_out[i_out] = gate_out_clone
		if by_utt:
			loss_pho = 0.0; loss_dur = 0.0; loss_spe, loss_gate, loss_spe_postnet, loss_aln = np.zeros(nb_out), np.zeros(nb_out), np.zeros(nb_out), np.zeros(nb_out)
			g_loss_per_utt=np.zeros(lg_batch);
			for i_out in range(nb_out):
				ind_lg_out=np.nonzero(batch[BATCH_lg_tgt][:,i_out])[0]
				if gate_out[i_out]!=None:
					fe=hps['fe_data'][i_out]
					# with mask
					mask = ~get_mask_from_lengths(batch[BATCH_lg_tgt][:,i_out]).cpu().detach().numpy()
					l_spe = nnMSE_nored(spe_out[i_out], spe_tgt[i_out]).cpu().detach().numpy()
					l_spe = l_spe.mean(axis=1)
					l_spe = ma.masked_array(l_spe,mask)
					g_loss_per_utt += l_spe.mean(axis=1)
					loss_spe[i_out] = l_spe.mean()
					for i_ERR in range(lg_batch):
						nm_ficERR='_'+hps['dir_data'][i_out]+'/'+nms_data[batch[BATCH_i_nm][i_ERR][0]]+'_ERR_'+hps['ext_data'][i_out]+'.wav'
						deb=batch[BATCH_i_nm][i_ERR][1]*fe; lge=batch[BATCH_lg_tgt][i_ERR][i_out]; lg_in=batch[BATCH_lg_in][i_ERR]
						if (nm_ficERR!=nm_ficERR_prec[i_out]) & path.exists(nm_ficERR):
							if nm_ficERR_prec[i_out]!='' : fpERR.close()
							fpERR=open(nm_ficERR,'rb+'); feERR=100.0; nm_ficERR_prec[i_out]=nm_ficERR
							debERR=int(deb*feERR/fe); lgERR=int(lge*feERR/fe)
							vERR=np.interp(np.arange(lgERR)/lgERR,np.arange(lge)/lge,l_spe[i_ERR][0:lge])
							fpERR.seek(44+debERR*2); fpERR.write(bytearray(np.int16(100*vERR))); fpERR.flush()
					if hps['use_postnet'][i_out]:
						l_spe = nnMSE_nored(spe_out_postnet[i_out][ind_lg_out,:,:], spe_tgt[i_out][ind_lg_out,:,:]).cpu().detach().numpy()
						l_spe = l_spe.mean(axis=1)
						l_spe = ma.masked_array(l_spe,mask[ind_lg_out,:])
						g_loss_per_utt[ind_lg_out] += l_spe.mean(axis=1).data
						loss_spe_postnet[i_out] = l_spe.mean()
					l_gate = hps['factor_gate'][i_out]*nnBCE_nored(gate_out[i_out][ind_lg_out,:], gate_tgt[i_out][ind_lg_out,::hps['n_frames_per_step'][i_out]]).cpu().detach().numpy()
					l_gate = ma.masked_array(l_gate,mask[ind_lg_out,::hps['n_frames_per_step'][i_out]])
					g_loss_per_utt[ind_lg_out] += l_gate.mean(axis=1).data
					loss_gate[i_out]= l_gate.mean()
					if False: #(i_out>0) and (aln_out[i_out] is not None):
						aln_out0 = torch.nn.functional.interpolate(aln_out[0][:,ind_lg_out,:].permute((1,0,2)),size=aln_out[i_out].shape[2],mode='linear')
						l_aln = 100.0*nnMSE_nored(aln_out[i_out][:,ind_lg_out,:].permute((1,0,2)),aln_out0).cpu().detach().numpy()
#						l_aln = nnKLD_nored(aln_out[i_out].permute((1,0,2)).log(),aln_out0).cpu().detach().numpy()
						l_aln = l_aln.mean(axis=1)
						g_loss_per_utt[ind_lg_out] += l_aln.mean(axis=1).data
						loss_aln[i_out]= l_aln.mean()
			if pho_ok:
				loss_pho=hps['factor_pho']*nnCE_nored(pho_out[ind_pho_ok,:,:], pho_tgt[ind_pho_ok,:]).cpu().detach().numpy().mean(axis=1)
				g_loss_per_utt[ind_pho_ok] += loss_pho
				loss_pho=loss_pho.mean()
#			if dur_ok and (aln_out[0] is not None):
#				ms_from_act = 1000.0*hps['n_frames_per_step'][0]*aln_out[0][:,ind_pho_ok,:].permute((1,0,2)).sum(axis=2)/hps['fe_data'][0]
#				loss_dur=hps['factor_dur']*nnMSE_nored(ms_from_act, dur_tgt[ind_pho_ok,:])/1000.0 # error en s
#				loss_dur=loss_dur.cpu().detach().numpy().mean(axis=1)/np.array(batch[BATCH_lg_pho_out])[ind_pho_ok]
#				g_loss_per_utt[ind_pho_ok] += loss_dur
#				loss_dur=loss_dur.mean()
			Perf=np.concatenate((np.array(i_nm),g_loss_per_utt.reshape((lg_batch,1))),axis=1)
			T_Perf=np.concatenate((T_Perf,Perf))
		else:
			loss, loss_pho, loss_dur = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
			loss_spe, loss_gate, loss_spe_postnet, loss_aln = torch.zeros(nb_out).cuda(), torch.zeros(nb_out).cuda(), torch.zeros(nb_out).cuda(), torch.zeros(nb_out).cuda()
			for i_out in range(nb_out):
				ind_lg_out=np.nonzero(batch[BATCH_lg_tgt][:,i_out])[0]
				if gate_out[i_out]!=None:
					loss_spe[i_out] = nnMSE(spe_out[i_out][ind_lg_out,:,:], spe_tgt[i_out][ind_lg_out,:,:])
					loss_gate[i_out] = hps['factor_gate'][i_out]*nnBCE(gate_out[i_out][ind_lg_out,:], gate_tgt[i_out][ind_lg_out,::hps['n_frames_per_step'][i_out]]) #increase weight of gate loss
					loss += loss_spe[i_out]+loss_gate[i_out]
					if hps['use_postnet'][i_out]:
						loss_spe_postnet[i_out] = nnMSE(spe_out_postnet[i_out][ind_lg_out,:,:], spe_tgt[i_out][ind_lg_out,:,:])
						loss += loss_spe_postnet[i_out]
					else:
						loss_spe_postnet[i_out] = 0.0
				if False: #(i_out>0) and (aln_out[i_out] is not None):
					aln_out0=torch.nn.functional.interpolate(aln_out[0][:,ind_lg_out,:].permute((1,0,2)),size=aln_out[i_out].shape[2],mode='linear') # [lg_batch,lg_in,lg_out]
					loss_aln[i_out] = 100.0*nnMSE(aln_out[i_out][:,ind_lg_out,:].permute((1,0,2)),aln_out0)
#					loss_aln[i_out] =nnKLD(aln_out[i_out].permute((1,0,2)).log(),aln_out0)
					loss += loss_aln[i_out] 

					#~ import matplotlib
					#~ import matplotlib.pyplot as plt
					#~ from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
					#~ matfig=plt.figure(figsize=(15,10))
					#~ for i_b in range(lg_batch):
						#~ aa=plt.subplot(1, 3, 1)
						#~ plt.matshow(aln_out[0][:,i_b,:].detach().cpu().numpy(),origin='lower',aspect='auto',fignum=0,vmin=0,vmax=1)
						#~ plt.draw()
						#~ aa=plt.subplot(1, 3, 2)
						#~ plt.matshow(aln_out0[:,i_b,:].detach().cpu().numpy(),origin='lower',aspect='auto',fignum=0,vmin=0,vmax=1)
						#~ plt.draw()
						#~ aa=plt.subplot(1, 3, 3)
						#~ plt.matshow(aln_out[i_out][:,i_b,:].detach().cpu().numpy(),origin='lower',aspect='auto',fignum=0,vmin=0,vmax=1)
						#~ plt.draw()
						#~ l_org=batch[1][i_b]; ch=''.join([symbols[p] for p in batch[0][i_b][0:l_org]])
						#~ plt.title('{}: {}'.format(i_b,ch),fontsize=9,pad=10)	
						#~ axes = plt.gca()
						#~ axes.yaxis.set_ticks_position('left')
						#~ axes.yaxis.set_major_locator(MultipleLocator(1))
						#~ axes.yaxis.set_major_formatter(FormatStrFormatter("%s"))
						#~ axes.yaxis.set_ticklabels([''] +  [symbols[p].replace('@', ' ') for p in batch[0][i_b][0:l_org]], fontsize=7, rotation=90)

						#~ matfig.show()
						#~ plt.waitforbuttonpress()
					
			if pho_ok:
#			l_org=batch[BATCH_lg_in][i_b]
#			i_b=ind_pho_ok[0]
#			ind=pho_tgt[i_b,:l_org]; ph_tgt='|'.join(["{}".format(out_symbols[p]) for p in ind])
#			ind=pho_out[i_b,:,:l_org].argmax(axis=0); ph_prd='|'.join(["{}".format(out_symbols[p]) for p in ind])
				loss_pho=hps['factor_pho']*nnCE(pho_out[ind_pho_ok,:,:], pho_tgt[ind_pho_ok,:])
				loss += loss_pho
#			if dur_ok and (aln_out[0] is not None): # only on attention of first decoder
#				ms_from_act = 1000.0*hps['n_frames_per_step'][0]*aln_out[0][:,ind_pho_ok,:].permute((1,0,2)).sum(axis=2)/hps['fe_data'][0]
#				loss_dur=hps['factor_dur']*nnMSE(ms_from_act, dur_tgt[ind_pho_ok,:])/1000.0 # error en s
#				loss += loss_dur
		for i_b in range(lg_batch):
			l_org=batch[BATCH_lg_in][i_b]; ch='|'.join([hps['symbols'][p] for p in batch[BATCH_text_in][i_b][0:l_org]])
			style_utt=style[i_b,:].cpu().detach().numpy() if style.nelement() else np.empty(0)
			if by_utt:
			  print('{} [LOC={}] '.format(nms_data[batch[BATCH_i_nm][i_b][0]],batch[BATCH_spk_in][i_b]),end='')
			  if batch[BATCH_lg_tgt].any():
			    print('at {:.2f}+{:.2f}s,'.format(batch[BATCH_i_nm][i_b][1],batch[BATCH_lg_tgt][i_b][0]/hps['fe_data'][0]-hps['lgs_sil_add']),end='')
			    if max(lg[i_b,:])==0 and max(batch[BATCH_lg_tgt][i_b])>0: print('Pb EoS')
			  print('"{}" -> {}'.format(ch,lg[i_b,:]),end='');
			  print(', LOSS={:.3}'.format(g_loss_per_utt[i_b]),end='')
			  if len(style_utt): print(', STYLE={}'.format(style_utt),end='')
			  if pho_ok and batch[BATCH_lg_pho_out][i_b]:
			    ph_tgt='|'.join(["{}".format(hps['out_symbols'][p]) for p in pho_tgt[i_b,:l_org]])
			    ind=pho_out[i_b,:,:l_org].argmax(axis=0)
			    ph_prd='|'.join(["{}".format(hps['out_symbols'][p]) for p in ind]) #predicted aligned phonetic chain
			    ind_pok=torch.where(pho_tgt[i_b,:l_org]!=ind); ch_pok='|'.join(["{}".format(hps['out_symbols'][p]) for p in ind[ind_pok]])
			    print('DIFF: {}'.format(ch_pok))
			    print('ORTHO: {}\nPH_TGT:{}\nPH_PRD:{}'.format(ch,ph_tgt,ph_prd),end='')
#			  if batch[BATCH_lg_tgt][i_b][0]:
#			    print('DUR_ORG: {}\nPH_TGT:{}\nPH_PRD:{}'.format(ch,ph_tgt,ph_prd))
#			  	ms_from_act = 1000.0*hps['n_frames_per_step'][0]*aln_out[0][0:l_org,i_b,:].sum(axis=1).cpu().detach().numpy()/hps['fe_data'][0]
#			  	ms_from_act = np.diff(np.cumsum(ms_from_act).astype(int),prepend=0)
#			  	print('DUR_ACT: {}'.format(' '.join([str(x) for x in ms_from_act])),end='')			  	
			  print('');
		if is_train:
			if torch.isnan(loss): print("Pb Loss")
			if loss.item()<1000.0:
				loss.backward(); optimizer.step()
				loss = loss.cpu().detach().item()
			else:
				loss = 1000.0
			loss_spe_postnet = loss_spe_postnet.cpu().detach().numpy()
			loss_spe = loss_spe.cpu().detach().numpy()
			loss_gate = loss_gate.cpu().detach().numpy()
			loss_aln= loss_aln.cpu().detach().numpy()
			loss_dur = loss_dur.cpu().detach().numpy()
			loss_pho = loss_pho.cpu().detach().numpy()
		else:
			loss = np.mean(Perf[:,2])
		nt += lg_batch
		with np.printoptions(precision=2,suppress=True): print('{} Batch: {:6d}/{:6d} ({:5.2f}%)]\tLoss: {:.3f}={}+{}+{}+{}+{:.2f}+{:.2f}, Err_lg: {}ms'.format(phase, nt, len(loader.dataset), 100.*nt/len(loader.dataset), loss, loss_spe_postnet, loss_spe, loss_gate, loss_aln, loss_pho, loss_dur,  1000.0*err_lg), flush=True)
		mean_loss += loss; mean_err_lg += err_lg; lg_prd=np.concatenate((lg_prd,lg))
	mean_loss /= (i_batch+1); mean_err_lg /= (i_batch+1)
	# nb of predctions that exceed lg_max
	pct_ok = (100.0*sum(lg_prd/hps['fe_data']<hps['lgs_max']))/len(lg_prd) if lg_prd.any() else 0.0
	with np.printoptions(precision=3, suppress=True):
		print('{} Epoch: {}\tMean Loss: {:.3f}, Mean Err_lg: {},{}%'.format(phase, epoch, mean_loss, mean_err_lg, pct_ok), flush=True)
	return(nt, T_Perf)

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_directory', type=str, default='_out', help='directory to save checkpoints')
parser.add_argument('-c', '--pre_trained', type=str, default='', required=False, help='pre-trained model')
parser.add_argument('--hparams', type=str, default='', required=False, help='comma separated name=value pairs')
parser.add_argument('--freeze',  type=str, default='', required=False,  help='freeze units by reg expression')
parser.add_argument('--silent', action='store_true', default=False, help='run silently')
parser.add_argument('--phonetic_only', action='store_true', default=False, help='output only phonetic predictions')
parser.add_argument('--num_gpu', required=False, type=int, default='0', help='number of the gpu')
parser.add_argument('--config', required=False, type=str, default="tc2.yaml", help='configuration file')
parser.add_argument('--model_name', required=False, type=str, default="tacotron2", help='name of the saved model')
parser.add_argument('--id_new_speaker', required=False, type=int, default='0', help='id of the embeddings of the speaker to copy from')
parser.add_argument('--stateful', action='store_true', default=False, help='save state of encoder lstm for chaining utterances')
parser.add_argument('--save_embeddings', type=str, required=False, default='', help='Save ouput embeddings of text-encoder in .mat file')
args = parser.parse_args();
hps = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
hparams = args.hparams;
if hparams: hps.update(yaml.safe_load(hparams))

init_symbols(hps)

silent = args.silent
freeze_units = args.freeze
num_gpu = args.num_gpu
pre_trained = args.pre_trained
phonetic_only = args.phonetic_only
pb_min_gate = torch.tensor(hps['gate_threshold']); pb_min_gate=torch.log(pb_min_gate/(1-pb_min_gate))
model_name = args.model_name
id_new_speaker = args.id_new_speaker
hps['stateful'] = stateful = args.stateful
hps['save_embeddings'] = save_embeddings = args.save_embeddings
hps['code_PAR'] = text_to_sequence(hps,'§')[0]

num_gpu=min(num_gpu,torch.cuda.device_count()-1)
device = torch.device("cuda:%d" % (num_gpu) if torch.cuda.is_available() else "cpu")
print("Device: {}!!!".format(device))
torch.cuda.empty_cache()
check_gpu("START")

model = Tacotron2(hps).to(device)
if path.exists(pre_trained) :
  model = warm_start_model(pre_trained, model, hps['ignore_layers'])
if freeze_units:
  model.freeze(freeze_units)

print('Trainable parameters')
for name, param in model.named_parameters():
    if param.requires_grad: print('{} {}'.format(name,param.data.shape))

check_gpu("AFTER LOADING MODEL")
(data_test, nms_data)=load_csv(hps['nm_csv_test'], hps, utts=[], nms_data=[])
test_loader = torch.utils.data.DataLoader(data_test, batch_size=hps['batch_size'], drop_last=False, shuffle=False, collate_fn=collate_batch, num_workers=0)
if hps['nb_epochs']:
  if stateful:
  	(data_train, nms_data)=load_csv(hps['nm_csv_train'], hps, utts=[], nms_data=nms_data, sort_utt=False)
  	sampler_train = BatchSampler(data_train, hps['batch_size'], drop_last=False);
  else:
  	(data_train, nms_data)=load_csv(hps['nm_csv_train'], hps, utts=[], nms_data=nms_data)
  	sampler_train = OrderedSampler(data_train, hps['batch_size'], drop_last=False);
  train_loader = torch.utils.data.DataLoader(data_train, batch_sampler=sampler_train, collate_fn=collate_batch, num_workers=0)
  optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=hps['learning_rate'], weight_decay=hps['weight_decay'])
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=hps['milestones'], last_epoch=-1, gamma=hps['gamma'])
  nb_training_iter=0
  for i_epoch in range (hps['nb_epochs']):
    #training
    (nt, _) = process(model, True, device, train_loader, optimizer, i_epoch)
    nb_training_iter += nt
    nm_mod = path.join(args.output_directory, "{}_{}_{}".format(model_name,'_'.join(hps['ext_data']),i_epoch))
    save_checkpoint(model, optimizer, hps['learning_rate'], nb_training_iter, nm_mod)
    scheduler.step()
  # performance by utt
  print('Performance by file (training)')
  (nt_train, T_Perf_train) = process(model, False, device, train_loader, [], 0, by_utt=True)
else:
  T_Perf_train=[]
if data_test!=[]:
  if phonetic_only: hps['dim_data']=[]; model.set_dim_data([]) # only phonetic prediction
  (nt_test, T_Perf) = process(model, False, device, test_loader, [], 0, by_utt=True)
  if T_Perf_train:
    T_Perf=np.concatenate((T_Perf,T_Perf_train))
else :
  T_Perf=T_Perf_train
T_Perf_by_file=[];
for i_nms in range(len(nms_data)):
  ind=np.where(T_Perf[:,0]==i_nms)[0]
  #sort by Loss
  ind=ind[T_Perf[ind,:].argsort(axis=0)[:,2]]; ind=ind[::-1]
  print('{}:'.format(nms_data[i_nms]), end='')
  for i in range(min(len(ind),10)):
    print(' {:.1f}:{:.2f}'.format(T_Perf[ind[i],1],T_Perf[ind[i],2]),end='')
  print('',flush=True)
  T_Perf_by_file.append(np.mean(T_Perf[ind,2]))
ind=np.argsort(T_Perf_by_file)
print('Mean performance')
for i_nms in ind:
  print('{}: {:.2f}'.format(nms_data[i_nms],T_Perf_by_file[i_nms]))
if save_embeddings:
  if lg_embeddings: f_embeddings.close()
  input_embeddings = getattr(model,'embedding').weight.cpu().detach().numpy()
  speaker_embeddings = getattr(model,'speaker_embedding').weight.cpu().detach().numpy()
  scipy.io.savemat('EMB_'+nm_tacotron2+'.mat', mdict={'speaker_embeddings': speaker_embeddings, 'symbol_embeddings': input_embeddings, 'speakers': hps['speakers'], 'symbols':  hps['symbols']})

