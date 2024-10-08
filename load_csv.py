import sys
import numpy as np
import regex as re
from os import path, sys
#from text import text_to_sequence, _out_symbol_to_id, nb_voyelles_sequence
#from text.symbols import symbols, out_symbols
from def_symbols import text_to_sequence
global _symbol_to_id, _id_to_symbol, symbols, out_symbols, _out_symbol_to_id

def load_csv(nm_csv, hps, utts=[], nms_data=[], split="|", sort_utt=True, check_org=True):
    print('LOAD_CSV: {}'.format(nm_csv))
    i_space=hps['symbols'].index(' ')
    nb_out = len(hps['dim_data'])
    if path.isfile(nm_csv):
        f = open(nm_csv, encoding='utf-8')
        nm_prec=''; nb_s=0;
        for line in f:
            if line.startswith('%'): continue
            fields=line.strip().split(split)
            nb_fields=len(fields)
            nm=fields[0]; # first field is filename
            i_nms=-1
            if nm_prec!=nm:
            	lgs_data=np.zeros(nb_out)
            	for i_out in range(nb_out):
                    nm_data='_'+hps['dir_data'][i_out]+'/'+nm+'.'+hps['ext_data'][i_out]
                    if path.exists(nm_data): # read file header to get dimensions
                        (lg_data,dim,num,den) = tuple(np.fromfile(nm_data, count=4, dtype=np.int32)); fe=num/den; 
                        if (dim!=hps['dim_data'][i_out]) or (fe!=hps['fe_data'][i_out]):
                            print('{}> Error file({}): {}!={}, {:.2f}Hz!={:.2f}Hz'.format(len(utts),nm_data,dim,hps['dim_data'][i_out],fe,hps['fe_data'][i_out]), flush=True);
                        else: lgs_data[i_out]=lg_data/fe
            	nm_prec=nm
            if fields[1].isnumeric(): # next two fields are: start, end
                debs=int(fields[1])/1000.0
                lgs=int(fields[2])/1000.0+hps['lgs_sil_add']-debs
                i_txt=3
            else:
                debs=0.0; lgs=0.0; i_txt=1
            mn_lgs_data=lgs_data[lgs_data>0];
            mn_lgs_data=min(mn_lgs_data) if mn_lgs_data.any() else 0.0
            if check_org and (mn_lgs_data>0) and ((debs+lgs>mn_lgs_data) or (lgs>hps['lgs_max'])):
                print('{}> Error data({},{}): {:.2f}s-{:.2f}s, {}'.format(len(utts),nm,lgs_data,debs,lgs,nb_fields), flush=True)
            else:
                i_spk=[i for i, elem in enumerate(hps['speakers']) if '_'+elem in nm]; # speaker in filename
                i_spk=i_spk[0] if len(i_spk) else 0
                i_style=[i for i, elem in enumerate(hps['styles']) if '_'+elem in nm]; # style in filename
                i_style=i_style[0] if len(i_style) else 0
                txt=fields[i_txt]; l_tags=re.findall(r'\<([^\>]*)\>\s*',txt); # tags in text
                if l_tags:
                    txt=re.sub(r'\<([^\>]*)\>\s*','',fields[i_txt]);  #remove tags
                    for tags in l_tags:
                        lt=re.findall("(\w+)=([^;]+)",tags);
                        for t in lt:
                            if t[0]=='SPK' and t[1] in hps['speakers']: i_spk=hps['speakers'].index(t[1])
                            if t[0]=='STYLE' and t[1] in hps['styles']: i_style=hps['styles'].index(t[1])                  
                txt=re.sub(r'’','\'',txt);
                txt=re.sub(r'œ','oe',txt);
                text_norm = np.array(text_to_sequence(txt)).astype(np.int16)
                # ch='|'.join([hps['symbols'][p] for p in text_norm]); print(ch);

                if nb_fields>i_txt+1: # phonetic alignments of input symbols
                    out_symbols=np.array([hps['_out_symbol_to_id'].get(s,-1) for s in fields[i_txt+1].split()]).astype(np.int16)
                    if (-1 in out_symbols):
                    	print(line); print(out_symbols);
                    	print(fields[i_txt+1],out_symbols); exit(0);
                    if len(out_symbols)!=len(text_norm):
                        print('{}> Error file({}): lg_pho[{}]!=lg_txt[{}]\n{}'.format(len(utts),nm,len(out_symbols),len(text_norm),txt), flush=True);
                else:
                    out_symbols=[] 
                if nb_fields>i_txt+2: # durations
                    out_dur=[float(v) for v in fields[i_txt+2].split()] if nb_fields>i_txt+2 else [] # durations of input symbols in ms
                    #ind=np.max(np.nonzero(out_dur)); out_dur[ind]+= 1000*hps['lgs_sil_add'] # add hps['lgs_sil_add'] to the last no-silent symbol
                else:
                    out_dur=[];
                try:
                    i_nms=nms_data.index(nm)
                except ValueError as e:
                    print('new datafile: {} {:.2f}s'.format(nm,mn_lgs_data)); i_nms=len(nms_data); nms_data.append(nm);
                if i_nms>=0:
                    #print('{}: {}'.format(i_nms,''.join([symbols[p].replace('@', ' ') for p in text_norm])));
                    fuse=False; i_utts=len(utts)-1
                    #~ if i_utts>1:
                    	#~ if len(out_dur): # phonetic input
                    		#~ while i_utts>0 and len(utts[i_utts][7])==0: i_utts+=-1
                    	#~ else:
                    		#~ while i_utts>0 and len(utts[i_utts][7])>0: i_utts+=-1
                    	#~ fuse=(i_utts>=0 and debs+lgs<utts[i_utts][1]+hps['lgs_max'] and i_nms==utts[i_utts][0] and symbols[utts[i_utts][3][-1]]!='§') # fuse
                    if fuse:
                        nb_s_new=debs+lgs-utts[i_utts][1]
                        nb_s+=nb_s_new-utts[i_utts][2]
                        utts[i_utts][2]= nb_s_new # update lgs
                        if utts[i_utts][3][-1]==text_norm[0]: text_norm[0]=i_space
                        utts[i_utts][3].extend(text_norm)
                        utts[i_utts][4]+=len(text_norm)
                        if len(out_dur):
                            utts[i_utts][6].extend(out_symbols);
                            utts[i_utts][7].extend(out_dur) # pb silence
                    else:
                    	utts.append([i_nms, debs, lgs, text_norm, len(text_norm), i_spk, i_style, out_symbols, out_dur]); nb_s+=lgs;
        nb_utts=len(utts); nb_nms=len(nms_data);
        nb_h=int(nb_s/3600); nb_s=nb_s-nb_h*3600; nb_mn=int(nb_s/60); nb_s=nb_s-nb_mn*60;
        print('{}: {} utts - {} datafiles: {}:{}:{:.2f}'.format(nm_csv,nb_utts,nb_nms,nb_h,nb_mn,nb_s),flush=True)
        if sort_utt:
            # sort by increasing length
            def takeLen_out(elem):
                return elem[2]
            # utterances finally sorted by length
            utts.sort(key=takeLen_out,reverse = False)
#       for i_utt, fields in enumerate(utts):
#            print('{}: {} -> {} at {}'.format(nms_data[fields[0]],len(fields[3]),fields[2],fields[1]))
        f.close()
    else:
        print("File {} not accessible".format(nm_csv))

    return(utts, nms_data)
