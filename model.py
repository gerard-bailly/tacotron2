import numpy as np
import re
from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import pdb

global hps

def get_mask_from_lengths(lengths, n_frames_per_step = 1):
  max_lengths=max(lengths); nb_lengths=len(lengths)
  ids=np.arange(0, max_lengths)
  mask=(ids<np.reshape(lengths,(nb_lengths,1)))
  mask=torch.from_numpy(mask).cuda()
  return mask

def to_gpu(x):
  x=x.contiguous()
  if torch.cuda.is_available():
    x=x.cuda(non_blocking=True)
  return torch.autograd.Variable(x)

class LinearNorm(nn.Module):
  def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
    super(LinearNorm, self).__init__()
    self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
    torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

  def forward(self, x):
    return self.linear_layer(x)

class ConvNorm(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
    padding=None, dilation=1, bias=True, w_init_gain='linear'):
    super(ConvNorm, self).__init__()
    if padding is None:
      assert(kernel_size % 2 == 1)
      padding = int(dilation * (kernel_size - 1) / 2)
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

  def forward(self, signal):
    conv_signal = self.conv(signal)
    return conv_signal

class ConvNorm2D(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm2D, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

  def forward(self, signal):
    conv_signal = self.conv(signal)
    return conv_signal


class LocationLayer(nn.Module):
  def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
    super(LocationLayer, self).__init__()
    padding = int((attention_kernel_size - 1) / 2)
    self.location_conv = ConvNorm(2, attention_n_filters, kernel_size=attention_kernel_size, padding=padding, bias=False, stride=1, dilation=1)
    self.location_dense = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain='tanh')

  def forward(self, attention_weights_cat):
    processed_attention = self.location_conv(attention_weights_cat)
    processed_attention = processed_attention.transpose(1, 2)
    processed_attention = self.location_dense(processed_attention)
    return processed_attention


class Attention(nn.Module):
  def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, attention_location_n_filters, attention_location_kernel_size):
    super(Attention, self).__init__()
    self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
    self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
    self.v = LinearNorm(attention_dim, 1, bias=False)
    self.location_layer = LocationLayer(attention_location_n_filters, attention_location_kernel_size, attention_dim)
    self.score_mask_value = -float("inf")

  def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
    """
    PARAMS
    ------
    query: decoder output (batch, dim_data * n_frames_per_step)
    processed_memory: processed encoder outputs (B, T_in, attention_dim)
    attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
    RETURNSencoder_outputs
    -------
    alignment (batch, max_time)
    """
    processed_query = self.query_layer(query.unsqueeze(1))
    processed_attention_weights = self.location_layer(attention_weights_cat)
    energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))

    energies = energies.squeeze(-1)
    return energies

  def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat, mask):
    """
    PARAMS
    ------
    attention_hidden_state: attention rnn last output
    memory: encoder outputs
    processed_memory: processed encoder outputs
    attention_weights_cat: previous and cummulative attention weights
    mask: binary mask for padded data
    """
    alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat) # (B, 1024), (B, T_in, 128), (B, 2, T_in) -> (B, T_in)

    if mask is not None:
      alignment.data.masked_fill_(mask, self.score_mask_value)

    attention_weights = F.softmax(alignment, dim=1) # (B, T_in) -> (B, T_in)
    attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # (B, 1, T_in), (B, T_in, 512) -> (B, 1, 512)
    attention_context = attention_context.squeeze(1) # (B, 1, 512) -> (B, 512)

    return attention_context, attention_weights


class Prenet(nn.Module):
  def __init__(self, in_dim, sizes, p_prenet_dropout):
    super(Prenet, self).__init__()
    self.p_prenet_dropout = p_prenet_dropout
    in_sizes = [in_dim] + sizes[:-1]
    self.layers = nn.ModuleList([LinearNorm(in_size, out_size, bias=False)
    for (in_size, out_size) in zip(in_sizes, sizes)])

  def forward(self, x):
    for linear in self.layers:
      x = F.dropout(F.relu(linear(x)), p=self.p_prenet_dropout, training=True)
    return x


class Postnet(nn.Module):
  """Postnet
    - Five 1-d convolution with 512 channels and kernel size 5
  """

  def __init__(self, hps, i_decoder):
    super(Postnet, self).__init__()
    self.convolutions = nn.ModuleList()
    self.p_postnet_dropout = hps['p_postnet_dropout'][i_decoder]

    self.convolutions.append(
      nn.Sequential(
        ConvNorm(hps['dim_data'][i_decoder], hps['postnet_embedding_dim'][i_decoder], kernel_size=hps['postnet_kernel_size'][i_decoder], stride=1, padding=int((hps['postnet_kernel_size'][i_decoder] - 1) / 2), dilation=1, w_init_gain='tanh'),
        nn.BatchNorm1d(hps['postnet_embedding_dim'][i_decoder]))
    )

    for i in range(1, hps['postnet_n_convolutions'][i_decoder] - 1):
      self.convolutions.append(
      nn.Sequential(
        ConvNorm(hps['postnet_embedding_dim'][i_decoder], hps['postnet_embedding_dim'][i_decoder], kernel_size=hps['postnet_kernel_size'][i_decoder], stride=1, padding=int((hps['postnet_kernel_size'][i_decoder] - 1) / 2), dilation=1, w_init_gain='tanh'),
        nn.BatchNorm1d(hps['postnet_embedding_dim'][i_decoder]))
      )

    self.convolutions.append(
      nn.Sequential(
        ConvNorm(hps['postnet_embedding_dim'][i_decoder], hps['dim_data'][i_decoder], kernel_size=hps['postnet_kernel_size'][i_decoder], stride=1, padding=int((hps['postnet_kernel_size'][i_decoder] - 1) / 2), dilation=1, w_init_gain='linear'),
        nn.BatchNorm1d(hps['dim_data'][i_decoder])
      )
    )

  def forward(self, x):
    for i in range(len(self.convolutions) - 1):
      x = F.dropout(torch.tanh(self.convolutions[i](x)), self.p_postnet_dropout, self.training)
    x = F.dropout(self.convolutions[-1](x), self.p_postnet_dropout, self.training)
    return x

class Encoder(nn.Module):
  def __init__(self, hps):
    """Encoder module:
    - Three 1-d convolution banks
    - Bidirectional LSTM
    """
    super(Encoder, self).__init__()
    self.p_encoder_dropout = hps['p_encoder_dropout']
		
    convolutions = []
    for _ in range(hps['encoder_n_convolutions']):
      conv_layer = nn.Sequential(
        ConvNorm(hps['encoder_embedding_dim'], hps['encoder_embedding_dim'], kernel_size=hps['encoder_kernel_size'], stride=1, padding=int((hps['encoder_kernel_size'] - 1) / 2), dilation=1, w_init_gain='relu'),
        nn.BatchNorm1d(hps['encoder_embedding_dim'])
      )
      convolutions.append(conv_layer)
    self.convolutions = nn.ModuleList(convolutions)
    self.encoder_lstm_hidden_dim = hps['encoder_lstm_hidden_dim']
    self.encoder_embedding_dim = hps['encoder_embedding_dim']
    self.lstm = nn.LSTM(hps['encoder_embedding_dim'], self.encoder_lstm_hidden_dim, 1, batch_first=True, bidirectional=True)

  def forward(self, car_embeddings, input_lengths, lstm_state_ini): # car_embeddings[batchsize,dim_embedding,lg]
    batch_size=car_embeddings.shape[0]
    for conv in self.convolutions:
      car_embeddings = F.dropout(F.relu(conv(car_embeddings)), self.p_encoder_dropout, self.training) # car_embeddings[batchsize,nb_filters,lg]
    car_embeddings = car_embeddings.transpose(1, 2) # car_embeddings[batchsize,lg,nb_filters]
    car_embeddings = nn.utils.rnn.pack_padded_sequence(car_embeddings, input_lengths, batch_first=True)
    self.lstm.flatten_parameters()
#        h0 = torch.zeros(2, batch_size, self.lstm_hidden_dim); h0 = to_gpu(h0).float()
#        c0 = torch.zeros(2, batch_size, self.lstm_hidden_dim); c0 = to_gpu(c0).float()
#        lstm_state_ini = (h0,c0)
#        lstm_state_ini = to_gpu(lstm_state_ini).float()
    outputs, _ = self.lstm(car_embeddings, lstm_state_ini)
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
    return outputs

  def inference(self, car_embeddings, lstm_state_ini):
    for conv in self.convolutions:
      car_embeddings = F.relu(conv(car_embeddings))
    car_embeddings = car_embeddings.transpose(1, 2)
    self.lstm.flatten_parameters()
#        h0 = torch.zeros(2, 1, self.lstm_hidden_dim); h0 = to_gpu(h0).float()
#        c0 = torch.zeros(2, 1, self.lstm_hidden_dim); c0 = to_gpu(c0).float()
#        lstm_state_ini = (h0,c0)encoder_outputs
    outputs, _ = self.lstm(car_embeddings, lstm_state_ini)
    return outputs

class Style_encoder(nn.Module):
  """Style Encoder module (from https://github.com/Yeongtae/tacotron2):
    - Bidirectional LSTM
  """
  def __init__(self, hps):
    super(Style_encoder, self).__init__()

    self.dim_data = hps['dim_data']
    self.prosody_embedding_dim = hps['prosody_embedding_dim']
    self.batch_size = hps['batch_size']

    self.lstm_dim = hps['style_embedding_dim']
    self.nb_style_parameters = hps['nb_style_parameters']
    self.use_style_input = hps['use_style_input']
    self.bias_encoder_with_hc = hps['bias_encoder_with_hc']
    self.bias_encoder_with_entry = hps['bias_encoder_with_entry']
    self.encoder_embedding_dim = hps['encoder_embedding_dim']
    self.encoder_lstm_hidden_dim = hps['encoder_lstm_hidden_dim']

    convolutions = []
    n_out=hps['dim_data']
    for i in range(hps['prosody_n_convolutions']):
      n_pad=1
      conv_layer = nn.Sequential(
       ConvNorm2D(hps['prosody_conv_dim_in'][i],
        hps['prosody_conv_dim_out'][i],
        kernel_size=hps['prosody_conv_kernel'], stride=hps['prosody_conv_stride'][i],
        padding=n_pad,
        dilation=1, w_init_gain='relu'),
        nn.BatchNorm2d(hps['prosody_conv_dim_out'][i])
      )
#      print('{}: kernel={} stride={} padding={}\n'.format(i,hps['prosody_conv_kernel'],hps['prosody_conv_stride'][i],n_pad));
      convolutions.append(conv_layer)
      n_out = (n_out - hps['prosody_conv_kernel'] + 2*n_pad) // hps['prosody_conv_stride'][i] + 1

    self.convolutions = nn.ModuleList(convolutions)
    self.lstm = nn.LSTM(hps['prosody_conv_dim_out'][-1]*n_out, self.lstm_dim, 1, batch_first=True, bidirectional=True)
    # map final values of bi-LSTM style enscoder hidden & contexts
    self.linear_projection = nn.Linear(4*self.lstm_dim, self.nb_style_parameters, bias=True) # project on low_dimension space
    if self.bias_encoder_with_hc :
      self.to_c_encoder = nn.Linear(self.nb_style_parameters, 2*self.out_dim, bias=True) # map to initial values of bi-LSTM encoder context
      self.to_h_encoder = nn.Linear(self.nb_style_parameters, 2*self.out_dim, bias=True) # map to initial values of bi-LSTM encoder hidden
    if self.bias_encoder_with_entry :
      self.to_entry = nn.Linear(self.nb_style_parameters, self.encoder_embedding_dim, bias=False)
      val = 1e-3; self.to_entry.weight.data.uniform_(-val, val)

  def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
    for i in range(n_convs):
      L = (L - kernel_size + 2 * pad) // stride + 1
    return L
    
  def forward(self, spe, spe_lg): # spe[batchsize,dim_data,lg_max]
    batch_size=spe.shape[0]
    spe = spe.contiguous().view(batch_size,1,-1,self.dim_data) # spe[batch_size,1,lg_max,dim_data]
    for conv in self.convolutions:
      spe = F.relu(conv(spe)) # spe[batchsize,dim_embedding,lg_max/2^6,dim_data/2^6]
#      print('{}\n'.format(spe.shape))
    spe = spe.transpose(1, 2)
    T = spe.size(1)
    spe = spe.contiguous().view(batch_size,T,-1)
    _ , (hn,cn) = self.lstm(spe)
#    print(' {}'.format(hn[0,:,0]))
    hc=torch.cat((hn, cn),2).transpose(1,0).reshape((batch_size,-1))
#    print(' {}'.format(hc[:,0]))
    style=torch.tanh(self.linear_projection(hc)) # hc[batchsize,4*lstm_dim]
    if self.bias_encoder_with_hc :
      c0=self.to_c_encoder(style); c0=to_gpu(c0.reshape((batch_size,2,self.eencoder_lstm_hidden_dim)).transpose(0,1))
      h0=self.to_h_encoder(style); h0=to_gpu(h0.reshape((batch_size,2,self.encoder_lstm_hidden_dim)).transpose(0,1))
    if self.bias_encoder_with_entry :
      c0=self.to_entry(style); c0=c0.reshape((batch_size,1,self.encoder_embedding_dim)) # c0[batch_size,1,encoder_embedding_dim]
      h0=to_gpu(torch.zeros(2, batch_size, self.encoder_lstm_hidden_dim)) #h0[2,1,encoder_lstm_hidden_dim]
    return (c0,h0), style

  def forward_with_imposed_style(self, style):
    batch_size=style.shape[0]
    if self.bias_encoder_with_hc :
      c0=self.to_c_encoder(style); c0=to_gpu(c0.reshape((batch_size,2,self.eencoder_lstm_hidden_dim)).transpose(0,1))
      h0=self.to_h_encoder(style); h0=to_gpu(h0.reshape((batch_size,2,self.encoder_lstm_hidden_dim)).transpose(0,1))
    if self.bias_encoder_with_entry :
      c0=self.to_entry(style); c0=c0.reshape((batch_size,1,self.encoder_embedding_dim)) # c0[batch_size,1,encoder_embedding_dim]
      h0=to_gpu(torch.zeros(2, batch_size, self.encoder_lstm_hidden_dim)) #h0[2,1,encoder_lstm_hidden_dim]
    return (c0,h0)

  def inference(self, x):
    x = x.transpose(1, 2)
    _ , (hn,cn) = self.lstm(x)
    hc=torch.cat((hn, cn),2).transpose(1,0).reshape((batch_size,-1))
    style=torch.tanh(self.linear_projection(hc))
    if self.bias_encoder_with_hc :
      c0=to_gpu(self.to_c_encoder(style))
      h0=to_gpu(self.to_h_encoder(style))
    if self.bias_encoder_with_entry :
      c0=to_gpu(self.to_entry(style))
      h0=to_gpu(torch.zeros(2, 1, self.encoder_lstm_hidden_dim))
    return (c0,h0), style

  def inference_with_imposed_style(self, style):
    if self.bias_encoder_with_hc :
      c0=to_gpu(self.to_c_encoder(style)); c0=to_gpu(c0.reshape((1,2,self.out_dim)).transpose(0,1))
      h0=to_gpu(self.to_h_encoder(style)); h0=to_gpu(h0.reshape((1,2,self.out_dim)).transpose(0,1))
    if self.bias_encoder_with_entry :
      c0=to_gpu(self.to_entry(style))
      h0=to_gpu(torch.zeros(2, 1, self.encoder_lstm_hidden_dim))
    return (c0,h0)

class Decoder(nn.Module):
    def __init__(self, hps, i_decoder):
        super(Decoder, self).__init__()
        self.dim_data = hps['dim_data'][i_decoder]
        self.fe_data = hps['fe_data'][i_decoder]
        self.n_frames_per_step = hps['n_frames_per_step'][i_decoder]
        self.encoder_embedding_dim = hps['encoder_embedding_dim']
        self.attention_rnn_dim = hps['attention_rnn_dim'][i_decoder]
        self.attention_dim = hps['attention_dim'][i_decoder]
        self.attention_location_n_filters = hps['attention_location_n_filters'][i_decoder]
        self.attention_location_kernel_size = hps['attention_location_kernel_size'][i_decoder]
        self.decoder_rnn_dim = hps['decoder_rnn_dim'][i_decoder]
        self.prenet_dim = hps['prenet_dim'][i_decoder]
        self.p_prenet_dropout = hps['p_prenet_dropout'][i_decoder]
        self.lgs_max = hps['lgs_max']
        self.gate_threshold = hps['gate_threshold'][i_decoder]
        self.p_attention_dropout = hps['p_attention_dropout'][i_decoder]
        self.p_decoder_dropout = hps['p_decoder_dropout'][i_decoder]
        self.p_teacher_forcing = hps['p_teacher_forcing'][i_decoder]
        self.output_alignments = hps['output_alignments']

        self.prenet = Prenet(
            self.dim_data * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim],
            self.p_prenet_dropout)

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim,
            self.attention_rnn_dim)
        self.attention_layer = Attention(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            self.attention_dim, self.attention_location_n_filters,
            self.attention_location_kernel_size)
        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_embedding_dim,
            self.decoder_rnn_dim, 1)
        self.linear_projection = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.dim_data * self.n_frames_per_step)
        self.gate_layer = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')
            


    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, self.dim_data * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim).zero_())
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, dim_data, T_out) -> (B, T_out, dim_data)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.reshape(decoder_inputs.size(0),int(decoder_inputs.size(1)/self.n_frames_per_step), self.dim_data*self.n_frames_per_step)
        # (B, T_out, dim_data) -> (T_out, B, dim_data)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        if torch.is_tensor(alignments):
        	if alignments.nelement(): alignments = alignments.transpose(0, 2) # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1) # (T_out, B) -> (B, T_out)
        mel_outputs = mel_outputs.transpose(0, 1)
        mel_outputs = mel_outputs.reshape(mel_outputs.size(0),mel_outputs.size(1)*self.n_frames_per_step, self.dim_data)
        mel_outputs = mel_outputs.transpose(1,2) # (B, T_out, dim_data) -> (B, dim_data, T_out)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1) # (B, 128) + (B, 512) -> (B, 640)
        self.attention_hidden, self.attention_cell = self.attention_rnn( # -> (B, 1024), (B, 1024)
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        attention_weights_cat = torch.cat( # -> (B, 2, T_in)
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer( # -> (B, 512), (B, T_in)
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)
 #       if alignment: self.attention_weights=alignment
        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1) # (B, 1024), (B, 512) -> (B, 1536) 
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell)) # -> (B, 1024), (B, 1024) 
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)
        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)  # (B, 1024), (B, 512) -> (B, 1536)
        decoder_output = self.linear_projection(decoder_hidden_attention_context) # (B, 1536) -> (B, dim_out)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context).squeeze(1) # (B, 1536) -> (B)
        return decoder_output, gate_prediction, self.attention_weights # (B, dim_out), (B), (B, T_in),

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        
        memory: Encoder outputs [batchsize, lg_max_in, 512]
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs [batchsize, dim_spec, max_lg_out]
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        
#        print('BEFORE: Memory {:.2f}Mo'.format(torch.cuda.memory_allocated()/1024/1024))
        lg_max_in = memory.shape[1]
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs) # [max_lg_out, batchsize, dim_spec]
        [lg_out,batch_size,dim_out]=decoder_inputs.shape
        out_prev=to_gpu(torch.zeros(1,batch_size,dim_out)) # previous output set to zero!!!
        decoder_inputs = self.prenet(torch.cat((out_prev,decoder_inputs))) # [max_lg_out+1, batchsize, 256]
        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths))
        mel_outputs, gate_outputs = to_gpu(torch.zeros((lg_out,batch_size,dim_out),requires_grad=False)), to_gpu(torch.zeros((lg_out,batch_size),requires_grad=False))
        alignments = to_gpu(torch.empty((0,batch_size,lg_max_in))) if self.output_alignments else []
        for i_frame in range(lg_out):
#         decoder_input = self.p_teacher_forcing*decoder_inputs[i_frame,:,:] + (1-self.p_teacher_forcing)*self.prenet(melp) if i_frame else decoder_inputs[i_frame,:,:];
          decoder_input = decoder_inputs[i_frame,:,:]
          mel_outputs[i_frame,:,:], gate_outputs[i_frame,:], aln = self.decode(decoder_input)
          if self.output_alignments: alignments = torch.cat((alignments,aln))
#         gate_outputs[-1,:]=1000.0 # impose end of sequence at the end of the target
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
#        print('AFTER: Memory {:.2f}Mo'.format(torch.cuda.memory_allocated()/1024/1024))
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, duree_imposee = 0):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decodself.style_parameters)er
        alignments: sequence of attention weights from the decoder
        """
        lg_max_in = memory.shape[1]
        decoder_input = torch.zeros(1,self.dim_data*self.n_frames_per_step)
        decoder_input = to_gpu(decoder_input)
        self.initialize_decoder_states(memory, mask=None)
        mel_outputs, gate_outputs = to_gpu(torch.empty((0,self.dim_data*self.n_frames_per_step),requires_grad=False)), to_gpu(torch.empty((0),requires_grad=False)),
        if self.output_alignments: alignments = to_gpu(torch.empty((0,lg_max_in)))
        nb_mx=int(self.lgs_max*self.fe_data) if duree_imposee==0 else int(duree_imposee*self.fe_data/self.n_frames_per_step)
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)
            mel_outputs = torch.cat((mel_outputs,mel_output))
            gate_outputs = torch.cat((gate_outputs,gate_output))
            if self.output_alignments: alignments = torch.cat((alignments,alignment))
            if (torch.sigmoid(gate_output.data) > self.gate_threshold) and (duree_imposee==0):
                break
            elif len(mel_outputs) == nb_mx:
                print("Warning! Reached max decoder steps")
                break
            decoder_input = mel_output
        if self.output_alignments: 
        	mel_outputs, gate_outputs,  alignments = self.parse_decoder_outputs(mel_outputs[:,None,:], gate_outputs[:,None], alignments[:,:,None])
        else:
        	mel_outputs, gate_outputs,  alignments = self.parse_decoder_outputs(mel_outputs[:,None,:], gate_outputs[:,None], [])
        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
  def __init__(self, hps):
    super(Tacotron2, self).__init__()
    self.mask_padding = hps['mask_padding']
    self.fp16_run = hps['fp16_run']
    self.dim_data = hps['dim_data']
    self.fe_data = hps['fe_data']
    self.encoder_embedding_dim = hps['encoder_embedding_dim']
    self.prenet_dim = hps['prenet_dim']
    self.use_postnet = hps['use_postnet']
    self.postnet_kernel_size = hps['postnet_kernel_size']
    self.postnet_embedding_dim = hps['postnet_embedding_dim']
    self.decoder_rnn_dim = hps['decoder_rnn_dim']
    self.n_frames_per_step = hps['n_frames_per_step']
    self.embedding = nn.Embedding(hps['n_symbols'], hps['symbols_embedding_dim'])
    self.dim_out_symbols = hps['dim_out_symbols']
    self.compute_durations = hps['compute_durations']
    self.output_alignments = hps['output_alignments']
    self.save_embeddings = hps['save_embeddings']

    std = sqrt(2.0 / (hps['n_symbols'] + hps['symbols_embedding_dim']))
    val = sqrt(3.0) * std  # uniform bounds for std
    self.embedding.weight.data.uniform_(-val, val)
    self.nb_speakers=hps['nb_speakers']
    if self.nb_speakers>1: #speaker embeddings directly added to the ouput of the text encoder
      self.speaker_embedding = nn.Embedding(self.nb_speakers, self.encoder_embedding_dim)
      val = 1e-3; self.speaker_embedding.weight.data.uniform_(-val, val)
    self.nb_styles=hps['nb_styles']
    if self.nb_styles>1: #style embeddings directly added to the ouput of the text encoder
      self.style_embedding = nn.Embedding(self.nb_styles, self.encoder_embedding_dim)
      val = 1e-3; self.style_embedding.weight.data.uniform_(-val, val)
    self.nb_style_parameters=hps['nb_style_parameters']
    self.style_parameters=hps['style_parameters']
    self.use_style_input = hps['use_style_input']
    if self.nb_style_parameters: self.style_encoder = Style_encoder(hps)
    self.encoder = Encoder(hps)
    self.decoder=[]
    self.postnet=[]
    for i_out in range(len(self.dim_data)):
      self.decoder.append(Decoder(hps,i_out))
      p = Postnet(hps,i_out) if self.use_postnet[i_out] else None
      self.postnet.append(p)
    self.decoder=nn.ModuleList(self.decoder)
    self.postnet=nn.ModuleList(self.postnet)
    if self.dim_out_symbols: self.phonetize = LinearNorm(self.encoder_embedding_dim, self.dim_out_symbols)
    print('embedding:{} params'.format(sum([np.prod(p.size()) for p in self.embedding.parameters()])));
    print('encoder:{} params'.format(sum([np.prod(p.size()) for p in self.encoder.parameters()])));
    print('decoder:{} params'.format(sum([np.prod(p.size()) for p in self.decoder.parameters()])));
#    if self.compute_durations: self.compute_durations = LinearNorm(self.encoder_embedding_dim, 1)

	
  def set_dim_data(self, dim_data):
    self.dim_data = dim_data
    
  def set_output_alignments(self, output_alignments):
    self.output_alignments = output_alignments
    
  def freeze(self, pat):
    print('Freeze: {}'.format(pat))
    for name, param in self.named_parameters():
      if re.match(pat,name): param.requires_grad = False

  def parse_batch(self, batch):
    text_in_padded, lg_in, spk_in, style_in, spe_tgt_padded, gate_tgt_padded, lg_tgt, pho_tgt_padded, dur_tgt_padded, lg_pho_tgt, i_nm = batch
    [lg_batch,lg_max_in]=text_in_padded.shape
    text_in_padded = to_gpu(text_in_padded).long()
    spk_in=to_gpu(spk_in).long()
    style_in=to_gpu(style_in).long()
    for i_out in range(len(self.dim_data)):
      spe_tgt_padded[i_out] = to_gpu(spe_tgt_padded[i_out]).float()
      gate_tgt_padded[i_out] = to_gpu(gate_tgt_padded[i_out]).float()
    if self.dim_out_symbols and max(lg_pho_tgt)>0: pho_tgt_padded = to_gpu(pho_tgt_padded).long()
    if self.compute_durations and max(lg_pho_tgt)>0: dur_tgt_padded = to_gpu(dur_tgt_padded).float()
    return (
      (text_in_padded, lg_in, spk_in, style_in, spe_tgt_padded, lg_tgt, lg_pho_tgt),
      (spe_tgt_padded, gate_tgt_padded, pho_tgt_padded, dur_tgt_padded), i_nm)

  def parse_output(self, outputs, spe_lg=None):
    for i_out in range(len(self.dim_data)):
      if self.mask_padding and np.max(spe_lg[:,i_out]):
        mask = ~get_mask_from_lengths(spe_lg[:,i_out])
        mask = mask.expand(self.dim_data[i_out], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        outputs[0][i_out].data.masked_fill_(mask, 0.0) # mask decoder
        outputs[1][i_out].data.masked_fill_(mask, 0.0)# mask postnet
        outputs[2][i_out].data.masked_fill_(mask[:, 0, ::self.n_frames_per_step[i_out]], 1000.0)  # gate activation to sigmoid(1000)=1
    return outputs

  def forward(self, inputs):
    #Forward Tacotron with teacher forcing
    text_in, text_lg, spk_in, style_in, spe_tgt, spe_lg, pho_lg= inputs

#    print('BEFORE FORWARD: Memory {:.2f}Mo {}'.format(torch.cuda.memory_allocated()/1024/1024,spe_lg.max(axis=0)))
    [lg_batch,lg_max_in]=text_in.shape
    if self.nb_style_parameters:
      if self.use_style_input:
        style=to_gpu(torch.Tensor(self.style_parameters).repeat(lg_batch,1)).float()
        (c0,h0) = self.style_encoder.forward_with_imposed_style(style)
      else:
        (c0,h0), style = self.style_encoder(spe_tgt,spe_lg)
      if self.style_encoder.bias_encoder_with_entry:
        e_style=c0; c0=h0
    else:
      h0 = torch.zeros(2, lg_batch, self.encoder.encoder_lstm_hidden_dim); h0 = to_gpu(h0).float()
      c0 = torch.zeros(2, lg_batch, self.encoder.encoder_lstm_hidden_dim); c0 = to_gpu(c0).float()
      style = torch.zeros(lg_batch, 2); style = to_gpu(style).float()
    embedded_inputs = self.embedding(text_in).transpose(1, 2)
    encoder_outputs = self.encoder(embedded_inputs, text_lg, (h0,c0))
    if self.nb_speakers>1:
      e_spk=self.speaker_embedding(spk_in)
      if lg_batch>1: e_spk = e_spk[:,np.newaxis,:]
      e_spk = e_spk.repeat(1,lg_max_in,1); e_spk[encoder_outputs.eq(0.0)]=0 # respect du padding
      encoder_outputs = encoder_outputs + e_spk  # ajout d'un vecteur de locuteur à tous les vecteurs de plongement des caractères
    if self.nb_styles>1:
      e_style=self.style_embedding(style_in)
      if lg_batch>1: e_style = e_style[:,np.newaxis,:]
      e_style = e_style.repeat(1,lg_max_in,1); e_style[encoder_outputs.eq(0.0)]=0 # respect du padding
      encoder_outputs = encoder_outputs + e_style  # ajout d'un vecteur de locuteur à tous les vecteurs de plongement des caractères
    nb_out=len(self.dim_data)
    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = nb_out*[None], nb_out*[None], nb_out*[None], nb_out*[None]
    for i_out in range(nb_out):
      if spe_tgt[i_out].nelement():
        mel_outputs[i_out], gate_outputs[i_out], alignments[i_out]= self.decoder[i_out](encoder_outputs, spe_tgt[i_out], memory_lengths=text_lg)
        mel_outputs_postnet[i_out] = mel_outputs[i_out] + self.postnet[i_out](mel_outputs[i_out]) if self.postnet[i_out] else mel_outputs[i_out]
    pho_outputs = self.phonetize(encoder_outputs).transpose(1,2) if self.dim_out_symbols and max(pho_lg)>0 else torch.empty(lg_batch,0)
    dur_outputs = torch.empty(lg_batch,0)
#    dur_outputs = F.relu(self.compute_durations(encoder_outputs)).squeeze(2) if self.compute_durations and max(pho_lg)>0 else torch.empty(lg_batch,0)
    return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, pho_outputs, dur_outputs, style, alignments, encoder_outputs], spe_lg)

  def inference(self, inputs, seed):
    text_in, spk_in, style_in= inputs; lg_in=text_in.shape[1]
    
    embeddings=np.zeros([lg_in,0], dtype='float32')
    torch.manual_seed(seed)
    embedded_inputs = self.embedding(text_in).transpose(1, 2)
    if self.nb_style_parameters:
      style = to_gpu(torch.as_tensor([self.style_parameters])).float()
      (h0,c0) = self.style_encoder.inference_with_imposed_style(style)
    else:
      h0 = torch.zeros(2, 1, self.encoder.encoder_lstm_hidden_dim); h0 = to_gpu(h0).float()
      c0 = h0
      style = torch.zeros(2, 1); style = to_gpu(style).float()
#        print('embedded_inputs.shape={}'.format(embedded_inputs.shape))
    encoder_outputs = self.encoder.inference(embedded_inputs,(h0,c0))
    if self.nb_speakers>1:
      embedded_spk_in = self.speaker_embedding(spk_in)
      encoder_outputs = encoder_outputs+embedded_spk_in.repeat(1,lg_in,1)
    if self.nb_styles>1:
      embedded_style_in = self.style_embedding(style_in)
      encoder_outputs = encoder_outputs+embedded_style_in.repeat(1,lg_in,1)
    if "encoder.lstm" in self.save_embeddings:
      embeddings=np.concatenate((embeddings,encoder_outputs.cpu().data.numpy()[0,:,:]),axis=1)
    nb_out=len(self.dim_data)
    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = nb_out*[None], nb_out*[None], nb_out*[None], nb_out*[None]
    duree_imposee=0
    for i_out in range(nb_out):
      mel_outputs[i_out], gate_outputs[i_out] , alignments[i_out] = self.decoder[i_out].inference(encoder_outputs,duree_imposee)
      if (i_out==0): duree_imposee=mel_outputs[i_out].shape[2]/self.fe_data[i_out]
      mel_outputs_postnet[i_out] = mel_outputs[i_out] + self.postnet[i_out](mel_outputs[i_out]) if self.postnet[i_out] else mel_outputs[i_out]
    pho_outputs = self.phonetize(encoder_outputs).transpose(1,2) if self.dim_out_symbols else torch.empty(1,0)
    dur_outputs = torch.empty(1,0)
#    dur_outputs = F.relu(self.compute_durations(encoder_outputs)).squeeze(2) if self.compute_durations else torch.empty(1,0)
    outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, pho_outputs, dur_outputs, style, alignments, embeddings],None)
    return outputs
