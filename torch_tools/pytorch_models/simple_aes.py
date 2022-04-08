import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from dataclasses import dataclass
from functools import reduce

from typing import List

TORCH_PI = torch.acos(torch.zeros(1))

class BasicAE(nn.Module):
    def __init__(self, feature_dims : int = 10, latent_dim : int = 64, latent_out=False, softplus=True):
        super().__init__()
        self._latent_dim = latent_dim

        if isinstance(feature_dims, tuple):
            self.encoding_layer = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(tuple_product(feature_dims), self._latent_dim)
            )
            self.decoding_layer = torch.nn.Sequential(
                torch.nn.Linear(self._latent_dim, tuple_product(feature_dims)),
                ReshapeView(feature_dims)
            )
        else:
            self.encoding_layer = nn.Linear(feature_dims, latent_dim)
            self.decoding_layer = nn.Linear(latent_dim, feature_dims)

        self.latent_out = latent_out

        self.softplus = softplus

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        if self.latent_out:
            return x, {"latent": z}
        else:
            return x

    def encode(self, x):
        if self.softplus:
            x = F.softplus(self.encoding_layer(x))
        else:
            x = self.encoding_layer(x)
        return x

    def decode(self, x):
        x = self.decoding_layer(x)
        return x


class Basic2LAE(nn.Module):
    def __init__(self, feature_dims : int = 10, intermediate_dim=100, latent_dim : int = 64):
        super().__init__()
        self._latent_dim = latent_dim


        if isinstance(feature_dims, tuple):
            self.encoding_layer = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(tuple_product(feature_dims), self._latent_dim)
            )
            self.decoding_layer = torch.nn.Sequential(
                torch.nn.Linear(self._latent_dim, tuple_product(feature_dims)),
                ReshapeView(feature_dims)
            )
        else:
            self.encoding_layer = nn.Linear(intermediate_dim, latent_dim)
            self.decoding_layer = nn.Linear(intermediate_dim, feature_dims)

        self.encoding_h = nn.Linear(feature_dims, intermediate_dim)

        self.decoding_h = nn.Linear(latent_dim, intermediate_dim)

        self.act = F.softplus #F.LeakyRelu
        self.fact = lambda x: x #F.softplus #F.tanh

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.act(self.encoding_h(x))
        x = self.fact(self.encoding_layer(x))
        return x

    def decode(self, x):
        x = self.act(self.decoding_h(x))
        x = self.fact(self.decoding_layer(x))
        return x



class BasicVAE(nn.Module):
    def __init__(self,
                 feature_dims : int = 10, latent_dim : int = 64,
                 encoding_act=None, decoding_act=None, final_act=None):
        super().__init__()
        self._latent_dim = latent_dim

        if isinstance(feature_dims, tuple):
            tot_features = tuple_product(feature_dims)
            im_features = tot_features//2
            print(tot_features)
            self.encoding_layer_0 = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(tot_features,
                                im_features)
            )
            self.decoding_layer_mu = torch.nn.Sequential(
                torch.nn.Linear(im_features, tot_features),
                ReshapeView(feature_dims)
            )
            self.decoding_layer_logvar = torch.nn.Sequential(
                torch.nn.Linear(im_features, tot_features),
                ReshapeView(feature_dims)
            )
        else:
            tot_features = feature_dims
            im_features = tot_features//2
            self.encoding_layer_0 = nn.Linear(feature_dims, im_features)
            self.decoding_layer_mu = nn.Linear(im_features, feature_dims)
            self.decoding_layer_logvar = nn.Linear(im_features, feature_dims)
        
        

        self.encoding_mu = nn.Linear(im_features, latent_dim)
        self.encoding_logvar = nn.Linear(im_features, latent_dim)

        self.decoding_layer_0 = nn.Linear(latent_dim, im_features)

        if encoding_act is not None:
            self.encoding_act = encoding_act
        else:
            self.encoding_act = F.leaky_relu

        if decoding_act is not None:
            self.decoding_act = decoding_act
        else:
            self.decoding_act = F.leaky_relu

    def encode(self, x):
        m, l = self.encode_(x)
        if self.training:
            return self.reparametrize(m, l)
        else:
            return m
        
    def encode_(self, x):
        x = self.encoding_act(self.encoding_layer_0(x))
        mu = self.encoding_mu(x)
        logvar = self.encoding_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x):
        x = self.decoding_act(self.decoding_layer_0(x))
        mu = self.decoding_layer_mu(x)
        logvar = self.decoding_layer_logvar(x)
        return mu, logvar

    def forward(self, x):
        mu_latent, logvar_latent = self.encode_(x)
        z = self.reparametrize(mu_latent, logvar_latent)
        mu_x, logvar_x = self.decode(z)
        if self.training:
            x_out = self.reparametrize(mu_x, logvar_x)
        else:
            x_out = mu_x
        return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent,
                       "mu_x" : mu_x, "logvar_x" : logvar_x}


def tuple_product(t):
    return reduce(lambda x, y: x * y, t, 1)
    
@dataclass
class ConvSpec1D:
    # convolution
    channels : int = 1
    kernel : int = 2
    stride : int = 1
    dilation : int = 1
    # batch_norm
    batch_norm : bool = False
    # activation
    activation : callable = torch.nn.Identity
    _reverse_activation: callable = None

    @property
    def reverse_activation(self):
        if self._reverse_activation is not None:
            return self._reverse_activation
        else:
            return self.activation
    
    
    
@dataclass
class LinearSpec:
    size : [int, float] = 0.5
    # batch_norm
    dropout : bool = False
    batch_norm : bool = False
    # activation
    activation : callable = torch.nn.Identity
    _reverse_activation: callable = None

    @property
    def reverse_activation(self):
        if self._reverse_activation is not None:
            return self._reverse_activation
        else:
            return self.activation

class ReshapeView(nn.Module):
    def __init__(self, shape):
        super(ReshapeView, self).__init__()
        self._shape = shape

    def forward(self, x):
        return x.view(-1, *self._shape)
    
    def extra_repr(self) -> str:
        return f'reshape to [B, {self._shape}]'

    

class UberVAE(nn.Module):
    def __init__(self,
                 feature_dims : int = 10, latent_dim : int = 64,
                 n_conv_layers_1d : List[ConvSpec1D] = None,
                 n_linear_layers : List[LinearSpec] = None,
                 distributive_latent=True, distributive_recon=False,
                 output_mu=True, logloss2sigmoid=False,
                 #latent_activation=None, final_activation=False, # TODO!
                 ):
        super(UberVAE, self).__init__()
        self._conv_encoder = torch.nn.Identity()
        self._conv_decoder = torch.nn.Identity()

        self._lin_encoder = torch.nn.Identity()
        self._lin_decoder = torch.nn.Identity()

        self._feature_dims = feature_dims
        self._flatten_input = False
        self._blowup_channels = False
        self._conv_interface_shape = None
        self._conv_interface_size = None
        self._latent_interface_size = feature_dims
        self._latent_size = latent_dim

        # behavioral parameters
        self._distributive_latent = distributive_latent
        self._distributive_recon = distributive_recon
        self._output_mu = output_mu
        self._logloss2sigmoid = logloss2sigmoid

        conv_encoding_layers = []
        conv_decoding_layers = []
        deconv_channel_sizes = []
        deconv_output_sizes = []
        last_conv_decoding_layer_idx = 0
        if n_conv_layers_1d:
            self._blowup_channels = True
            if isinstance(feature_dims, tuple):
                self._blowup_channels = False
                conv_inchannels = feature_dims[0]
                conv_prev_output_dims = [feature_dims]
            else:
                conv_encoding_layers.append(ReshapeView((1, feature_dims)))
                conv_inchannels = 1
                conv_prev_output_dims = [(1, feature_dims,)]

            conv_channel_sizes = [conv_inchannels] + \
                [(cspec.channels) for cspec in n_conv_layers_1d]

            for conv_idx in range(len(n_conv_layers_1d)):
                cspec = n_conv_layers_1d[conv_idx]
                conv_part = torch.nn.Conv1d(conv_channel_sizes[conv_idx],
                                            cspec.channels, cspec.kernel,
                                            stride=cspec.stride, dilation=cspec.dilation)
                sample_batch = conv_part(torch.rand(1, *(conv_prev_output_dims[conv_idx])))
                conv_encoding_layers.append(conv_part)
                conv_prev_output_dims.append(tuple(sample_batch.shape[1:]))
                if cspec.batch_norm:
                    conv_encoding_layers.append(torch.nn.BatchNorm1d(sample_batch.shape[1]))
                conv_encoding_layers.append(cspec.activation())
            conv_encoding_layers.append(torch.nn.Flatten())

            print("convolution outputs ", "-".join([str(x) for x in conv_prev_output_dims]))            
   
            self._conv_interface_shape = tuple(conv_prev_output_dims[-1])
            self._conv_interface_size = tuple_product(self._conv_interface_shape)
            self._latent_interface_size = self._conv_interface_size

            deconv_channel_sizes = list(reversed(conv_prev_output_dims))
            deconv_layers1d = list(reversed(n_conv_layers_1d))
            
            conv_decoding_layers.append(ReshapeView(self._conv_interface_shape))
            for dc_idx in range(len(deconv_layers1d)):
                last_conv_decoding_layer_idx = len(conv_decoding_layers)
                cspec = deconv_layers1d[dc_idx]
                output_shape = (0, 0)
                deconv_part = None
                out_padding = 0
                for i in range(2):
                    deconv_part = torch.nn.ConvTranspose1d(
                        cspec.channels, deconv_channel_sizes[dc_idx+1][0], cspec.kernel,
                        stride=cspec.stride, dilation=cspec.dilation,
                        output_padding=out_padding)
                    sample_batch = deconv_part(torch.rand(1, *deconv_channel_sizes[dc_idx]))
                    output_shape = sample_batch.shape
                    if output_shape[-1] != deconv_channel_sizes[dc_idx+1][-1]:
                        out_padding = deconv_channel_sizes[dc_idx+1][-1]-output_shape[-1]
                    else:
                        break
                deconv_output_sizes.append(output_shape)
                conv_decoding_layers.append(deconv_part)
                if cspec.batch_norm:
                    conv_decoding_layers.append(torch.nn.BatchNorm1d(output_shape[0]))
                conv_decoding_layers.append(cspec.reverse_activation())

            if self._blowup_channels:
                conv_decoding_layers.append(ReshapeView((feature_dims,)))

        lin_encoding_layers = []
        lin_decoding_layers = []
        lin_layer_upsizes = []
        last_decoding_layer_idx = 0
        if n_linear_layers:
            if not n_conv_layers_1d and isinstance(feature_dims, tuple):
                self._flatten_input = True
                lin_input = tuple_product(feature_dims)
                lin_encoding_layers.append(torch.nn.Flatten())
            elif n_conv_layers_1d:
                lin_input = self._conv_interface_size
            else:
                lin_input = feature_dims

            lin_layer_insizes = [lin_input,] + \
                [int(lspec.size*self._latent_interface_size) if isinstance(lspec.size, float) else lspec.size
                 for lspec in n_linear_layers]

            for lin_idx in range(len(n_linear_layers)):
                lspec = n_linear_layers[lin_idx]
                if lspec.dropout:
                    raise NotImplementedError
                lin_layer = torch.nn.Linear(lin_layer_insizes[lin_idx],
                                            lin_layer_insizes[lin_idx+1])
                lin_encoding_layers.append(lin_layer)
                if lspec.batch_norm:
                    lin_encoding_layers.append(torch.nn.BatchNorm1d(lin_layer_insizes[lin_idx+1]))
                lin_encoding_layers.append(lspec.activation())

            self._latent_interface_size = lin_layer_insizes[-1]

            lin_layer_upsizes = list(reversed(lin_layer_insizes))
            lin_layers_up = list(reversed(n_linear_layers))

            for rlin_idx in range(len(lin_layers_up)):
                last_decoding_layer_idx = len(lin_decoding_layers)
                lspec = lin_layers_up[rlin_idx]
                lin_layer = torch.nn.Linear(lin_layer_upsizes[rlin_idx], lin_layer_upsizes[rlin_idx+1])
                lin_decoding_layers.append(lin_layer)
                if lspec.batch_norm:
                    lin_decoding_layers.append(torch.nn.BatchNorm1d(
                        lin_layer_upsizes[rlin_idx+1]))
                lin_decoding_layers.append(lspec.reverse_activation())

            if self._flatten_input:
                lin_decoding_layers.append(ReshapeView(feature_dims))
                

        if isinstance(self._feature_dims, tuple) and (not n_conv_layers_1d) and not(n_linear_layers):
            self._mu_latent = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(tuple_product(self._feature_dims), self._latent_size)
            )
            self._logvar_latent = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(tuple_product(self._feature_dims), self._latent_size)
            )
            
        else:
            self._mu_latent = torch.nn.Linear(self._latent_interface_size, self._latent_size)
            self._logvar_latent = torch.nn.Linear(self._latent_interface_size, self._latent_size)


        if isinstance(self._feature_dims, tuple) and (not n_conv_layers_1d) and not(n_linear_layers):
            self._mu_x = torch.nn.Sequential(
                torch.nn.Linear(self._latent_size, tuple_product(self._feature_dims)),
                ReshapeView(self._feature_dims)
            )
            self._logvar_x = torch.nn.Sequential(
                torch.nn.Linear(self._latent_size, tuple_product(self._feature_dims)),
                ReshapeView(self._feature_dims)
            )
        elif (not n_conv_layers_1d) and not(n_linear_layers):
            self._mu_x = torch.nn.Linear(self._latent_size, self._feature_dims)
            self._logvar_x = torch.nn.Linear(self._latent_size, self._feature_dims)
        else:
            self._mu_x = None
            self._logvar_x = None


        if conv_encoding_layers:
            self._conv_encoder = torch.nn.Sequential(*conv_encoding_layers)
            if lin_encoding_layers:
                self._conv_decoder = torch.nn.Sequential(*conv_decoding_layers[:last_conv_decoding_layer_idx])
            else:
                self._conv_decoder = torch.nn.Sequential(
                    torch.nn.Linear(self._latent_size, self._latent_interface_size),
                    *conv_decoding_layers[:last_conv_decoding_layer_idx])
            self._mu_x = torch.nn.Sequential(*conv_decoding_layers[last_conv_decoding_layer_idx:])
            #print(deconv_channel_sizes)
            if self._blowup_channels:
                self._logvar_x = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(tuple_product(deconv_channel_sizes[len(n_conv_layers_1d)-1]),
                                    self._feature_dims),
                    torch.nn.Tanh()
                )
            else:
                self._logvar_x = torch.nn.Sequential(
                    torch.nn.Linear(deconv_channel_sizes[len(n_conv_layers_1d)-1][-1],
                                    self._feature_dims[-1]),
                    torch.nn.Tanh()
                )

        if lin_encoding_layers:
            self._lin_encoder = torch.nn.Sequential(*lin_encoding_layers)
            if conv_encoding_layers:
                self._lin_decoder = torch.nn.Sequential(
                    torch.nn.Linear(self._latent_size, self._latent_interface_size),
                    *lin_decoding_layers)
            #print(lin_layer_upsizes)
            else:
                self._lin_decoder = torch.nn.Sequential(
                    torch.nn.Linear(self._latent_size, self._latent_interface_size),
                    *lin_decoding_layers[:last_decoding_layer_idx])
                self._mu_x = torch.nn.Sequential(*lin_decoding_layers[last_decoding_layer_idx:])
                if not self._flatten_input:
                    self._logvar_x = torch.nn.Sequential(
                        torch.nn.Linear(lin_layer_upsizes[len(n_linear_layers)-1],
                                        self._feature_dims),
                        torch.nn.Tanh()
                    )
                else:
                    self._logvar_x = torch.nn.Sequential(
                        torch.nn.Linear(lin_layer_upsizes[len(n_linear_layers)-1],
                                        tuple_product(self._feature_dims)),
                        torch.nn.Tanh(),
                        ReshapeView(self._feature_dims)
                    )


    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        m, l = self.encode_(x)
        if self.training and self._distributive_latent:
            latent = self.reparametrize(m, l)
        elif self._distributive_latent and not self._output_mu:
            latent = self.reparametrize(m, l)
        else:
            latent = m
        return latent
            
    def encode_(self, x):
        x = self._conv_encoder(x)
        x = self._lin_encoder(x)
        mu = self._mu_latent(x)
        logvar = self._logvar_latent(x)
        return mu, logvar

    def decode_(self, x):
        x = self._lin_decoder(x)
        x = self._conv_decoder(x)
        mu = self._mu_x(x)
        logvar = self._logvar_x(x)
        return mu, logvar

    def forward(self, x):
        mu_latent, logvar_latent = self.encode_(x)
        if self.training and self._distributive_latent:
            latent = self.reparametrize(mu_latent, logvar_latent)
        elif self._distributive_latent and not self._output_mu:
            latent = self.reparametrize(mu_latent, logvar_latent)
        else:
            latent = mu_latent

        mu_x, logvar_x = self.decode_(latent)
        if self.training and self._distributive_recon:
            x_out = self.reparametrize(mu_x, logvar_x)
        elif self._distributive_latent and not self._output_mu:
            x_out = self.reparametrize(mu_x, logvar_x)
        else:
            x_out = mu_x
            
        return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent,
                       "mu_x" : mu_x, "logvar_x" : logvar_x}

    def add_variational_latent(self, size, lock=False):
        pass



def kld(mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar
                           - mu.pow(2)
                           - logvar.exp(), axis=1)
    return kld


def vae_loss(recon_x, tru_x, beta=1, **kwargs):
    # full vae loss for modeling a distributive latent space
    # AND a distributive reconstruction
    # correct formulation here:
    mu_latent = kwargs["mu_latent"]
    logvar_latent = kwargs["logvar_latent"]
    mu_x = kwargs["mu_x"]
    logvar_x = kwargs["logvar_x"]

    loss_rec = -torch.sum(
        (-0.5 * torch.log2(TORCH_PI.to(mu_x.device)*2))
        + (-0.5 * logvar_x)
        + ((-0.5 / torch.exp(logvar_x))
           * (tru_x - mu_x) ** 2.0),
        axis=1
    )

    KLD = beta * kld(mu_latent, logvar_latent)
    loss = torch.mean(loss_rec + KLD, dim=0)
    return loss

def vae_loss_cnguyen(recon_x, tru_x, beta=1, **kwargs):
    # full vae loss for modeling a distributive latent space
    # AND a reconstruction with a simple error estimate.
    # https://cnguyen10.github.io/2020/11/24/vae-normalizing-constant-matters.html
    pass

def naive_vae_loss(recon_x, tru_x, beta=1, **kwargs):
    mu_latent = kwargs["mu_latent"]
    logvar_latent = kwargs["logvar_latent"]
    mu_x = kwargs["mu_x"]
    logvar_x = kwargs["logvar_x"]

    loss_rec = F.mse_loss(mu_x, tru_x, reduction='mean')

    KLD = beta * kld(mu_latent, logvar_latent)
    loss = torch.mean(loss_rec + KLD, dim=0)
    return loss

def cont_bernoulli_vae_loss(recon_x, tru_x, beta=1, **kwargs):
    # vae loss for continouus [0,1]-variables
    pass

def plain_mse(recon_x, tru_x, **kwargs):
    loss_rec = F.mse_loss(recon_x, tru_x, reduction='mean')
    return loss_rec

def ortho_loss(recon_x, tru_x, **kwargs):
    latent = kwargs["latent"]
    
