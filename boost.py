#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

# --- Gradient Reversal Layer ---
class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer.
    Forward: Identity
    Backward: Negate gradients (multiply by -lambda)
    """
    @staticmethod
    def forward(ctx, x, lambda_val=1.0):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None

def grad_reverse(x, lambda_val=1.0):
    return GradReverse.apply(x, lambda_val)


def torch_convolve_1d(a, b):
    """
    实现 np.convolve(a, b, mode='full') 的批处理版本
    a: (Batch, L1)
    b: (Batch, L2)
    返回: (Batch, L1+L2-1)
    """
    batch_size = a.shape[0]
    L1, L2 = a.shape[1], b.shape[1]
    
    # 使用 conv1d 但需要正确设置padding
    a_in = a.view(1, batch_size, L1)
    b_flipped = b.flip(1).view(batch_size, 1, L2)
    
    # padding = L2 - 1 得到 full convolution
    out = F.conv1d(a_in, b_flipped, groups=batch_size, padding=L2-1)
    
    return out.view(batch_size, -1)


def genNotchCoeffs(fc, bw, c, G, fs):
    """
    Differentiable Notch Filter Generation using PyTorch
    Args:
        fc: Center frequencies (Batch, nBands)
        bw: Bandwidths (Batch, nBands)
        c: Filter length (int) - fixed for all bands
        G: Gain (Batch,)
        fs: Sampling rate (int)
    Returns:
        b: Filter coefficients (Batch, TotalLength)
    """
    device = fc.device
    batch_size, nBands = fc.shape
    
    # Flatten for processing
    fc_flat = fc.reshape(-1, 1) # (batch_size * nBands, 1)
    bw_flat = bw.reshape(-1, 1)
    
    # Calculate f1, f2
    f1 = fc_flat - bw_flat / 2
    f2 = fc_flat + bw_flat / 2
    
    # Clamp frequencies to valid range
    f1 = torch.clamp(f1, min=1e-3)
    f2 = torch.clamp(f2, max=fs/2 - 1e-3)
    
    # Normalize frequencies
    nyq = fs / 2.0
    f1_n = f1 / nyq
    f2_n = f2 / nyq
    
    # Generate filter coefficients (Hamming windowed sinc)
    n = torch.arange(c, device=device).float().view(1, -1) # (c) -> (1, c)
    alpha = (c - 1) / 2.0
    m = n - alpha
    
    # Bandstop = Delta - Bandpass
    # Bandpass = 2f2 sinc(2f2 m) - 2f1 sinc(2f1 m)
    # torch.sinc(x) = sin(pi*x)/(pi*x)
    
    h1 = f1_n * torch.sinc(f1_n * m) 
    h2 = f2_n * torch.sinc(f2_n * m)
    h_bp = h2 - h1
    
    delta = torch.zeros_like(m)
    if c % 2 == 1:
        delta[:, int(alpha)] = 1.0
    else:
        delta[:, c // 2] = 1.0 
        
    h_bs = delta - h_bp
    
    # Apply Hamming window
    window = torch.hamming_window(c, periodic=False, device=device).view(1, -1)
    h_bs = h_bs * window
    
    # Reshape to (Batch, nBands, c)
    filters = h_bs.view(batch_size, nBands, c)
    
    # Convolve filters sequentially
    # Start with the first filter
    b_final = filters[:, 0, :]
    
    for i in range(1, nBands):
        b_final = torch_convolve_1d(b_final, filters[:, i, :])
        
    # Apply Gain
    # Normalize by max frequency response
    fft_len = 2 ** int(np.ceil(np.log2(b_final.shape[-1]) + 1))
    b_padded = F.pad(b_final, (0, fft_len - b_final.shape[-1]))
    
    H = torch.fft.rfft(b_padded, dim=-1)
    max_H = torch.amax(torch.abs(H), dim=-1, keepdim=True)
    max_H = torch.clamp(max_H, min=1e-8)
    
    # Handle G with flexible shape: (Batch,) or (Batch, 1)
    gain_factor = torch.pow(10.0, G / 20.0).view(batch_size, -1)
    b_norm = b_final * gain_factor / max_H
    
    return b_norm


def filterFIR(x, b):
    """
    Differentiable FIR filtering using PyTorch
    Args:
        x: Input signal (Batch, Time) 
        b: Filter coefficients (Batch, FilterLength)
    Returns:
        y: Filtered signal (Batch, Time)
    """  
 
    batch_size, time_len = x.shape
    filter_len = b.shape[1]
    
    # Reshape for grouped convolution
    # Input: (1, Batch, Time) - treating batch as channels for grouped conv
    x_in = x.view(1, batch_size, time_len)
    
    # Weight: (Batch, 1, FilterLength)
    # Flip b for convolution (lfilter is convolution)
    weight = b.flip(1).view(batch_size, 1, filter_len)
    
    # Padding to maintain size (Same padding)
    padding_total = filter_len - 1
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left
    
    x_padded = F.pad(x_in, (padding_left, padding_right))
    
    y = F.conv1d(x_padded, weight, groups=batch_size)
    
    # y shape: (1, Batch, Time)
    y = y.view(batch_size, time_len)
    
    return y

# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x, fc, bw, gain, fs, c):
    """
    Differentiable Linear and Non-linear convolutive noise
    Args:
        x: Input signal (Batch, Time)
        fc: Center frequencies (Batch, N_f, nBands)
        bw: Bandwidths (Batch, N_f, nBands)
        gain: Gain (Batch, N_f)
        fs: Sampling rate
        c: Filter length
    """ 
    N_f = fc.shape[1] # Number of non-linear components
    
    y_accum = torch.zeros_like(x)
    
    for i in range(N_f):
        order = i + 1
        
        # Extract parameters for current order
        # fc[:, i, :] -> (Batch, nBands)
        curr_fc = fc[:, i, :]
        curr_bw = bw[:, i, :]
        curr_gain = gain[:, i]
        
        # Generate filter coefficients
        b = genNotchCoeffs(curr_fc, curr_bw, c, curr_gain, fs)
        
        # Non-linear transformation: x^order
        # Note: x^2 is positive. x^3 preserves sign.
        x_pow = torch.pow(x, order)
        
        # Filter
        y_filtered = filterFIR(x_pow, b)
        
        # Accumulate
        y_accum = y_accum + y_filtered
        
    # Remove mean
    y_accum = y_accum - torch.mean(y_accum, dim=-1, keepdim=True)
    
    # Normalize
    max_val = torch.amax(torch.abs(y_accum), dim=-1, keepdim=True)
    scale = torch.where(max_val > 1.0, 1.0 / (max_val + 1e-8), torch.ones_like(max_val))
    y = y_accum * scale
    
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, density, gain):
    """
    Differentiable Impulsive Signal Dependent (ISD) additive noise
    Args:
        x: Input signal (Batch, Time)
        density: Density of impulsive noise (Batch, 1), range [0, 1]
        gain: Gain of the noise (Batch, 1)
    """ 
    # Generate soft mask (differentiable approximation of discrete sampling)
    # We want to select approximately 'density' fraction of samples
    # Lower sharpness for more stable gradients during training
    u = torch.rand_like(x)
    mask = torch.sigmoid((density - u) * 10.0) # Sharpness factor 10 (reduced from 50 for stability)
    
    # Generate noise component f_r
    # f_r = U[-1, 1] * U[-1, 1]
    n1 = 2 * torch.rand_like(x) - 1
    n2 = 2 * torch.rand_like(x) - 1
    f_r = n1 * n2
    
    # Signal dependent noise: r = gain * x * f_r
    noise = gain * x * f_r
    
    # Add noise where mask is active
    y = x + mask * noise
    
    # Differentiable Normalization
    max_val = torch.amax(torch.abs(y), dim=-1, keepdim=True)
    scale = torch.where(max_val > 1.0, 1.0 / (max_val + 1e-8), torch.ones_like(max_val))
    y = y * scale
    
    return y


# Stationary signal independent noise
def SSI_additive_noise(x, fc, bw, gain, snr, fs, c):
    """
    Differentiable Stationary Signal Independent (SSI) additive noise
    Args:
        x: Input signal (Batch, Time)
        fc: Center frequencies (Batch, nBands)
        bw: Bandwidths (Batch, nBands)
        gain: Gain (Batch, 1) - for the filter
        snr: SNR (Batch, 1) - for the additive noise
        fs: Sampling rate
        c: Filter length (int)
    """ 
    # 1. Generate white noise
    noise = torch.randn_like(x)
    
    # 2. Generate filter coefficients
    b = genNotchCoeffs(fc, bw, c, gain, fs)
    
    # 3. Filter noise (Coloring)
    noise_colored = filterFIR(noise, b)
    
    # 4. Always normalize noise to [-1, 1] (matching RawBoost: normWav(noise, always=1))
    max_noise = torch.amax(torch.abs(noise_colored), dim=-1, keepdim=True)
    noise_normalized = noise_colored / (max_noise + 1e-8)
    
    # 5. Scale by SNR
    # SNR = 20 log10( ||x|| / ||n|| )
    # ||n_target|| = ||x|| / 10^(SNR/20)
    
    x_energy = torch.norm(x, p=2, dim=-1, keepdim=True)
    n_energy = torch.norm(noise_normalized, p=2, dim=-1, keepdim=True)
    
    target_n_energy = x_energy / (torch.pow(10.0, snr / 20.0) + 1e-8)
    
    noise_scaled = noise_normalized * (target_n_energy / (n_energy + 1e-8))
    
    # 6. Add to signal (no normalization after adding, matching RawBoost)
    y = x + noise_scaled
    
    return y


class GlobalRouter(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 4 channels (1 raw + 3 augmented)
        # Extremely lightweight encoder to capture global statistics
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=128, stride=8, padding=64),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=64, stride=8, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.head_mix = nn.Linear(32, 3)

    def forward(self, x, y_aug):
        # x: (B, Time)
        # y_aug: (B, 3, Time)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, 1, Time)
        
        # Concatenate raw and augmented signals along channel dimension
        # This allows the router to compare the raw signal with the augmented versions
        inp = torch.cat([x, y_aug], dim=1) # (B, 4, Time)
        
        feat = self.encoder(inp)
        feat = self.fc(feat)
        
        # No GRL: The router adaptively selects the most effective noise type
        # (or the one the model is most robust to, depending on training dynamics,
        # but the experts ensure the noise itself is hard).
        mix_weights = F.softmax(self.head_mix(feat), dim=1)
        return mix_weights


class ParameterGenerator(nn.Module):
    def __init__(self, n_bands=5, n_f=5):
        super().__init__()
        self.n_bands = n_bands
        self.n_f = n_f

        # LnL expert: Spectral focus, reduced channels
        self.lnl_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=64, stride=4, padding=32),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=32, stride=4, padding=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.lnl_fc = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.head_lnl_fc = nn.Sequential(nn.Linear(32, n_f * n_bands), nn.Sigmoid())
        self.head_lnl_bw = nn.Sequential(nn.Linear(32, n_f * n_bands), nn.Sigmoid())

        # ISD expert: Temporal/Envelope focus, minimal capacity
        self.isd_encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=32, stride=4, padding=16),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.isd_fc = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        # Density is a shape/placement knob; keep adversarial but enforce a floor to prevent collapse.
        self.head_isd_density = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

        # SSI expert: Spectral focus, similar to LnL but independent
        self.ssi_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=64, stride=4, padding=32),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=32, stride=4, padding=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.ssi_fc = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.head_ssi_fc = nn.Sequential(nn.Linear(32, n_bands), nn.Sigmoid())
        self.head_ssi_bw = nn.Sequential(nn.Linear(32, n_bands), nn.Sigmoid())

    def forward(self, x):
        # x: (Batch, Time)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # --- LnL Expert (adversarial for shape) ---
        lnl_feat = self.lnl_encoder(x)
        lnl_feat = self.lnl_fc(lnl_feat)
        lnl_feat_adv = grad_reverse(lnl_feat, lambda_val=1.0)
        lnl_fc = self.head_lnl_fc(lnl_feat_adv).view(-1, self.n_f, self.n_bands)
        lnl_bw = self.head_lnl_bw(lnl_feat_adv).view(-1, self.n_f, self.n_bands)

        # --- ISD Expert (adversarial for placement) ---
        isd_feat = self.isd_encoder(x)
        isd_feat = self.isd_fc(isd_feat)
        isd_feat_adv = grad_reverse(isd_feat, lambda_val=1.0)
        # Floor density to avoid the trivial "no-impulse" solution.
        isd_density = self.head_isd_density(isd_feat_adv)
        isd_density = torch.clamp(isd_density, min=0.05, max=1.0)

        # --- SSI Expert (adversarial for shape) ---
        ssi_feat = self.ssi_encoder(x)
        ssi_feat = self.ssi_fc(ssi_feat)
        ssi_feat_adv = grad_reverse(ssi_feat, lambda_val=1.0)
        ssi_fc = self.head_ssi_fc(ssi_feat_adv)
        ssi_bw = self.head_ssi_bw(ssi_feat_adv)

        return {
            'lnl': (lnl_fc, lnl_bw),
            'isd': (isd_density,),
            'ssi': (ssi_fc, ssi_bw)
        }

class AutoBoost(nn.Module):
    def __init__(self, fs=16000, n_bands=5, n_f=5, filter_len=101):
        super().__init__()
        self.fs = fs
        self.n_bands = n_bands
        self.n_f = n_f
        self.filter_len = filter_len
        self.param_gen = ParameterGenerator(n_bands=n_bands, n_f=n_f)
        self.router = GlobalRouter()

    def forward(self, x):
        """
        Args:
            x: Input waveform (Batch, Time)
        Returns:
            y: Augmented waveform (Batch, Time)
            params: Dictionary of generated parameters (for regularization/logging)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Generate parameters (Shape params are adversarial)
        params = self.param_gen(x)
        
        # --- 1. LnL Augmentation ---
        # Map parameters to physical ranges
        # fc: [0, 1] -> [0, fs/2]
        lnl_fc = params['lnl'][0] * (self.fs / 2.0)
        # bw: [0, 1] -> [0, fs/4] (Limit bandwidth)
        lnl_bw = params['lnl'][1] * (self.fs / 4.0)
        
        # Intensity Parameter Decoupling: Random Sampling
        # Gain: Uniform [-10, 10] dB
        lnl_gain = (torch.rand(batch_size, self.n_f, device=device) * 20.0) - 10.0
        
        y_lnl = LnL_convolutive_noise(
            x, lnl_fc, lnl_bw, lnl_gain, self.fs, self.filter_len
        )
        
        # --- 2. ISD Augmentation ---
        # Density is learned (placement), but bounded away from 0 to avoid collapse
        isd_density = params['isd'][0]
        # Gain remains random to enforce strong perturbations
        isd_gain = torch.rand(batch_size, 1, device=device) * 5.0
        
        y_isd = ISD_additive_noise(x, isd_density, isd_gain)
        
        # --- 3. SSI Augmentation ---
        # fc, bw: similar to LnL
        ssi_fc = params['ssi'][0] * (self.fs / 2.0)
        ssi_bw = params['ssi'][1] * (self.fs / 4.0)
        
        # Intensity Parameter Decoupling: Random Sampling
        # Gain: Uniform [-10, 10] dB
        ssi_gain = (torch.rand(batch_size, 1, device=device) * 20.0) - 10.0
        # SNR: Uniform [0, 40] dB
        ssi_snr = torch.rand(batch_size, 1, device=device) * 40.0
        
        y_ssi = SSI_additive_noise(
            x, ssi_fc, ssi_bw, ssi_gain, ssi_snr, self.fs, self.filter_len
        )
        
        # --- 4. Weighted Sum ---
        # Stack outputs: (Batch, 3, Time)
        y_stack = torch.stack([y_lnl, y_isd, y_ssi], dim=1)
        
        # Calculate Mix Weights using Global Router
        # Router sees both Raw (x) and Augmented (y_stack) signals
        mix_weights = self.router(x, y_stack)
        
        # Weighted sum
        w = mix_weights.unsqueeze(-1) # (Batch, 3, 1)
        y_out = torch.sum(y_stack * w, dim=1)
        
        # Add mix weights to params for logging
        params['mix'] = mix_weights
        
        return y_out, params

