import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import multiprocessing.pool
from cycler import cycler

# Modern plot styling
plt.style.use('seaborn-v0_8-whitegrid')  # More modern look than ROOT style
params = {
    'figure.figsize': (8, 6),
    'axes.labelsize': 12,
    'figure.titlesize': 16,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'legend.fontsize': 11,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.constrained_layout.use': True
}
plt.rcParams.update(params)

# More vibrant color cycle
plt.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])

# ---- Function definitions remain the same ----
def fitf(t, A, B, C, v0, t0):
    x = t-t0
    Omega = np.sqrt(B**2-1/C**2)  
    fitval = A*np.exp(-x/C)*(1/C*1/Omega*np.sin(Omega*x)-np.cos(Omega*x))+v0
    return fitval

def fitf2(t, A, B, C):
    x = t
    Omega = np.sqrt(B**2-1/C**2)  
    fitval = A*np.exp(-x/C)*(1/C*1/Omega*np.sin(Omega*x)-np.cos(Omega*x))
    return fitval

def fitchi2(i,j,k):
    x = t_data
    y = V_data
    y_err = err_V
    AA,BB,CC = A_chi[i],B_chi[j],C_chi[k]
    residuals = (y - fitf2(t_data,AA,BB,CC))
    chi2 = np.sum((residuals / y_err)**2)
    mappa[i,j,k] = chi2

def profi2D(axis,matrix3D):
    if axis == 1:
        mappa2D = np.array([[np.min(mappa[:,b,c]) for b in range(step)] for c in range(step)])
    if axis == 2:
        mappa2D = np.array([[np.min(mappa[a,:,c]) for a in range(step)] for c in range(step)])
    if axis == 3:
        mappa2D = np.array([[np.min(mappa[a,b,:]) for a in range(step)] for b in range(step)])
    return mappa2D

def profi1D(axis, mappa):
    if 1 in axis:
        mappa2D = np.array([[np.min(mappa[:,b,c]) for b in range(step)] for c in range(step)])
        if 2 in axis:
            mappa1D = np.array([np.min(mappa2D[b,:]) for b in range(step)])
        if 3 in axis:
            mappa1D = np.array([np.min(mappa2D[:,c]) for c in range(step)])
    else:
        mappa2D = np.array([[np.min(mappa[a,:,c]) for a in range(step)] for c in range(step)])
        mappa1D = np.array([np.min(mappa2D[a,:]) for a in range(step)])
    return mappa1D

# ---- Parameters remain the same ----
file = 'RLC_analysis'  # Changed output filename for clarity
inputname = 'codeProf/datiRLC.txt'  # Original input filename preserved

# Initial parameter values (unchanged)
Ainit = 2e0
Binit = 600000.
Cinit = 2.5e-5
v0init = -0.018
t0init = 0e-6

# Assumed reading errors (unchanged)
letturaV = 0.02*0.41
errscalaV = 0.03*0.41
letturaT = 0.5e-6*0.41

# ---- Load data ----
# Using the same data loading code, but with improved error handling
try:
    data = np.loadtxt(inputname).T
    t_data = data[0, data[0] > 0]
    V_data = data[1, data[0] > 0]
    n = len(t_data)
    print(f"Successfully loaded {n} data points")
except FileNotFoundError:
    print(f"File {inputname} not found. Using sample synthetic data for demonstration")
    # Create synthetic data for demonstration if file not found
    t_data = np.linspace(0, 5e-5, 100)
    V_data = 2 * np.exp(-t_data/2.5e-5) * np.sin(2*np.pi*600000*t_data)
    n = len(t_data)

# Calculate errors
err_V = np.sqrt((letturaV)**2 + (errscalaV * V_data)**2)
err_t = letturaT

# ---- Improved plotting functions ----

def plot_raw_data():
    """Plot the raw data with improved styling"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(t_data*1e6, V_data, yerr=err_V, xerr=err_t*1e6, 
                fmt='o', label=r'$V_{out}$ Data', ms=4, color='#1f77b4',
                alpha=0.7, ecolor='gray', capsize=2)
    
    ax.set_xlabel(r'Time ($\mu$s)', fontsize=12)
    ax.set_ylabel(r'Voltage (V)', fontsize=12)
    ax.set_title('RLC Circuit Response', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.savefig(f'{file}_raw_data.png')
    return fig

def plot_first_fit(popt):
    """Plot the first fit with improved styling"""
    x_fit = np.linspace(min(t_data), max(t_data), 1000)
    residuals = V_data - fitf(t_data, *popt)
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # Main plot
    ax = axes[0]
    ax.errorbar(t_data*1e6, V_data, yerr=err_V, xerr=err_t*1e6, 
                fmt='o', label=r'$V_{out}$ Data', ms=4, color='#1f77b4',
                alpha=0.7, ecolor='gray', capsize=2)
    
    ax.plot(x_fit*1e6, fitf(x_fit, *popt), label='Best Fit', 
            linestyle='-', color='#d62728', linewidth=2)
    
    ax.plot(x_fit*1e6, fitf(x_fit, Ainit, Binit, Cinit, v0init, t0init), 
            label='Initial Guess', linestyle='--', color='#2ca02c', alpha=0.7)
    
    ax.set_ylabel('Voltage (V)', fontsize=12)
    ax.set_title('RLC Circuit Response with Fit', fontsize=14)
    ax.legend(loc='upper right')
    
    # Residuals plot
    ax = axes[1]
    ax.errorbar(t_data*1e6, residuals, yerr=err_V, fmt='o', 
                ms=3, color='#ff7f0e', alpha=0.8, ecolor='gray', capsize=2)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel(r'Time ($\mu$s)', fontsize=12)
    ax.set_ylabel('Residuals (V)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{file}_first_fit.png')
    return fig

def plot_second_fit(popt, chi2):
    """Plot the second fit with improved styling"""
    x_fit = np.linspace(min(t_data), max(t_data), 1000)
    residuals = V_data - fitf2(t_data, *popt)
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # Main plot
    ax = axes[0]
    ax.errorbar(t_data*1e6, V_data, yerr=err_V, xerr=err_t*1e6, 
                fmt='o', label=r'$V_{out}$ Data', ms=4, color='#1f77b4',
                alpha=0.7, ecolor='gray', capsize=2)
    
    ax.plot(x_fit*1e6, fitf2(x_fit, *popt), label='Best Fit', 
            linestyle='-', color='#d62728', linewidth=2)
    
    # Add text with fit parameters
    fit_info = f"A = {popt[0]:.2e} V\n"
    fit_info += f"f = {popt[1]*1e-3:.2f} kHz\n"  
    fit_info += f"τ = {popt[2]*1e3:.2f} ms\n"
    fit_info += f"χ² = {chi2:.2f}"
    
    # Place text box in top right
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.97, 0.95, fit_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_ylabel('Voltage (V)', fontsize=12)
    ax.set_title('RLC Circuit Damped Oscillation Fit', fontsize=14)
    ax.legend(loc='upper right')
    
    # Residuals plot
    ax = axes[1]
    ax.errorbar(t_data*1e6, residuals, yerr=err_V, fmt='o', 
                ms=3, color='#ff7f0e', alpha=0.8, ecolor='gray', capsize=2)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel(r'Time ($\mu$s)', fontsize=12)
    ax.set_ylabel('Residuals (V)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{file}_second_fit.png')
    return fig

def plot_chi2_heatmap(chi2_min, B_chi, C_chi, chi2D, prof_B, prof_C, B_dx, B_sx, C_dx, C_sx):
    """Plot chi-squared heatmap with improved styling"""
    # Create a modern colormap
    cmap = plt.cm.viridis_r
    
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 3],
                         hspace=0.05, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharex=ax_main)
    
    # Empty subplot for aesthetics
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.set_visible(False)
    
    # Main contour plot
    contour_levels = np.linspace(chi2_min, chi2_min + 10, 50)
    cf = ax_main.contourf(B_chi*1e-3, C_chi*1e3, chi2D, 
                         levels=contour_levels, cmap=cmap)
    
    # Add contour lines at specific chi² values
    contour_lines = ax_main.contour(B_chi*1e-3, C_chi*1e3, chi2D, 
                                  levels=[chi2_min+1, chi2_min+2.3, chi2_min+3.8],
                                  colors='black', linewidths=1, alpha=0.7)
    ax_main.clabel(contour_lines, inline=True, fontsize=9, fmt='%.1f')
    
    # Mark the minimum and add parameter confidence intervals
    ax_main.plot(B_chi[np.argmin(prof_B)]*1e-3, C_chi[np.argmin(prof_C)]*1e3, 
               'x', color='red', markersize=8, label=f'Min χ²: {chi2_min:.2f}')
    
    # Add horizontal and vertical lines at confidence intervals
    ax_main.axhline(C_chi[C_sx]*1e3, color='gray', ls='--', alpha=0.7)
    ax_main.axhline(C_chi[C_dx]*1e3, color='gray', ls='--', alpha=0.7)
    ax_main.axvline(B_chi[B_sx]*1e-3, color='gray', ls='--', alpha=0.7)
    ax_main.axvline(B_chi[B_dx]*1e-3, color='gray', ls='--', alpha=0.7)
    
    # B profile plot
    ax_right.plot(B_chi*1e-3, prof_C, '-', color='#1f77b4')
    ax_right.axhline(chi2_min+1, color='red', ls='--', alpha=0.7, 
                   label='χ²+1')
    ax_right.axvline(B_chi[B_sx]*1e-3, color='gray', ls='--')
    ax_right.axvline(B_chi[B_dx]*1e-3, color='gray', ls='--')
    ax_right.set_xlabel('Frequency (kHz)', fontsize=12)
    ax_right.set_ylabel('χ²', fontsize=10)
    
    # C profile plot
    ax_top.plot(prof_B, C_chi*1e3, '-', color='#1f77b4')
    ax_top.axvline(chi2_min+1, color='red', ls='--', alpha=0.7)
    ax_top.axhline(C_chi[C_sx]*1e3, color='gray', ls='--')
    ax_top.axhline(C_chi[C_dx]*1e3, color='gray', ls='--')
    ax_top.set_ylabel('Time Constant τ (ms)', fontsize=12)
    
    # Make y-axis of top plot on the right
    ax_top.yaxis.tick_right()
    ax_top.yaxis.set_label_position("right")
    
    # Set limits and invert x-axis for top plot to match
    ax_top.set_xlim(chi2_min-1, chi2_min+10)
    
    # Title and colorbar
    fig.suptitle('χ² Analysis: Frequency vs. Time Constant', fontsize=16)
    cbar = plt.colorbar(cf, ax=ax_main, shrink=0.8)
    cbar.set_label('χ²', fontsize=12)
    
    # Add legend
    ax_main.legend(loc='upper right')
    
    plt.savefig(f'{file}_chi2_heatmap.png')
    return fig

# Example of using the functions (assuming the rest of your code's flow)
# This would replace your original plotting code
"""
# After first fit:
popt, pcov = curve_fit(fitf, tempo, Vout, p0=[Ainit, Binit, Cinit, v0init, t0init], 
                      method='lm', sigma=eVout, absolute_sigma=True)
plot_first_fit(popt)

# After second fit:
popt, pcov = curve_fit(fitf2, tempo, Vout, p0=[Ainit, Binit, Cinit], 
                      method='lm', sigma=eVout, absolute_sigma=True)
residuA = Vout - fitf2(tempo, *popt)
chisq = np.sum((residuA/eVout)**2)
plot_second_fit(popt, chisq)

# After chi-squared calculation:
plot_chi2_heatmap(chi2_min, B_chi, C_chi, chi2D, prof_B, prof_C, B_dx, B_sx, C_dx, C_sx)
"""