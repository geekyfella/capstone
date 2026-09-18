"""
BDD100k Presentation-Ready Scientific Illustrations Generator
Produces publication and presentation grade figures in both SVG and JPG formats:
  1. Correlation Heatmap (figure1_correlation_heatmap)
  2. Environmental Histograms & Density Curves (figure2_environmental_histograms)
  3. Cumulative Exposure & Dynamic Range Line Chart (figure3_luminance_cdf_linechart)
  4. Environmental Domain Benchmark Bar Graph with Error Bars (figure4_domain_benchmark_bargraph)
  5. Unified Master 16:9 Presentation Slide Dashboard (figure5_master_presentation_dashboard)
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle, FancyBboxPatch
from scipy import stats

# Paths
WORKSPACE_DIR = Path("g:/project")
RESULTS_DIR = WORKSPACE_DIR / "results"
CSV_PATH = RESULTS_DIR / "bdd100k_10k_image_quality.csv"
PRESENTATION_DIR = RESULTS_DIR / "presentation"
PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\54142845-452a-4a5f-b683-561f1abbab49")

# Presentation Style Config
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#D1D5DB'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.facecolor'] = '#FFFFFF'
plt.rcParams['axes.facecolor'] = '#FFFFFF'


def save_dual_format(fig, base_name):
    """Saves a matplotlib figure in both SVG and high-resolution JPG formats."""
    svg_path = PRESENTATION_DIR / f"{base_name}.svg"
    jpg_path = PRESENTATION_DIR / f"{base_name}.jpg"
    
    # Save SVG (scalable vector format)
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    # Save JPG (high-res 300 DPI for slides)
    fig.savefig(jpg_path, format='jpg', dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})
    
    # Copy to artifact dir
    if ARTIFACT_DIR.exists():
        shutil.copy2(svg_path, ARTIFACT_DIR / f"{base_name}.svg")
        shutil.copy2(jpg_path, ARTIFACT_DIR / f"{base_name}.jpg")
        
    print(f"Saved: {svg_path.name} and {jpg_path.name}")


# ==============================================================================
# 1. CORRELATION HEATMAP
# ==============================================================================
def generate_correlation_heatmap(df):
    cols = [
        ('mean_luminance', 'Mean Luminance (μL)'),
        ('luminance_median', 'Median Luminance'),
        ('contrast_std', 'Contrast Std Dev (σL)'),
        ('blur_laplacian_var', 'Laplacian Blur Var'),
        ('underexposed_pct', 'Underexposed % (L≤10)'),
        ('overexposed_pct', 'Overexposed % (L≥245)'),
    ]
    col_keys = [c[0] for c in cols]
    labels = [c[1] for c in cols]
    
    sub_df = df[col_keys].dropna()
    corr = sub_df.corr(method='spearman').values
    
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1.0, vmax=1.0)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Spearman Rank Correlation Coefficient (r)", fontsize=11, fontweight='bold', labelpad=10)
    
    # Ticks & Labels
    n = len(cols)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10, fontweight='medium')
    ax.set_yticklabels(labels, fontsize=10, fontweight='medium')
    
    # Grid lines between cells
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    # Annotate numbers
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            txt_color = 'white' if abs(val) > 0.45 else '#1F2937'
            weight = 'bold' if abs(val) > 0.5 else 'normal'
            ax.text(j, i, f"{val:+.2f}", ha='center', va='center',
                    color=txt_color, fontsize=11, fontweight=weight)
            
    ax.set_title("BDD100k Image Quality Feature Correlation Heatmap\nSpearman Inter-Property Cross-Correlation Matrix",
                 fontsize=13, fontweight='bold', pad=16)
    
    save_dual_format(fig, "figure1_correlation_heatmap")
    plt.close(fig)


# ==============================================================================
# 2. ENVIRONMENTAL HISTOGRAMS & KDE
# ==============================================================================
def generate_environmental_histograms(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    axes = axes.flatten()
    
    day = df[df['timeofday'] == 'daytime']
    night = df[df['timeofday'] == 'night']
    clear = df[df['weather'] == 'clear']
    rain = df[df['weather'] == 'rainy']
    
    configs = [
        {
            'ax': axes[0],
            'col': 'mean_luminance',
            'title': 'A. Brightness Distribution: Day vs. Night',
            'xlabel': 'Mean Luminance (μL) [0–255]',
            'xlim': (0, 200),
            'group_a': (day, 'Daytime (N=4,190)', '#D97706'),
            'group_b': (night, 'Night (N=202)', '#1E3A8A'),
        },
        {
            'ax': axes[1],
            'col': 'blur_laplacian_var',
            'title': 'B. Sharpness / Blur Distribution: Day vs. Night',
            'xlabel': 'Laplacian Variance (Var(∇²L))',
            'xlim': (0, 1200),
            'group_a': (day, 'Daytime (N=4,190)', '#D97706'),
            'group_b': (night, 'Night (N=202)', '#1E3A8A'),
        },
        {
            'ax': axes[2],
            'col': 'underexposed_pct',
            'title': 'C. Underexposure Distribution: Day vs. Night',
            'xlabel': 'Underexposed Pixel Percentage (L ≤ 10) [%]',
            'xlim': (0, 70),
            'group_a': (day, 'Daytime (N=4,190)', '#D97706'),
            'group_b': (night, 'Night (N=202)', '#1E3A8A'),
        },
        {
            'ax': axes[3],
            'col': 'overexposed_pct',
            'title': 'D. Overexposure Distribution: Clear vs. Rain',
            'xlabel': 'Overexposed Pixel Percentage (L ≥ 245) [%]',
            'xlim': (0, 15),
            'group_a': (clear, 'Clear (N=1,591)', '#0284C7'),
            'group_b': (rain, 'Rainy (N=373)', '#DC2626'),
        }
    ]
    
    for cfg in configs:
        ax = cfg['ax']
        col = cfg['col']
        xlim = cfg['xlim']
        
        df_a, label_a, color_a = cfg['group_a']
        df_b, label_b, color_b = cfg['group_b']
        
        vals_a = df_a[col].dropna()
        vals_b = df_b[col].dropna()
        
        bins = np.linspace(xlim[0], xlim[1], 35)
        
        # Histograms
        ax.hist(vals_a, bins=bins, density=True, alpha=0.35, color=color_a, label=label_a, edgecolor='white')
        ax.hist(vals_b, bins=bins, density=True, alpha=0.35, color=color_b, label=label_b, edgecolor='white')
        
        # KDE lines
        grid = np.linspace(xlim[0], xlim[1], 250)
        kde_a = stats.gaussian_kde(vals_a[vals_a <= xlim[1] * 1.1])
        kde_b = stats.gaussian_kde(vals_b[vals_b <= xlim[1] * 1.1])
        
        ax.plot(grid, kde_a(grid), color=color_a, lw=2.2)
        ax.plot(grid, kde_b(grid), color=color_b, lw=2.2, ls='--')
        
        # Mean dashed lines
        ax.axvline(vals_a.mean(), color=color_a, ls='-', lw=1.5, alpha=0.9, label=f"Mean {vals_a.mean():.1f}")
        ax.axvline(vals_b.mean(), color=color_b, ls='--', lw=1.5, alpha=0.9, label=f"Mean {vals_b.mean():.1f}")
        
        ax.set_title(cfg['title'], fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(cfg['xlabel'], fontsize=9.5)
        ax.set_ylabel("Probability Density", fontsize=9.5)
        ax.set_xlim(xlim[0], xlim[1])
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    plt.suptitle("BDD100k Environmental Quality Distributions\nComparative Histograms & Kernel Density Estimation (KDE)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_dual_format(fig, "figure2_environmental_histograms")
    plt.close(fig)


# ==============================================================================
# 3. LUMINANCE CDF LINE CHART
# ==============================================================================
def generate_luminance_cdf_linechart(df):
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    day = df[df['timeofday'] == 'daytime']['mean_luminance'].dropna()
    night = df[df['timeofday'] == 'night']['mean_luminance'].dropna()
    clear = df[df['weather'] == 'clear']['mean_luminance'].dropna()
    rain = df[df['weather'] == 'rainy']['mean_luminance'].dropna()
    overall = df['mean_luminance'].dropna()
    
    groups = [
        (day, 'Daytime (N=4,190)', '#D97706', '-', 2.5),
        (night, 'Night (N=202)', '#1E3A8A', '-', 3.0),
        (clear, 'Clear Weather (N=1,591)', '#0284C7', '-.', 2.0),
        (rain, 'Rainy Weather (N=373)', '#DC2626', '--', 2.0),
        (overall, 'Overall Benchmark (N=8,027)', '#4B5563', ':', 2.0),
    ]
    
    x_eval = np.linspace(0, 255, 500)
    
    for vals, label, color, ls, lw in groups:
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.step(sorted_vals, cdf, label=label, color=color, linestyle=ls, linewidth=lw, where='post')
        
    # Shaded Exposure Critical Threshold Zones
    ax.axvspan(0, 10, color='#DC2626', alpha=0.15, label='Underexposure Zone (L ≤ 10)')
    ax.axvspan(245, 255, color='#F59E0B', alpha=0.2, label='Overexposure Zone (L ≥ 245)')
    
    # 50th Percentile (Median) Guide Line
    ax.axhline(0.5, color='#9CA3AF', linestyle='--', linewidth=1.0)
    ax.text(250, 0.51, "Median (50%)", color='#6B7280', fontsize=9, ha='right')
    
    # Annotations
    ax.annotate("Night: 50% of images\nhave μL ≤ 34.4",
                xy=(34.4, 0.5), xytext=(55, 0.65),
                arrowprops=dict(arrowstyle="->", color="#1E3A8A", lw=1.5),
                fontsize=9.5, fontweight='bold', color="#1E3A8A",
                bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#93C5FD"))
    
    ax.annotate("Daytime: 50% of images\nhave μL ≤ 103.3",
                xy=(103.3, 0.5), xytext=(120, 0.35),
                arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.5),
                fontsize=9.5, fontweight='bold', color="#D97706",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#FCD34D"))
    
    ax.set_title("BDD100k Mean Luminance Cumulative Distribution Function (CDF)\nDynamic Range Trajectory Across Environmental Domains",
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel("Mean Luminance (μL) [0–255 Standard 8-bit Scale]", fontsize=11, fontweight='medium')
    ax.set_ylabel("Cumulative Probability P(L ≤ x)", fontsize=11, fontweight='medium')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    save_dual_format(fig, "figure3_luminance_cdf_linechart")
    plt.close(fig)


# ==============================================================================
# 4. GROUPED BENCHMARK BAR GRAPH WITH ERROR BARS
# ==============================================================================
def generate_domain_benchmark_bargraph(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
    axes = axes.flatten()
    
    # Filter domains with robust sample sizes
    domains = [
        ('Daytime', df[df['timeofday'] == 'daytime'], '#F59E0B'),
        ('Night', df[df['timeofday'] == 'night'], '#1E3A8A'),
        ('Clear', df[df['weather'] == 'clear'], '#0284C7'),
        ('Rainy', df[df['weather'] == 'rainy'], '#DC2626'),
        ('Overcast', df[df['weather'] == 'overcast'], '#64748B'),
    ]
    
    names = [d[0] for d in domains]
    colors = [d[2] for d in domains]
    
    metrics_cfg = [
        {
            'ax': axes[0],
            'col': 'mean_luminance',
            'title': 'A. Mean Luminance (Brightness)',
            'ylabel': 'Mean Luminance (μL)',
            'fmt': '%.1f'
        },
        {
            'ax': axes[1],
            'col': 'contrast_std',
            'title': 'B. Contrast (Luminance Std Dev)',
            'ylabel': 'Contrast (σL)',
            'fmt': '%.1f'
        },
        {
            'ax': axes[2],
            'col': 'blur_laplacian_var',
            'title': 'C. Sharpness / Laplacian Variance',
            'ylabel': 'Var(∇²L)',
            'fmt': '%.0f'
        },
        {
            'ax': axes[3],
            'col': 'overexposed_pct',
            'title': 'D. Overexposed Pixels (Specular Saturation)',
            'ylabel': '% Pixels with L ≥ 245',
            'fmt': '%.2f%%'
        },
    ]
    
    for m in metrics_cfg:
        ax = m['ax']
        col = m['col']
        
        means = []
        sems = []
        
        for name, sub, col_color in domains:
            v = sub[col].dropna()
            means.append(v.mean())
            sems.append(stats.sem(v) if len(v) > 1 else 0)
            
        x = np.arange(len(domains))
        bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.85, edgecolor='#1F2937', lw=1.0)
        
        # Value labels above bars
        for bar, mean_val in zip(bars, means):
            h = bar.get_height()
            label_txt = m['fmt'] % mean_val
            ax.text(bar.get_x() + bar.get_width() / 2., h * 1.03 + (ax.get_ylim()[1]*0.01 if ax.get_ylim()[1] else 0.5),
                    label_txt, ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        ax.set_title(m['title'], fontsize=11, fontweight='bold', pad=8)
        ax.set_ylabel(m['ylabel'], fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10, fontweight='medium')
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        # Room for labels
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1] * 1.18)
        
    plt.suptitle("BDD100k Environmental Domain Benchmarks\nCondition Comparison Across Visual Quality Metrics (Mean ± SEM)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_dual_format(fig, "figure4_domain_benchmark_bargraph")
    plt.close(fig)


# ==============================================================================
# 5. MASTER 16:9 PRESENTATION DASHBOARD
# ==============================================================================
def generate_master_presentation_dashboard(df):
    """
    Creates a unified 16:9 executive presentation slide combining:
      - Top-Left: Correlation Heatmap
      - Top-Right: Brightness & Sharpness Histograms
      - Bottom-Left: Luminance CDF Dynamic Range
      - Bottom-Right: Environmental Quality Benchmarking Bars
    """
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22, top=0.90, bottom=0.08, left=0.07, right=0.95)
    
    # 1. Top-Left: Heatmap
    ax_heat = fig.add_subplot(gs[0, 0])
    cols = ['mean_luminance', 'contrast_std', 'blur_laplacian_var', 'underexposed_pct', 'overexposed_pct']
    labels = ['Brightness', 'Contrast', 'Sharpness', 'Under %', 'Over %']
    corr = df[cols].corr(method='spearman').values
    im = ax_heat.imshow(corr, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    ax_heat.set_xticks(range(len(cols)))
    ax_heat.set_yticks(range(len(cols)))
    ax_heat.set_xticklabels(labels, fontsize=8.5, rotation=25, ha='right')
    ax_heat.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr[i, j]
            c = 'white' if abs(v) > 0.45 else 'black'
            ax_heat.text(j, i, f"{v:+.2f}", ha='center', va='center', fontsize=8, color=c, fontweight='bold')
    ax_heat.set_title("1. Inter-Feature Correlation Heatmap", fontsize=11, fontweight='bold', pad=8)
    
    # 2. Top-Right: Day vs Night Brightness Histogram
    ax_hist = fig.add_subplot(gs[0, 1])
    day = df[df['timeofday'] == 'daytime']['mean_luminance'].dropna()
    night = df[df['timeofday'] == 'night']['mean_luminance'].dropna()
    bins = np.linspace(0, 200, 30)
    ax_hist.hist(day, bins=bins, density=True, alpha=0.4, color='#D97706', label='Daytime (N=4,190)')
    ax_hist.hist(night, bins=bins, density=True, alpha=0.4, color='#1E3A8A', label='Night (N=202)')
    grid = np.linspace(0, 200, 200)
    ax_hist.plot(grid, stats.gaussian_kde(day)(grid), color='#D97706', lw=2.0)
    ax_hist.plot(grid, stats.gaussian_kde(night)(grid), color='#1E3A8A', lw=2.0, ls='--')
    ax_hist.axvline(day.mean(), color='#D97706', ls='-', lw=1.2, label=f'Day Mean: {day.mean():.1f}')
    ax_hist.axvline(night.mean(), color='#1E3A8A', ls='--', lw=1.2, label=f'Night Mean: {night.mean():.1f}')
    ax_hist.set_title("2. Brightness Histogram: Day vs. Night Shift", fontsize=11, fontweight='bold', pad=8)
    ax_hist.set_xlabel("Mean Luminance (μL)", fontsize=9)
    ax_hist.set_ylabel("Probability Density", fontsize=9)
    ax_hist.legend(fontsize=7.5, loc='upper right')
    ax_hist.grid(True, linestyle=':', alpha=0.5)
    
    # 3. Bottom-Left: Luminance CDF Curves
    ax_cdf = fig.add_subplot(gs[1, 0])
    for vals, l, col, ls in [
        (day, 'Daytime', '#D97706', '-'),
        (night, 'Night', '#1E3A8A', '-'),
        (df[df['weather'] == 'clear']['mean_luminance'].dropna(), 'Clear', '#0284C7', '-.'),
        (df[df['weather'] == 'rainy']['mean_luminance'].dropna(), 'Rainy', '#DC2626', '--')
    ]:
        s = np.sort(vals)
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax_cdf.step(s, cdf, label=l, color=col, linestyle=ls, lw=1.8, where='post')
    ax_cdf.axvspan(0, 10, color='#DC2626', alpha=0.15, label='Crushed Blacks (L≤10)')
    ax_cdf.axvspan(245, 255, color='#F59E0B', alpha=0.2, label='Saturation (L≥245)')
    ax_cdf.set_title("3. Cumulative Luminance & Exposure Trajectory (CDF)", fontsize=11, fontweight='bold', pad=8)
    ax_cdf.set_xlabel("Mean Luminance (μL)", fontsize=9)
    ax_cdf.set_ylabel("Cumulative P(L ≤ x)", fontsize=9)
    ax_cdf.legend(fontsize=7.5, loc='lower right')
    ax_cdf.grid(True, linestyle=':', alpha=0.5)
    
    # 4. Bottom-Right: Environmental Domain Bar Chart
    ax_bar = fig.add_subplot(gs[1, 1])
    groups = [
        ('Daytime', df[df['timeofday'] == 'daytime']['blur_laplacian_var'].dropna(), '#D97706'),
        ('Night', df[df['timeofday'] == 'night']['blur_laplacian_var'].dropna(), '#1E3A8A'),
        ('Clear', df[df['weather'] == 'clear']['blur_laplacian_var'].dropna(), '#0284C7'),
        ('Rainy', df[df['weather'] == 'rainy']['blur_laplacian_var'].dropna(), '#DC2626'),
        ('Overcast', df[df['weather'] == 'overcast']['blur_laplacian_var'].dropna(), '#64748B'),
    ]
    b_names = [g[0] for g in groups]
    b_means = [g[1].mean() for g in groups]
    b_sems = [stats.sem(g[1]) for g in groups]
    b_colors = [g[2] for g in groups]
    x_pos = np.arange(len(b_names))
    b_bars = ax_bar.bar(x_pos, b_means, yerr=b_sems, capsize=4, color=b_colors, alpha=0.85, edgecolor='#1F2937', lw=0.8)
    for b, m_val in zip(b_bars, b_means):
        ax_bar.text(b.get_x() + b.get_width() / 2., b.get_height() * 1.04,
                    f"{m_val:.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_bar.set_title("4. Sharpness Benchmark Across Conditions (Mean Var(∇²L) ± SEM)", fontsize=11, fontweight='bold', pad=8)
    ax_bar.set_ylabel("Laplacian Variance", fontsize=9)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(b_names, fontsize=8.5)
    ax_bar.set_ylim(0, max(b_means) * 1.22)
    ax_bar.grid(True, axis='y', linestyle=':', alpha=0.5)
    
    # Master Header
    fig.suptitle("BDD100k Image Quality Characterization: Scientific Presentation Dashboard\nComprehensive Empirical Evaluation across Brightness, Contrast, Blur, and Exposure (N = 8,027)",
                 fontsize=15, fontweight='bold', y=0.97)
    
    save_dual_format(fig, "figure5_master_presentation_dashboard")
    plt.close(fig)


def main():
    print("=" * 70)
    print("Generating Scientific Presentation Illustrations in SVG and JPG formats")
    print("=" * 70)
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records from {CSV_PATH.name}\n")
    
    # Generate the 5 scientific presentation figures
    print("1. Generating Correlation Heatmap...")
    generate_correlation_heatmap(df)
    
    print("2. Generating Environmental Histograms & KDE...")
    generate_environmental_histograms(df)
    
    print("3. Generating Luminance CDF Line Chart...")
    generate_luminance_cdf_linechart(df)
    
    print("4. Generating Domain Benchmark Bar Graph...")
    generate_domain_benchmark_bargraph(df)
    
    print("5. Generating Master 16:9 Presentation Slide Dashboard...")
    generate_master_presentation_dashboard(df)
    
    print("\nAll presentation illustrations generated successfully in SVG and JPG formats!")
    print(f"Destination: {PRESENTATION_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
