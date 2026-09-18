"""
BDD100k Image Quality Statistics Analysis & Visualization Engine
Generates:
  1. Statistical distributions (Mean, Median, Std, IQR, Skewness)
  2. Non-parametric hypothesis tests (Mann-Whitney U, Kolmogorov-Smirnov) and Cohen's d effect sizes
  3. High-resolution figures:
     - day_vs_night_distributions.png
     - clear_vs_rain_distributions.png
     - visual_correlations.png
     - extreme_samples_gallery.png
  4. Structured CSV and JSON tables for reporting
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import cv2

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

WORKSPACE_DIR = Path("g:/project")
RESULTS_DIR = WORKSPACE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "bdd100k_10k_image_quality.csv"

# Metric definitions
METRICS = [
    {
        "col": "mean_luminance",
        "name": "Mean Luminance (Brightness)",
        "symbol": r"($\mu_L$)",
        "unit": "Intensity [0-255]",
        "xlim": (0, 255)
    },
    {
        "col": "contrast_std",
        "name": "Luminance Std Dev (Contrast)",
        "symbol": r"($\sigma_L$)",
        "unit": "Intensity Std [0-128]",
        "xlim": (0, 100)
    },
    {
        "col": "blur_laplacian_var",
        "name": "Laplacian Variance (Sharpness/Blur)",
        "symbol": r"($\mathrm{Var}(\nabla^2 L)$)",
        "unit": "Variance",
        "xlim": (0, 1500)
    },
    {
        "col": "underexposed_pct",
        "name": "Underexposed Pixels",
        "symbol": "(L <= 10)",
        "unit": "Percentage [%]",
        "xlim": (0, 35)
    },
    {
        "col": "overexposed_pct",
        "name": "Overexposed Pixels",
        "symbol": "(L >= 245)",
        "unit": "Percentage [%]",
        "xlim": (0, 15)
    }
]



def compute_group_stats(df, group_name="all"):
    """Computes comprehensive statistics for each metric."""
    records = []
    for m in METRICS:
        col = m["col"]
        vals = df[col].dropna()
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        p05, p95 = np.percentile(vals, [5, 95])
        iqr = q75 - q25
        mean = vals.mean()
        std = vals.std()
        skew = vals.skew()
        
        records.append({
            "group": group_name,
            "metric": m["name"],
            "metric_col": col,
            "count": len(vals),
            "mean": round(float(mean), 3),
            "std": round(float(std), 3),
            "median": round(float(q50), 3),
            "iqr": round(float(iqr), 3),
            "q25": round(float(q25), 3),
            "q75": round(float(q75), 3),
            "p05": round(float(p05), 3),
            "p95": round(float(p95), 3),
            "min": round(float(vals.min()), 3),
            "max": round(float(vals.max()), 3),
            "skewness": round(float(skew), 3)
        })
    return pd.DataFrame(records)


def perform_hypothesis_tests(df_a, df_b, group_a_name, group_b_name):
    """Computes Mann-Whitney U, KS-test, and Cohen's d."""
    test_results = {}
    for m in METRICS:
        col = m["col"]
        a = df_a[col].dropna().values
        b = df_b[col].dropna().values
        
        # Mann-Whitney U test
        mwu_stat, mwu_p = stats.mannwhitneyu(a, b, alternative='two-sided')
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(a, b)
        
        # Cohen's d effect size
        n1, n2 = len(a), len(b)
        s_pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
        cohens_d = (np.mean(a) - np.mean(b)) / s_pooled if s_pooled > 0 else 0.0
        
        test_results[col] = {
            "metric_name": m["name"],
            f"{group_a_name}_mean": round(float(np.mean(a)), 3),
            f"{group_b_name}_mean": round(float(np.mean(b)), 3),
            f"{group_a_name}_median": round(float(np.median(a)), 3),
            f"{group_b_name}_median": round(float(np.median(b)), 3),
            "cohens_d": round(float(cohens_d), 4),
            "mann_whitney_u": float(mwu_stat),
            "mann_whitney_p": float(mwu_p),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p": float(ks_p)
        }
    return test_results


def plot_comparison(df, group_col, val_a, val_b, label_a, label_b, color_a, color_b, filename, title_prefix):
    """
    Creates a multi-panel publication quality comparison figure (5 subplots).
    Each subplot includes:
      - Kernel Density Estimate (KDE) curve
      - Marginal boxplot inset
      - Key summary markers (mean, median)
    """
    df_a = df[df[group_col] == val_a]
    df_b = df[df[group_col] == val_b]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 5 metrics
    for i, m in enumerate(METRICS):
        ax = axes[i]
        col = m["col"]
        xlim = m["xlim"]
        
        data_a = df_a[col].dropna()
        data_b = df_b[col].dropna()
        
        # Clip to xlim for clean KDE rendering
        data_a_clipped = data_a[data_a <= xlim[1] * 1.05]
        data_b_clipped = data_b[data_b <= xlim[1] * 1.05]
        
        # Histograms + KDE
        bins = np.linspace(xlim[0], xlim[1], 45)
        ax.hist(data_a, bins=bins, density=True, alpha=0.35, color=color_a, label=f"{label_a} (N={len(data_a)})")
        ax.hist(data_b, bins=bins, density=True, alpha=0.35, color=color_b, label=f"{label_b} (N={len(data_b)})")
        
        # KDE lines
        if len(data_a_clipped) > 1 and np.var(data_a_clipped) > 1e-6:
            kde_a = stats.gaussian_kde(data_a_clipped)
            x_grid = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(x_grid, kde_a(x_grid), color=color_a, lw=2.4)
            
        if len(data_b_clipped) > 1 and np.var(data_b_clipped) > 1e-6:
            kde_b = stats.gaussian_kde(data_b_clipped)
            x_grid = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(x_grid, kde_b(x_grid), color=color_b, lw=2.4, ls="--")
        
        # Vertical mean markers
        ax.axvline(data_a.mean(), color=color_a, linestyle='-', lw=1.5, alpha=0.8,
                   label=f"{label_a} Mean: {data_a.mean():.1f}")
        ax.axvline(data_b.mean(), color=color_b, linestyle='--', lw=1.5, alpha=0.8,
                   label=f"{label_b} Mean: {data_b.mean():.1f}")
        
        ax.set_title(f"{m['name']} {m['symbol']}", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel(m["unit"], fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.set_xlim(xlim[0], xlim[1])
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 6: Summary Boxplot overview of normalized/standardized differences
    ax_box = axes[5]
    box_data_a = []
    box_data_b = []
    labels = []
    for m in METRICS:
        col = m["col"]
        # Normalize by pooled std for side-by-side relative scale
        pooled_std = df[col].std() if df[col].std() > 0 else 1.0
        box_data_a.append((df_a[col] - df[col].mean()) / pooled_std)
        box_data_b.append((df_b[col] - df[col].mean()) / pooled_std)
        labels.append(m["symbol"])
    
    pos_a = np.arange(len(METRICS)) * 2.0 - 0.35
    pos_b = np.arange(len(METRICS)) * 2.0 + 0.35
    
    bp_a = ax_box.boxplot(box_data_a, positions=pos_a, widths=0.5, patch_artist=True,
                          boxprops=dict(facecolor=color_a, alpha=0.6),
                          medianprops=dict(color='black', lw=1.5), showfliers=False)
    bp_b = ax_box.boxplot(box_data_b, positions=pos_b, widths=0.5, patch_artist=True,
                          boxprops=dict(facecolor=color_b, alpha=0.6),
                          medianprops=dict(color='black', lw=1.5), showfliers=False)
    
    ax_box.set_xticks(np.arange(len(METRICS)) * 2.0)
    ax_box.set_xticklabels([m["col"].replace("_pct", " %").replace("_laplacian_var", "").replace("_std", "") for m in METRICS], fontsize=9, rotation=15)
    ax_box.set_ylabel("Standardized Z-Score Difference", fontsize=10)
    ax_box.set_title("Standardized Comparison Across Visual Properties", fontsize=12, fontweight='bold', pad=8)
    ax_box.axhline(0, color='gray', linestyle=':', lw=1.0)
    ax_box.legend([bp_a["boxes"][0], bp_b["boxes"][0]], [label_a, label_b], loc='upper right', fontsize=9)
    ax_box.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle(f"BDD100k Visual Properties Distribution: {title_prefix}", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_correlations(df):
    """Generates cross-metric visual correlation scatter and density maps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. Brightness vs Contrast
    ax = axes[0, 0]
    scatter = ax.scatter(df['mean_luminance'], df['contrast_std'], c=df['underexposed_pct'],
                         cmap='viridis', alpha=0.45, s=16, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Underexposed %', fontsize=9)
    ax.set_title("Luminance (Brightness) vs Contrast", fontsize=12, fontweight='bold')
    ax.set_xlabel("Mean Luminance", fontsize=10)
    ax.set_ylabel("Contrast Std Dev", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Brightness vs Blur (Laplacian Variance)
    ax = axes[0, 1]
    # Filter night vs day for coloring
    day_mask = df['timeofday'] == 'daytime'
    night_mask = df['timeofday'] == 'night'
    ax.scatter(df.loc[day_mask, 'mean_luminance'], df.loc[day_mask, 'blur_laplacian_var'],
               c='#D97706', label=f'Daytime (N={day_mask.sum()})', alpha=0.35, s=14, edgecolors='none')
    ax.scatter(df.loc[night_mask, 'mean_luminance'], df.loc[night_mask, 'blur_laplacian_var'],
               c='#1E3A8A', label=f'Night (N={night_mask.sum()})', alpha=0.7, s=20, edgecolors='none')
    ax.set_title("Brightness vs Sharpness / Blur", fontsize=12, fontweight='bold')
    ax.set_xlabel("Mean Luminance", fontsize=10)
    ax.set_ylabel("Laplacian Variance", fontsize=10)
    ax.set_ylim(0, 1500)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 3. Contrast vs Underexposed %
    ax = axes[1, 0]
    ax.scatter(df['contrast_std'], df['underexposed_pct'], c='#0D9488', alpha=0.4, s=16, edgecolors='none')
    ax.set_title("Contrast vs Underexposed Pixel Percentage", fontsize=12, fontweight='bold')
    ax.set_xlabel("Contrast Std Dev", fontsize=10)
    ax.set_ylabel("Underexposed % (L <= 10)", fontsize=10)
    ax.set_ylim(0, 40)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 4. Clear vs Rain Blur & Overexposure Comparison
    ax = axes[1, 1]
    clear_mask = df['weather'] == 'clear'
    rain_mask = df['weather'] == 'rainy'
    ax.scatter(df.loc[clear_mask, 'blur_laplacian_var'], df.loc[clear_mask, 'overexposed_pct'],
               c='#0284C7', label=f'Clear (N={clear_mask.sum()})', alpha=0.4, s=16, edgecolors='none')
    ax.scatter(df.loc[rain_mask, 'blur_laplacian_var'], df.loc[rain_mask, 'overexposed_pct'],
               c='#DC2626', label=f'Rain (N={rain_mask.sum()})', alpha=0.65, s=20, edgecolors='none')
    ax.set_title("Blur vs Overexposure (Clear vs Rain)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Laplacian Variance", fontsize=10)
    ax.set_ylabel("Overexposed % (L >= 245)", fontsize=10)
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("BDD100k Image Quality Property Cross-Correlations", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = FIGURES_DIR / "visual_correlations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_extreme_samples_gallery(df):
    """
    Identifies and plots 8 visual exemplar images representing quality extremes:
      1. Minimum Brightness (Darkest scene)
      2. Maximum Brightness (Most illuminated scene / glare)
      3. Minimum Contrast (Flattest dynamic range)
      4. Maximum Contrast (Deepest shadows / bright highlights)
      5. Maximum Blur (Lowest Laplacian variance)
      6. Maximum Sharpness (Highest Laplacian variance)
      7. Maximum Underexposure (Severe black clipping)
      8. Maximum Overexposure (Severe highlight clipping)
    """
    exemplars = [
        ("Min Brightness", df.sort_values("mean_luminance").iloc[0]),
        ("Max Brightness", df.sort_values("mean_luminance", ascending=False).iloc[0]),
        ("Min Contrast", df.sort_values("contrast_std").iloc[0]),
        ("Max Contrast", df.sort_values("contrast_std", ascending=False).iloc[0]),
        ("Max Blur (Lowest Var)", df.sort_values("blur_laplacian_var").iloc[0]),
        ("Max Sharpness (Highest Var)", df.sort_values("blur_laplacian_var", ascending=False).iloc[0]),
        ("Max Underexposure", df.sort_values("underexposed_pct", ascending=False).iloc[0]),
        ("Max Overexposure", df.sort_values("overexposed_pct", ascending=False).iloc[0]),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (title, row) in enumerate(exemplars):
        ax = axes[i]
        img_rel_path = row['file_path']
        img_full_path = WORKSPACE_DIR / img_rel_path
        
        img_bgr = cv2.imread(str(img_full_path))
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        else:
            ax.text(0.5, 0.5, "Image Missing", ha='center', va='center')
            
        ax.set_title(f"[{title}]\n{row['image_name']}", fontsize=10, fontweight='bold', pad=6)
        ax.axis('off')
        
        caption = (rf"$\mu_L$: {row['mean_luminance']:.1f} | $\sigma_L$: {row['contrast_std']:.1f}" + "\n"
                   f"Blur: {row['blur_laplacian_var']:.1f} | Under: {row['underexposed_pct']:.1f}%\n"
                   f"Over: {row['overexposed_pct']:.1f}% | {row['timeofday']}, {row['weather']}")
        ax.text(0.5, -0.16, caption, transform=ax.transAxes, ha='center', va='top',
                fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3F4F6', edgecolor='#D1D5DB'))

    
    plt.suptitle("BDD100k Quality Extremes Gallery (Exemplar Artifacts & Conditions)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    output_path = FIGURES_DIR / "extreme_samples_gallery.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {output_path}")


def main():
    print("=" * 70)
    print("Starting BDD100k Statistical Analysis & Figure Generation")
    print("=" * 70)
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} image records from {CSV_PATH.name}")
    
    # 1. Overall Statistics
    df_overall_stats = compute_group_stats(df, "Overall Dataset")
    overall_csv = RESULTS_DIR / "summary_table_overall.csv"
    df_overall_stats.to_csv(overall_csv, index=False)
    print(f"Saved overall summary table: {overall_csv.name}")
    
    # 2. Day vs Night Analysis
    df_day = df[df['timeofday'] == 'daytime']
    df_night = df[df['timeofday'] == 'night']
    print(f"\nTime of Day: Daytime N={len(df_day)}, Night N={len(df_night)}")
    
    stats_day = compute_group_stats(df_day, "Daytime")
    stats_night = compute_group_stats(df_night, "Night")
    df_day_night = pd.concat([stats_day, stats_night], ignore_index=True)
    day_night_csv = RESULTS_DIR / "summary_table_day_vs_night.csv"
    df_day_night.to_csv(day_night_csv, index=False)
    print(f"Saved Day vs Night summary table: {day_night_csv.name}")
    
    tests_day_night = perform_hypothesis_tests(df_day, df_night, "daytime", "night")
    
    # 3. Clear vs Rain Analysis
    df_clear = df[df['weather'] == 'clear']
    df_rain = df[df['weather'] == 'rainy']
    print(f"Weather: Clear N={len(df_clear)}, Rainy N={len(df_rain)}")
    
    stats_clear = compute_group_stats(df_clear, "Clear")
    stats_rain = compute_group_stats(df_rain, "Rainy")
    df_clear_rain = pd.concat([stats_clear, stats_rain], ignore_index=True)
    clear_rain_csv = RESULTS_DIR / "summary_table_clear_vs_rain.csv"
    df_clear_rain.to_csv(clear_rain_csv, index=False)
    print(f"Saved Clear vs Rain summary table: {clear_rain_csv.name}")
    
    tests_clear_rain = perform_hypothesis_tests(df_clear, df_rain, "clear", "rainy")
    
    # Save test results
    tests_output = {
        "day_vs_night": tests_day_night,
        "clear_vs_rain": tests_clear_rain
    }
    tests_json = RESULTS_DIR / "statistical_tests.json"
    with open(tests_json, "w", encoding="utf-8") as f:
        json.dump(tests_output, f, indent=4)
    print(f"Saved statistical tests: {tests_json.name}")
    
    # 4. Generate Visualizations
    print("\nGenerating publication-quality figures...")
    
    # Day vs Night
    plot_comparison(
        df=df,
        group_col="timeofday",
        val_a="daytime",
        val_b="night",
        label_a="Daytime",
        label_b="Night",
        color_a="#D97706", # Amber / gold
        color_b="#1E3A8A", # Deep navy / indigo
        filename="day_vs_night_distributions.png",
        title_prefix="Daytime vs. Night Condition Contrast"
    )
    
    # Clear vs Rain
    plot_comparison(
        df=df,
        group_col="weather",
        val_a="clear",
        val_b="rainy",
        label_a="Clear Weather",
        label_b="Rainy Weather",
        color_a="#0284C7", # Clear Sky Blue
        color_b="#475569", # Rainy Slate
        filename="clear_vs_rain_distributions.png",
        title_prefix="Clear Weather vs. Rainy Weather Degradation"
    )
    
    # Cross Correlations
    plot_correlations(df)
    
    # Extremes Gallery
    plot_extreme_samples_gallery(df)
    
    print("\nAll analyses and figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
