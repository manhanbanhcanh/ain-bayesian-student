#!/usr/bin/env python3
"""
Create visualizations for the sensitivity experiment results
from testing @starter/student_core.py
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Set up matplotlib for better-looking plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11

# Data from the sensitivity experiment
states = ['Fail', 'Pass', 'Distinction']
baseline = [0.25, 0.55, 0.20]
modified = [0.05, 0.15, 0.80]
absolute_drift = [0.20, 0.40, 0.60]  # Already absolute values
kl_divergence = 0.8397062471561905

# Colors
colors = {
    'baseline': '#3498db',   # Blue
    'modified': '#e74c3c',   # Red
    'drift': '#f39c12',      # Orange
    'fail': '#e74c3c',       # Red
    'pass': '#f39c12',       # Orange
    'distinction': '#2ecc71' # Green
}

def create_grouped_bar_chart():
    """Create grouped bar chart comparing baseline vs modified posteriors"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(states))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline Posterior',
                   color=colors['baseline'], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, modified, width, label='Modified Posterior',
                   color=colors['modified'], alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Performance Outcome', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Bayesian Network Sensitivity Analysis\nBaseline vs Modified Posterior Probabilities',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Set y-limit to make room for labels
    ax.set_ylim(0, max(max(baseline), max(modified)) * 1.15)

    plt.tight_layout()
    plt.savefig('sensitivity_grouped_barchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: sensitivity_grouped_barchart.png")

def create_pie_charts():
    """Create pie charts for baseline and modified distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Baseline pie chart
    wedges1, texts1, autotexts1 = ax1.pie(baseline, labels=states, autopct='%1.1f%%',
                                          startangle=90, colors=[colors['fail'], colors['pass'], colors['distinction']],
                                          explode=(0.05, 0.05, 0.05), shadow=True)
    ax1.set_title('Baseline Distribution\n(AssignmentCompletion=Medium, ExamDifficulty=Medium)',
                  fontsize=12, fontweight='bold', pad=20)

    # Modified pie chart
    wedges2, texts2, autotexts2 = ax2.pie(modified, labels=states, autopct='%1.1f%%',
                                          startangle=90, colors=[colors['fail'], colors['pass'], colors['distinction']],
                                          explode=(0.05, 0.05, 0.05), shadow=True)
    ax2.set_title('Modified Distribution\n(Performance CPT Modified)',
                  fontsize=12, fontweight='bold', pad=20)

    # Make percentage text bold
    for autotext in autotexts1 + autotexts2:
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    plt.suptitle('Bayesian Network Sensitivity Analysis\nProbability Distributions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sensitivity_pie_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: sensitivity_pie_charts.png")

def create_drift_bar_chart():
    """Create bar chart showing absolute probability drift"""
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(states, absolute_drift, color=colors['drift'], alpha=0.8,
                  edgecolor='black', linewidth=1)

    ax.set_xlabel('Performance Outcome', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Probability Drift', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Probability Drift by Performance State\n|Modified - Baseline|',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    # Add annotation for KL divergence
    ax.text(0.02, 0.98, f'KL Divergence: {kl_divergence:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('sensitivity_drift_barchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: sensitivity_drift_barchart.png")

def create_summary_dashboard():
    """Create a summary dashboard with all key information"""
    fig = plt.figure(figsize=(12, 10))

    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top left: Grouped bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(states))
    width = 0.35
    ax1.bar(x - width/2, baseline, width, label='Baseline', color=colors['baseline'], alpha=0.8)
    ax1.bar(x + width/2, modified, width, label='Modified', color=colors['modified'], alpha=0.8)
    ax1.set_xlabel('Performance Outcome')
    ax1.set_ylabel('Probability')
    ax1.set_title('Posterior Probabilities Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(states)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top right: Pie charts side by side
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('equal')
    wedges, texts, autotexts = ax2.pie(baseline, labels=states, autopct='%1.1f%%',
                                      startangle=90, colors=[colors['fail'], colors['pass'], colors['distinction']])
    ax2.set_title('Baseline Distribution')
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    # Bottom left: Modified pie chart
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('equal')
    wedges2, texts2, autotexts2 = ax3.pie(modified, labels=states, autopct='%1.1f%%',
                                         startangle=90, colors=[colors['fail'], colors['pass'], colors['distinction']])
    ax3.set_title('Modified Distribution')
    for autotext in autotexts2:
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    # Bottom right: Drift bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.bar(states, absolute_drift, color=colors['drift'], alpha=0.8)
    ax4.set_xlabel('Performance Outcome')
    ax4.set_ylabel('Absolute Drift')
    ax4.set_title('Absolute Probability Drift')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on drift bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    # Add overall title and info
    fig.suptitle('Bayesian Network Sensitivity Analysis Dashboard\nTesting @starter/student_core.py Implementation',
                 fontsize=16, fontweight='bold')

    # Add text box with experiment details
    info_text = (
        f"Experiment: P05-CPT-SENSITIVITY\n"
        f"Scenario: AssignmentCompletion=Medium, ExamDifficulty=Medium → Performance\n"
        f"Modification: Changed CPT row Performance|Medium|Medium\n"
        f"  From: [0.25, 0.55, 0.20] → To: [0.05, 0.15, 0.80]\n\n"
        f"Results:\n"
        f"  KL Divergence: {kl_divergence:.3f}\n"
        f"  Total Variation Distance: {sum(absolute_drift):.3f}\n"
        f"  Max Absolute Drift: {max(absolute_drift):.3f} (Distinction)"
    )

    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.savefig('sensitivity_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: sensitivity_dashboard.png")

def main():
    """Main function to create all visualizations"""
    print("Creating visualizations for sensitivity experiment results...")
    print("=" * 60)

    # Create output directory if it doesn't exist
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Change to output directory
    original_dir = os.getcwd()
    os.chdir(output_dir)

    try:
        create_grouped_bar_chart()
        create_pie_charts()
        create_drift_bar_chart()
        create_summary_dashboard()

        print("=" * 60)
        print("All visualizations created successfully!")
        print(f"Files saved in: {os.path.join(original_dir, output_dir)}")
        print("\nGenerated files:")
        for file in os.listdir('.'):
            if file.endswith('.png'):
                print(f"  - {file}")

    finally:
        # Change back to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()