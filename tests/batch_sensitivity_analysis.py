#!/usr/bin/env python3
"""
Batch sensitivity analysis for Bayesian network testing.
Runs 100+ test cases and creates summary visualizations.
Fixed version for matplotlib compatibility.
"""

import json
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, '.')

from starter import loader, student_core

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load all necessary data files"""
    bn_schema = loader.load_bn_schema()
    base_cpt = loader.load_base_cpt()
    scenarios = loader.load_scenarios()
    return bn_schema, base_cpt, scenarios

def create_modified_cpt(base_cpt, modifications):
    """
    Create a modified CPT based on base CPT and modification specifications.

    Args:
        base_cpt: Original CPT dictionary
        modifications: List of (node, key, new_values) tuples

    Returns:
        Modified CPT dictionary
    """
    modified_cpt = copy.deepcopy(base_cpt)

    for node, key, new_values in modifications:
        if node in modified_cpt and key in modified_cpt[node]['table']:
            # Ensure it's a valid probability vector
            new_values = np.array(new_values, dtype=float)
            if np.any(new_values < 0):
                new_values = np.clip(new_values, 0, None)  # Remove negative values
            if np.sum(new_values) > 0:
                new_values = new_values / np.sum(new_values)  # Normalize
            else:
                # If all zeros, uniform distribution
                new_values = np.ones_like(new_values) / len(new_values)

            modified_cpt[node]['table'][key] = new_values.tolist()

    return modified_cpt

def run_sensitivity_test(evidence, query, bn_schema, base_cpt, modified_cpt):
    """Run a single sensitivity test and return results"""
    try:
        # Compute posteriors
        baseline_post = student_core.infer_posterior(evidence, query, bn_schema, base_cpt)
        modified_post = student_core.infer_posterior(evidence, query, bn_schema, modified_cpt)

        # Get states in consistent order
        states = list(baseline_post.keys())
        baseline_vals = [baseline_post[s] for s in states]
        modified_vals = [modified_post[s] for s in states]

        # Calculate metrics
        absolute_drift = [abs(m - b) for b, m in zip(baseline_vals, modified_vals)]
        max_drift = max(absolute_drift)
        max_drift_state = states[absolute_drift.index(max_drift)]

        # KL divergence
        kl_divergence = 0.0
        for b, m in zip(baseline_vals, modified_vals):
            if b > 0 and m > 0:
                kl_divergence += b * np.log(b / m)

        return {
            'success': True,
            'baseline_posterior': dict(zip(states, baseline_vals)),
            'modified_posterior': dict(zip(states, modified_vals)),
            'absolute_drift': dict(zip(states, absolute_drift)),
            'max_drift': max_drift,
            'max_drift_state': max_drift_state,
            'kl_divergence': kl_divergence,
            'total_variation_distance': sum(absolute_drift)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def generate_test_cases(bn_schema, base_cpt, scenarios, num_cases=100):
    """
    Generate 100 test cases across all scenarios with variations.

    Returns:
        List of test case dictionaries
    """
    test_cases = []

    # We'll distribute test cases across scenarios
    cases_per_scenario = num_cases // len(scenarios)
    remainder = num_cases % len(scenarios)

    case_idx = 0

    for scenario_idx, scenario in enumerate(scenarios):
        # Determine how many cases for this scenario
        n_cases_this = cases_per_scenario + (1 if scenario_idx < remainder else 0)

        evidence = scenario['evidence']
        query = scenario['query']
        scenario_id = scenario['id']

        print(f"Generating {n_cases_this} test cases for scenario: {scenario_id}")

        # Generate different types of modifications for this scenario
        for i in range(n_cases_this):
            # Select a node to modify (prefer nodes with parents for more interesting changes)
            nodes_with_parents = [node for node in bn_schema['nodes'] if node['parents']]
            root_nodes = [node for node in bn_schema['nodes'] if not node['parents']]

            # Skip root node modifications for sensitivity analysis as they don't depend on evidence in a meaningful way for this context
            if not nodes_with_parents:
                # If no nodes with parents, we'll have to use root nodes but note the limitation
                target_node = np.random.choice(bn_schema['nodes'])
            else:
                # Mix of modifications: mostly non-root nodes
                if np.random.random() < 0.8 and nodes_with_parents:  # 80% chance to modify non-root node
                    target_node = np.random.choice(nodes_with_parents)
                else:  # 20% chance to modify root node (less informative but still valid)
                    target_node = np.random.choice(bn_schema['nodes'])

            node_name = target_node['name']
            parent_names = target_node['parents']
            states = target_node['states']

            # Determine the key for modification
            if not parent_names:  # Root node
                key = ""
                # For root nodes, we still modify but the sensitivity to evidence will be limited
                # This is okay for demonstrating the mechanics
            else:
                # Try to use evidence values if available, otherwise use parent states from evidence or random
                parent_values = []
                valid_evidence = True
                for parent in parent_names:
                    if parent in evidence:
                        parent_values.append(evidence[parent])
                    else:
                        # Find parent node to get its states
                        parent_node = next((n for n in bn_schema['nodes'] if n['name'] == parent), None)
                        if parent_node:
                            parent_values.append(np.random.choice(parent_node['states']))
                        else:
                            parent_values.append('Unknown')  # Fallback
                        valid_evidence = False  # Mark that we had to guess/default

                key = "|".join(parent_values)

                # If key doesn't exist in CPT, try to find a similar one or use first available
                if key not in base_cpt[node_name]['table']:
                    # Get available keys
                    available_keys = list(base_cpt[node_name]['table'].keys())
                    if available_keys:
                        key = np.random.choice(available_keys)
                    else:
                        continue  # Skip this modification

            # Generate modification type
            mod_type = np.random.choice([
                'increase_first',
                'decrease_first',
                'swap_extremes',
                'uniform_shift',
                'random_dirichlet',
                'zero_one'
            ])

            original_probs = base_cpt[node_name]['table'][key]
            n_states = len(states)

            if mod_type == 'increase_first':
                # Increase first state, decrease last
                new_probs = original_probs.copy()
                if n_states >= 2:
                    transfer = min(0.3, original_probs[-1] * 0.8)  # Transfer up to 30% or 80% of last
                    new_probs[0] += transfer
                    new_probs[-1] -= transfer

            elif mod_type == 'decrease_first':
                # Decrease first state, increase last
                new_probs = original_probs.copy()
                if n_states >= 2:
                    transfer = min(0.3, original_probs[0] * 0.8)
                    new_probs[0] -= transfer
                    new_probs[-1] += transfer

            elif mod_type == 'swap_extremes':
                # Swap first and last states
                new_probs = original_probs.copy()
                if n_states >= 2:
                    new_probs[0], new_probs[-1] = new_probs[-1], new_probs[0]

            elif mod_type == 'uniform_shift':
                # Shift probability uniformly toward one state
                new_probs = original_probs.copy()
                target_state = np.random.randint(n_states)
                total_shift = 0.4
                if original_probs[target_state] < 0.9:  # Can still increase
                    # Take equally from all other states
                    if n_states > 1:
                        shift_per_other = total_shift / (n_states - 1)
                        for j in range(n_states):
                            if j != target_state:
                                new_probs[j] = max(0, original_probs[j] - shift_per_other)
                        new_probs[target_state] = min(1.0, original_probs[target_state] + total_shift * (n_states - 1))

            elif mod_type == 'random_dirichlet':
                # Generate completely new random probabilities
                new_probs = np.random.dirichlet(np.ones(n_states) * 0.5)  # More concentrated than uniform

            elif mod_type == 'zero_one':
                # Make one state very likely, others very unlikely
                new_probs = np.full(n_states, 0.001)  # Small baseline
                dominant_state = np.random.randint(n_states)
                new_probs[dominant_state] = 0.996
                # Renormalize to sum to 1
                new_probs = new_probs / np.sum(new_probs)

            # Ensure valid probabilities
            new_probs = np.array(new_probs)
            new_probs = np.clip(new_probs, 0.001, None)  # Minimum probability
            new_probs = new_probs / np.sum(new_probs)  # Renormalize

            # Create the modification specification
            modifications = [(node_name, key, new_probs.tolist())]

            test_case = {
                'case_id': case_idx,
                'scenario_id': scenario_id,
                'scenario_class': scenario['class'],
                'evidence': evidence.copy(),
                'query': query,
                'modified_node': node_name,
                'modified_key': key,
                'modification_type': mod_type,
                'original_probs': original_probs.copy(),
                'new_probs': new_probs.tolist(),
                'modifications': modifications
            }

            test_cases.append(test_case)
            case_idx += 1

    print(f"Generated {len(test_cases)} total test cases")
    return test_cases

def run_all_tests(test_cases, bn_schema, base_cpt):
    """Run all test cases and collect results"""
    results = []

    print("Running sensitivity tests...")
    for i, test_case in enumerate(test_cases):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(test_cases)}")

        # Create modified CPT
        modified_cpt = create_modified_cpt(base_cpt, test_case['modifications'])

        # Run sensitivity test
        test_result = run_sensitivity_test(
            test_case['evidence'],
            test_case['query'],
            bn_schema,
            base_cpt,
            modified_cpt
        )

        # Combine test case info with results
        combined = {**test_case, **test_result}
        results.append(combined)

    successful = sum(1 for r in results if r.get('success', False))
    print(f"Completed: {successful}/{len(results)} tests successful")

    return results

def create_summary_visualizations(results, output_dir="batch_visualizations"):
    """Create summary visualizations from the test results"""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Filter successful tests
    successful_results = [r for r in results if r.get('success', False)]

    if not successful_results:
        print("No successful tests to visualize")
        return

    print(f"Creating visualizations from {len(successful_results)} successful tests...")

    # Extract metrics for easy plotting
    kl_divergences = [r['kl_divergence'] for r in successful_results]
    max_drifts = [r['max_drift'] for r in successful_results]
    total_variations = [r['total_variation_distance'] for r in successful_results]

    # Group by scenario
    scenario_data = {}
    for result in successful_results:
        scenario = result['scenario_id']
        if scenario not in scenario_data:
            scenario_data[scenario] = []
        scenario_data[scenario].append(result)

    # 1. Distribution Plots (Histograms)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    n, bins, patches = plt.hist(kl_divergences, bins=20, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    plt.xlabel('KL Divergence')
    plt.ylabel('Frequency')
    plt.title('Distribution of KL Divergence\nAcross All Test Cases')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    n, bins, patches = plt.hist(max_drifts, bins=20, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
    plt.xlabel('Maximum Absolute Drift')
    plt.ylabel('Frequency')
    plt.title('Distribution of Maximum Drift\nAcross All Test Cases')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    n, bins, patches = plt.hist(total_variations, bins=20, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=0.5)
    plt.xlabel('Total Variation Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Variation\nDistance Across All Test Cases')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_distribution_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Heatmap: Scenario vs Modification Type Impact (using imshow)
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap
    modification_types = list(set(r['modification_type'] for r in successful_results))
    scenarios_list = list(scenario_data.keys())

    # Create matrix: mean KL divergence by scenario and modification type
    heatmap_data = np.zeros((len(scenarios_list), len(modification_types)))
    count_data = np.zeros((len(scenarios_list), len(modification_types)))

    for i, scenario in enumerate(scenarios_list):
        for j, mod_type in enumerate(modification_types):
            vals = [r['kl_divergence'] for r in scenario_data[scenario]
                   if r['modification_type'] == mod_type]
            if vals:
                heatmap_data[i, j] = np.mean(vals)
                count_data[i, j] = len(vals)

    # Create heatmap using imshow
    # Mask where we have no data
    display_data = np.where(count_data > 0, heatmap_data, np.nan)

    im = plt.imshow(display_data, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Mean KL Divergence')

    plt.xticks(range(len(modification_types)), modification_types, rotation=45, ha='right')
    plt.yticks(range(len(scenarios_list)), scenarios_list)
    plt.xlabel('Modification Type')
    plt.ylabel('Scenario')
    plt.title('Heatmap: Mean KL Divergence by Scenario and Modification Type')

    # Add text annotations
    for i in range(len(scenarios_list)):
        for j in range(len(modification_types)):
            if count_data[i, j] > 0:
                text = f'{heatmap_data[i, j]:.3f}\n(n={int(count_data[i, j])})'
                # Choose text color based on background darkness
                text_color = 'white' if heatmap_data[i, j] > np.nanmax(heatmap_data) * 0.6 else 'black'
                plt.text(j, i, text, ha='center', va='center',
                        fontsize=8, color=text_color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_heatmap_scenario_modtype.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Trend Analysis: How modification magnitude affects impact
    plt.figure(figsize=(14, 5))

    # Calculate modification magnitude (L1 distance between original and new probabilities)
    mod_magnitudes = []
    kl_values = []
    drift_values = []

    for result in successful_results:
        orig = np.array(result['original_probs'])
        new = np.array(result['new_probs'])
        magnitude = np.sum(np.abs(orig - new))  # L1 distance
        mod_magnitudes.append(magnitude)
        kl_values.append(result['kl_divergence'])
        drift_values.append(result['max_drift'])

    plt.subplot(1, 2, 1)
    plt.scatter(mod_magnitudes, kl_values, alpha=0.6, s=30, c='purple', edgecolors='black', linewidth=0.5)
    plt.xlabel('Modification Magnitude (L1 distance)')
    plt.ylabel('KL Divergence')
    plt.title('Trend: KL Divergence vs Modification Magnitude')
    plt.grid(True, alpha=0.3)

    # Add trend line
    if len(mod_magnitudes) > 1:
        z = np.polyfit(mod_magnitudes, kl_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(mod_magnitudes), max(mod_magnitudes), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend line')
        plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(mod_magnitudes, drift_values, alpha=0.6, s=30, c='orange', edgecolors='black', linewidth=0.5)
    plt.xlabel('Modification Magnitude (L1 distance)')
    plt.ylabel('Maximum Absolute Drift')
    plt.title('Trend: Max Drift vs Modification Magnitude')
    plt.grid(True, alpha=0.3)

    # Add trend line
    if len(mod_magnitudes) > 1:
        z = np.polyfit(mod_magnitudes, drift_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(mod_magnitudes), max(mod_magnitudes), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend line')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_trend_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Top Examples: Most impactful test cases
    plt.figure(figsize=(12, 10))

    # Sort by KL divergence and take top 10
    top_by_kl = sorted(successful_results, key=lambda x: x['kl_divergence'], reverse=True)[:10]
    top_by_drift = sorted(successful_results, key=lambda x: x['max_drift'], reverse=True)[:10]

    plt.subplot(2, 1, 1)
    # Top KL divergence cases
    case_ids = [r['case_id'] for r in top_by_kl]
    kl_vals = [r['kl_divergence'] for r in top_by_kl]
    scenarios_kl = [r['scenario_id'] for r in top_by_kl]

    bars = plt.bar(range(len(case_ids)), kl_vals, alpha=0.8, color='red', edgecolor='black', linewidth=0.5)
    plt.xlabel('Test Case Rank (by KL Divergence)')
    plt.ylabel('KL Divergence')
    plt.title('Top 10 Test Cases by KL Divergence')
    # Set x-axis labels manually
    plt.xticks(range(len(case_ids)), [f'#{cid}\n({scen[:10]}...)' for cid, scen in zip(case_ids, scenarios_kl)],
               rotation=45, ha='right', fontsize=8)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, kl_vals):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(kl_vals)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.subplot(2, 1, 2)
    # Top max drift cases
    case_ids_drift = [r['case_id'] for r in top_by_drift]
    drift_vals = [r['max_drift'] for r in top_by_drift]
    scenarios_drift = [r['scenario_id'] for r in top_by_drift]

    bars = plt.bar(range(len(case_ids_drift)), drift_vals, alpha=0.8, color='blue', edgecolor='black', linewidth=0.5)
    plt.xlabel('Test Case Rank (by Max Drift)')
    plt.ylabel('Maximum Absolute Drift')
    plt.title('Top 10 Test Cases by Maximum Absolute Drift')
    # Set x-axis labels manually
    plt.xticks(range(len(case_ids_drift)),
               [f'#{cid}\n({scen[:10]}...)' for cid, scen in zip(case_ids_drift, scenarios_drift)],
               rotation=45, ha='right', fontsize=8)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, drift_vals):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(drift_vals)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_top_examples.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Scenario Comparison: How effects vary across scenarios
    plt.figure(figsize=(12, 10))

    scenario_names = list(scenario_data.keys())
    scenario_means_kl = [np.mean([r['kl_divergence'] for r in scenario_data[sc]])
                        for sc in scenario_names]
    scenario_means_drift = [np.mean([r['max_drift'] for r in scenario_data[sc]])
                           for sc in scenario_names]
    scenario_stds_kl = [np.std([r['kl_divergence'] for r in scenario_data[sc]])
                       for sc in scenario_names]
    scenario_stds_drift = [np.std([r['max_drift'] for r in scenario_data[sc]])
                          for sc in scenario_names]

    x = np.arange(len(scenario_names))
    width = 0.35

    plt.subplot(2, 2, 1)
    bars1 = plt.bar(x - width/2, scenario_means_kl, width, yerr=scenario_stds_kl,
           label='KL Divergence', alpha=0.8, color='skyblue', edgecolor='black', capsize=3)
    bars2 = plt.bar(x + width/2, scenario_means_drift, width, yerr=scenario_stds_drift,
           label='Max Drift', alpha=0.8, color='lightcoral', edgecolor='black', capsize=3)
    plt.xlabel('Scenario')
    plt.ylabel('Mean Value')
    plt.title('Scenario Comparison: Mean Impact Metrics')
    # Set x-axis labels manually
    plt.xticks(x, scenario_names, rotation=45, ha='right', fontsize=9)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (mean_kl, std_kl, mean_drift, std_drift) in enumerate(zip(scenario_means_kl, scenario_stds_kl,
                                                                     scenario_means_drift, scenario_stds_drift)):
        plt.text(i - width/2, mean_kl + std_kl + 0.01, f'{mean_kl:.3f}',
                ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, mean_drift + std_drift + 0.01, f'{mean_drift:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.subplot(2, 2, 2)
    # Box plot of KL divergence by scenario
    kl_data_by_scenario = [[r['kl_divergence'] for r in scenario_data[sc]]
                          for sc in scenario_names]
    # Create boxplot without labels parameter, then set labels
    box_plots = plt.boxplot(kl_data_by_scenario, patch_artist=True)
    plt.xticks(range(1, len(scenario_names) + 1), scenario_names)
    plt.ylabel('KL Divergence')
    plt.title('Distribution of KL Divergence by Scenario')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(scenario_names)))
    for patch, color in zip(box_plots['boxes'], colors):
        patch.set_facecolor(color)

    plt.subplot(2, 2, 3)
    # Box plot of max drift by scenario
    drift_data_by_scenario = [[r['max_drift'] for r in scenario_data[sc]]
                             for sc in scenario_names]
    box_plots2 = plt.boxplot(drift_data_by_scenario, patch_artist=True)
    plt.xticks(range(1, len(scenario_names) + 1), scenario_names)
    plt.ylabel('Maximum Absolute Drift')
    plt.title('Distribution of Maximum Drift by Scenario')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    # Color the boxes
    for patch, color in zip(box_plots2['boxes'], colors):
        patch.set_facecolor(color)

    plt.subplot(2, 2, 4)
    # Scatter: KL divergence vs max drift, colored by scenario
    colors_scatter = plt.cm.Set1(np.linspace(0, 1, len(scenario_names)))
    for i, scenario in enumerate(scenario_names):
        scenario_results = scenario_data[scenario]
        kl_vals = [r['kl_divergence'] for r in scenario_results]
        drift_vals = [r['max_drift'] for r in scenario_results]
        plt.scatter(kl_vals, drift_vals, alpha=0.7, s=40,
                   label=scenario, color=colors_scatter[i], edgecolors='black', linewidth=0.5)

    plt.xlabel('KL Divergence')
    plt.ylabel('Maximum Absolute Drift')
    plt.title('KL Divergence vs Max Drift (by Scenario)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_scenario_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create a summary statistics file
    summary_stats = {
        'total_tests': len(results),
        'successful_tests': len(successful_results),
        'failed_tests': len(results) - len(successful_results),
        'success_rate': len(successful_results) / len(results) if results else 0,
        'kl_divergence': {
            'mean': float(np.mean(kl_divergences)),
            'median': float(np.median(kl_divergences)),
            'std': float(np.std(kl_divergences)),
            'min': float(np.min(kl_divergences)),
            'max': float(np.max(kl_divergences)),
            'percentiles': {
                '25': float(np.percentile(kl_divergences, 25)),
                '75': float(np.percentile(kl_divergences, 75)),
                '90': float(np.percentile(kl_divergences, 90)),
                '95': float(np.percentile(kl_divergences, 95))
            }
        },
        'max_drift': {
            'mean': float(np.mean(max_drifts)),
            'median': float(np.median(max_drifts)),
            'std': float(np.std(max_drifts)),
            'min': float(np.min(max_drifts)),
            'max': float(np.max(max_drifts)),
            'percentiles': {
                '25': float(np.percentile(max_drifts, 25)),
                '75': float(np.percentile(max_drifts, 75)),
                '90': float(np.percentile(max_drifts, 90)),
                '95': float(np.percentile(max_drifts, 95))
            }
        },
        'by_scenario': {
            scenario: {
                'count': len(scenario_data[scenario]),
                'mean_kl': float(np.mean([r['kl_divergence'] for r in scenario_data[scenario]])),
                'mean_max_drift': float(np.mean([r['max_drift'] for r in scenario_data[scenario]])),
                'kl_std': float(np.std([r['kl_divergence'] for r in scenario_data[scenario]])),
                'drift_std': float(np.std([r['max_drift'] for r in scenario_data[scenario]]))
            }
            for scenario in scenario_names
        }
    }

    # Save summary statistics
    with open(f"{output_dir}/summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print(f"Visualizations saved to {output_dir}/")
    print("Created files:")
    for i in range(1, 6):
        print(f"  {output_dir}/0{i}_*.png")
    print(f"  {output_dir}/summary_statistics.json")

    return summary_stats

def main():
    """Main function to run batch sensitivity analysis"""
    print("=" * 70)
    print("BATCH SENSITIVITY ANALYSIS")
    print("Testing @starter/student_core.py with 100+ test cases")
    print("=" * 70)

    # Load data
    print("Loading Bayesian network data...")
    bn_schema, base_cpt, scenarios = load_data()
    print(f"Loaded: {len(scenarios)} scenarios, {len(bn_schema['nodes'])} nodes")
    print(f"Scenarios: {[s['id'] for s in scenarios]}")

    # Generate test cases
    print("\nGenerating test cases...")
    test_cases = generate_test_cases(bn_schema, base_cpt, scenarios, num_cases=100)

    # Run all tests
    print("\nRunning sensitivity experiments...")
    results = run_all_tests(test_cases, bn_schema, base_cpt)

    # Create visualizations
    print("\nCreating summary visualizations...")
    summary_stats = create_summary_visualizations(results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total test cases: {len(results)}")
    print(f"Successful: {summary_stats['successful_tests']}")
    print(f"Success rate: {summary_stats['success_rate']*100:.1f}%")
    print(f"Mean KL divergence: {summary_stats['kl_divergence']['mean']:.4f}")
    print(f"Mean max drift: {summary_stats['max_drift']['mean']:.4f}")
    print(f"\nVisualizations saved in: batch_visualizations/")
    print("Check the batch_visualizations folder for 5 summary PNG files and statistics.")
    print("\nGenerated files:")
    print("  01_distribution_plots.png - Histograms of KL divergence, max drift, total variation")
    print("  02_heatmap_scenario_modtype.png - Impact heatmap by scenario and modification type")
    print("  03_trend_analysis.png - How modification size affects impact")
    print("  04_top_examples.png - Most influential test cases")
    print("  05_scenario_comparison.png - How effects vary across the 4 scenarios")
    print("  summary_statistics.json - Numerical summary of all results")

if __name__ == "__main__":
    main()