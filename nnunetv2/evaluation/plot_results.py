import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List

def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_confusion_matrix(confusion_matrix: List[List[int]], output_dir: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_per_class_metrics(per_class_metrics: Dict, output_dir: str) -> None:
    classes = list(per_class_metrics.keys())
    metrics_df = pd.DataFrame({
        'Class': classes * 2,
        'Metric': ['Precision'] * len(classes) + ['Recall'] * len(classes),
        'Value': [per_class_metrics[c]['precision'] for c in classes] +
                [per_class_metrics[c]['recall'] for c in classes]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Class', y='Value', hue='Metric')
    plt.title('Per Class Metrics')
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), bbox_inches='tight', dpi=300)
    plt.close()

def extract_per_case_segmentation_metrics(json_data: Dict) -> pd.DataFrame:
    metrics_list = []
    
    for case in json_data['metric_per_case']:
        case_metrics = case['metrics']
        case_name = os.path.basename(case['prediction_file'])
        
        for label, metrics in case_metrics.items():
            metrics_list.append({
                'case': case_name,
                'label': label,
                'Dice': metrics['Dice'],
                'IoU': metrics['IoU'],
                'Surface Dice 2mm': metrics['surface_dice_at_2mm'],
                'Hausdorff 95': metrics.get('robust_hausdorff_95', np.nan)  # Some cases might not have this
            })
    
    return pd.DataFrame(metrics_list)

def plot_segmentation_boxplots(seg_metrics_df: pd.DataFrame, output_dir: str) -> None:
    metrics = ['Dice', 'IoU', 'Surface Dice 2mm', 'Hausdorff 95']
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=seg_metrics_df, x='label', y=metric)
        plt.title(f'{metric} Boxplot by Label')
        plt.savefig(os.path.join(output_dir, f'{metric.lower()}_boxplot.png'), bbox_inches='tight', dpi=300)
        plt.close()

def generate_tex_tables(json_data: Dict, output_dir: str) -> None:
    seg_metrics_df = extract_per_case_segmentation_metrics(json_data)
    classification_metrics = json_data.get('classification_metrics', {})

    # Segmentation table
    pancreas_dice = seg_metrics_df[seg_metrics_df['label'] == '1']['Dice'].median() * 100
    lesion_dice = seg_metrics_df[seg_metrics_df['label'] == '2']['Dice'].median() * 100
    pancreas_surface = seg_metrics_df[seg_metrics_df['label'] == '1']['Surface Dice 2mm'].median() * 100
    lesion_surface = seg_metrics_df[seg_metrics_df['label'] == '2']['Surface Dice 2mm'].median() * 100
    pancreas_hd = seg_metrics_df[seg_metrics_df['label'] == '1']['Hausdorff 95'].median()
    lesion_hd = seg_metrics_df[seg_metrics_df['label'] == '2']['Hausdorff 95'].median()

    seg_table = (
        "\\begin{table}[ht]\n"
        "\\caption{Segmentation Results. Results reported are median values.}\n"
        "\\label{tab:segmentation-results}\n"
        "\\centering\n"
        "\\begin{tabular}{@{}lcc@{}}\n"
        "\\toprule\n"
        "Metric & Pancreas & Lesion \\\\ \\midrule\n"
        f"Dice Score (\\%) & {pancreas_dice:.1f} & {lesion_dice:.1f} \\\\\n"
        f"Surface Dice Score (\\%) & {pancreas_surface:.1f} & {lesion_surface:.1f} \\\\\n"
        f"Hausdorff Distance (mm) & {pancreas_hd:.1f} & {lesion_hd:.1f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    # Classification table
    if classification_metrics:
        per_class = classification_metrics['per_class']
        overall_acc = classification_metrics['accuracy'] * 100
        overall_prec = classification_metrics['precision'] * 100
        overall_rec = classification_metrics['recall'] * 100

        class_table = (
            "\\begin{table}[ht]\n"
            "\\caption{Classification Results. Results reported are mean values.}\n"
            "\\label{tab:classification-results}\n"
            "\\centering\n"
            "\\begin{tabular}{@{}lcccc@{}}\n"
            "\\toprule\n"
            "Metric & Subtype 0 & Subtype 1 & Subtype 2 & Overall \\\\ \\midrule\n"
            f"Accuracy (\\%) & {per_class['0']['precision']*100:.1f} & {per_class['1']['precision']*100:.1f} & {per_class['2']['precision']*100:.1f} & {overall_acc:.1f} \\\\\n"
            f"Precision (\\%) & {per_class['0']['precision']*100:.1f} & {per_class['1']['precision']*100:.1f} & {per_class['2']['precision']*100:.1f} & {overall_prec:.1f} \\\\\n"
            f"Recall (\\%) & {per_class['0']['recall']*100:.1f} & {per_class['1']['recall']*100:.1f} & {per_class['2']['recall']*100:.1f} & {overall_rec:.1f} \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
    else:
        class_table = ""

    # Write tables to files
    with open(os.path.join(output_dir, 'segmentation_results.tex'), 'w') as f:
        f.write(seg_table)
    
    if class_table:
        with open(os.path.join(output_dir, 'classification_results.tex'), 'w') as f:
            f.write(class_table)

def generate_plots(json_data: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # Classification plots
    if 'classification_metrics' in json_data:
        plot_confusion_matrix(json_data['classification_metrics']['confusion_matrix'], output_dir)
        plot_per_class_metrics(json_data['classification_metrics']['per_class'], output_dir)
    
    # Segmentation plots
    if 'metric_per_case' in json_data:
        seg_metrics_df = extract_per_case_segmentation_metrics(json_data)
        plot_segmentation_boxplots(seg_metrics_df, output_dir)
        
        # Save metrics summary
        summary_file = os.path.join(output_dir, 'metrics_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("Segmentation Metrics Summary\n")
            f.write("===========================\n\n")
            
            for label in seg_metrics_df['label'].unique():
                f.write(f"Label {label}:\n")
                label_metrics = seg_metrics_df[seg_metrics_df['label'] == label]
                for metric in ['Dice', 'IoU', 'Surface Dice 2mm', 'Hausdorff 95']:
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean: {label_metrics[metric].mean():.4f}\n")
                    f.write(f"  Std:  {label_metrics[metric].std():.4f}\n")
                    f.write(f"  Med:  {label_metrics[metric].median():.4f}\n")
                f.write("\n")
    
    # Generate LaTeX tables
    generate_tex_tables(json_data, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Generate plots and LaTeX tables from JSON metrics.')
    parser.add_argument('input_json', type=str, help='Path to the input JSON file.')
    parser.add_argument('output_dir', type=str, help='Directory to save the plots and tables.')
    args = parser.parse_args()

    json_data = load_json(args.input_json)
    generate_plots(json_data, args.output_dir)

if __name__ == '__main__':
    main()
