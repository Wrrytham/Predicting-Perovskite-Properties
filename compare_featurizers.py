import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model2 import train_models
import json
import numpy as np

def compare_CBFV_techniques(csv_file):
    """Compare different CBFV featurization techniques"""
    
    CBFV_presets = ['jarvis', 'magpie', 'oliynyk']
    results = {}
    
    # Model parameters consistent across all runs
    model_params = {
        'hidden_dim': 128,
        'num_conv_layers': 4,
        'dropout': 0.3
    }
    
    training_params = {
        'num_epochs': 200,
        'lr': 0.0005,
        'batch_size': 32,
        'patience': 20
    }
    
    # Train models with each featurization technique
    for preset in CBFV_presets:
        print(f"\n{'='*50}")
        print(f"Training models with {preset} featurization")
        print('='*50)
        
        try:
            train_models(
                csv_file=csv_file,
                CBFV_preset=preset,
                model_params=model_params,
                training_params=training_params
            )
            
            # Collect results for each model type
            for model_type in ['materials_gnn', 'materials_gcn', 'materials_graphsage']:
                # Load metrics
                metrics_df = pd.read_csv(f'metrics_{model_type}_test.csv')
                results.setdefault(preset, {})[model_type] = metrics_df.to_dict('records')[0]
                
                # Load feature info
                with open(f'feature_info_{model_type}_{preset}.json', 'r') as f:
                    feature_info = json.load(f)
                results[preset][f'{model_type}_features'] = feature_info
                
        except Exception as e:
            print(f"Error with {preset}: {e}")
    
    # Create comparison plots
    plot_featurization_comparison(results)
    save_comparison_results(results)

def plot_featurization_comparison(results):
    """Create visualization comparing different featurization techniques"""
    
    # Prepare data for plotting
    comparison_data = []
    for preset in results.keys():
        for model_type in ['materials_gnn', 'materials_gcn', 'materials_graphsage']:
            metrics = results[preset][model_type]
            comparison_data.append({
                'Featurization': preset,
                'Model': model_type.split('_')[1].upper(),
                'Formation Energy MAE': metrics['formation_mae'],
                'Band Gap MAE': metrics['bandgap_mae'],
                'Formation Energy R²': metrics['formation_r2'],
                'Band Gap R²': metrics['bandgap_r2']
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['Formation Energy MAE', 'Band Gap MAE', 
              'Formation Energy R²', 'Band Gap R²']
    
    for i, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        sns.barplot(data=df, x='Featurization', y=metric, hue='Model', ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=45)
        if 'MAE' in metric:
            ax.set_ylabel('Mean Absolute Error')
        else:
            ax.set_ylabel('R² Score')
    
    plt.tight_layout()
    plt.savefig('featurization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create feature dimension comparison
    plt.figure(figsize=(10, 6))
    feature_dims = {preset: results[preset]['materials_gnn_features']['total_features'] 
                   for preset in results.keys()}
    plt.bar(feature_dims.keys(), feature_dims.values())
    plt.title('Feature Dimensions by Featurization Method')
    plt.ylabel('Number of Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_results(results):
    """Save detailed comparison results to CSV"""
    
    # Prepare summary table
    summary = []
    for preset in results.keys():
        for model_type in ['materials_gnn', 'materials_gcn', 'materials_graphsage']:
            metrics = results[preset][model_type]
            features = results[preset][f'{model_type}_features']
            
            summary.append({
                'Featurization': preset,
                'Model': model_type.split('_')[1].upper(),
                'Total Features': features['total_features'],
                'Formation Energy MAE': metrics['formation_mae'],
                'Formation Energy RMSE': metrics['formation_rmse'],
                'Formation Energy R²': metrics['formation_r2'],
                'Band Gap MAE': metrics['bandgap_mae'],
                'Band Gap RMSE': metrics['bandgap_rmse'],
                'Band Gap R²': metrics['bandgap_r2']
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('featurization_comparison_summary.csv', index=False)
    print("\nComparison results saved to:")
    print("- featurization_comparison.png")
    print("- feature_dimensions.png")
    print("- featurization_comparison_summary.csv")

if __name__ == "__main__":
    csv_file = "Perovskite_data_cleaned.csv"
    compare_CBFV_techniques(csv_file)
