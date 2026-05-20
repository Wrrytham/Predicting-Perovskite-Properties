import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model2 import train_models, get_selected_features
from GraphBuild2 import CBFVFeaturizer
import numpy as np
import json
from pathlib import Path

def analyze_featurization_techniques(csv_file):
    """Analyze and compare different CBFV featurization techniques"""
    presets = ['jarvis', 'magpie', 'oliynyk']
    results = {}
    
    for preset in presets:
        print(f"\nAnalyzing {preset} featurization...")
        try:
            # Train models and collect metrics
            model_results = train_and_evaluate(csv_file, preset)
            
            # Store results
            results[preset] = {
                'preset': preset,
                'metrics': model_results['metrics'] if model_results else {},
                'feature_count': len(get_selected_features(preset))
            }
            
        except Exception as e:
            print(f"Error analyzing {preset}: {e}")
            results[preset] = {
                'preset': preset,
                'metrics': {},
                'feature_count': 0,
                'error': str(e)
            }
    
    # Generate comparison plots and reports
    generate_comparison_report(results)
    return results

def train_and_evaluate(csv_file, preset):
    """Train models with specific featurization and collect results"""
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
    
    try:
        return train_models(
            csv_file=csv_file,
            CBFV_preset=preset,
            model_params=model_params,
            training_params=training_params
        )
    except Exception as e:
        print(f"Training failed for {preset}: {e}")
        return None

def generate_comparison_report(results):
    """Generate comprehensive comparison report"""
    Path("featurization_results").mkdir(exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    for preset, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            for model_type in ['GAT', 'GCN', 'GraphSAGE']:
                if model_type in metrics:
                    plot_data.append({
                        'Featurization': preset,
                        'Model': model_type,
                        'Formation Energy MAE': metrics[model_type].get('formation_mae', 0),
                        'Band Gap MAE': metrics[model_type].get('bandgap_mae', 0),
                        'Formation Energy R²': metrics[model_type].get('formation_r2', 0),
                        'Band Gap R²': metrics[model_type].get('bandgap_r2', 0)
                    })
    
    df = pd.DataFrame(plot_data)
    
    # Feature count comparison
    plt.figure(figsize=(10, 6))
    feature_counts = {k: v['feature_count'] for k, v in results.items()}
    plt.bar(feature_counts.keys(), feature_counts.values())
    plt.title('Number of Features by Featurization Method')
    plt.ylabel('Feature Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('featurization_results/feature_counts.png')
    plt.close()
    
    if not df.empty:
        # Model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = [
            ('Formation Energy MAE', 'Formation Energy MAE'),
            ('Band Gap MAE', 'Band Gap MAE'),
            ('Formation Energy R²', 'Formation Energy R²'),
            ('Band Gap R²', 'Band Gap R²')
        ]
        
        for (col, title), ax in zip(metrics, axes.flat):
            if col in df.columns:
                sns.barplot(data=df, x='Featurization', y=col, hue='Model', ax=ax)
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('featurization_results/model_performance.png')
        plt.close()
        
        # Save detailed results
        df.to_csv('featurization_results/performance_metrics.csv', index=False)
    
    # Save complete results
    with open('featurization_results/full_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    csv_file = "Perovskite_data_cleaned.csv"
    try:
        results = analyze_featurization_techniques(csv_file)
        print("\nAnalysis complete. Results saved in 'featurization_results' directory.")
    except Exception as e:
        print(f"Analysis failed: {e}")
