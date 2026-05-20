import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
from torch_geometric.data import DataLoader, Batch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
from GraphBuild2 import process_csv_CBFV, CBFVFeaturizer

class MaterialsGNN(nn.Module):
    def __init__(self, 
                 node_features,  # Renamed for clarity
                 edge_features,  # Renamed for clarity
                 global_features,  # Renamed for clarity
                 hidden_dim=256,
                 num_conv_layers=6,
                 dropout=0.2,
                 heads=12):  # Increased heads for better attention
        super(MaterialsGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout

        # Initial feature normalization
        self.node_norm = nn.BatchNorm1d(node_features)
        self.edge_norm = nn.BatchNorm1d(edge_features) if edge_features > 0 else None
        
        # Enhanced embeddings with layer normalization
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        ) if edge_features > 0 else None

        # Graph Attention Convolutions with multi-head attention
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.conv_layers.append(
                GATConv(hidden_dim, hidden_dim,
                        edge_dim=hidden_dim if edge_features > 0 else None,
                        heads=heads,
                        concat=False,
                        dropout=0.1)
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # Global (graph-level) feature processor
        self.global_mlp = nn.Sequential(
            nn.Linear(global_features, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout)
        )

        # Final regression head (predicting 2 targets)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch

        # Initial feature normalization
        x = self.node_norm(x)
        if edge_attr is not None and self.edge_norm is not None:
            edge_attr = self.edge_norm(edge_attr)

        # Node embedding
        x = self.node_embedding(x)
        
        # Edge embedding
        if edge_attr is not None and edge_attr.size(0) > 0 and self.edge_embedding is not None:
            edge_attr = self.edge_embedding(edge_attr)
        else:
            edge_attr = torch.zeros(edge_index.size(1), self.hidden_dim,
                                    device=x.device, dtype=x.dtype)

        # Graph convolutions with residual connections
        x_prev = x
        for i in range(self.num_conv_layers):
            x_conv = self.conv_layers[i](x, edge_index, edge_attr)
            x_conv = F.leaky_relu(x_conv, negative_slope=0.1)
            x_conv = self.bn_layers[i](x_conv)
            x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
            # Residual connection
            x = x_conv + x_prev if i > 0 else x_conv
            x_prev = x

        # Pooling
        x_pooled = global_mean_pool(x, batch)

        # Global features
        if u.dim() == 3:
            u = u.squeeze(1)
        elif u.dim() == 2 and u.size(0) == 1:
            u = u.squeeze(0).unsqueeze(0)
        u = self.global_mlp(u)

        # Concatenate pooled + global features
        combined = torch.cat([x_pooled, u], dim=1)
        out = self.final_mlp(combined)
        return out


class MaterialsGCN(nn.Module):
    def __init__(self, 
                 node_features,
                 edge_features,
                 global_features,
                 hidden_dim=128,
                 num_conv_layers=4,
                 dropout=0.3):
        super(MaterialsGCN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout

        # Embeddings for node and edge features
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim) if edge_features > 0 else None

        # Projection for edge weights (since GCNConv uses 1D edge_weight)
        self.edge_proj = nn.Linear(hidden_dim, 1) if edge_features > 0 else None

        # Graph Convolutions
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # Global (graph-level) feature processor
        self.global_mlp = nn.Sequential(
            nn.Linear(global_features, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout)
        )

        # Final regression head (predicting 2 targets)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch

        # Node embedding
        x = F.leaky_relu(self.node_embedding(x), negative_slope=0.01)

        # Edge embedding and projection to weight
        if edge_attr is not None and self.edge_embedding:
            edge_attr = F.leaky_relu(self.edge_embedding(edge_attr), negative_slope=0.01)
        edge_weight = None
        if self.edge_proj and edge_attr is not None:
            edge_weight = torch.sigmoid(self.edge_proj(edge_attr)).squeeze()
            edge_weight = torch.clamp(edge_weight, min=1e-6, max=1.0)  # Ensure positive, stable weights

        # Graph convolutions
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x, negative_slope=0.01)
            x = self.bn_layers[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        x_pooled = global_mean_pool(x, batch)

        # Global features
        if u.dim() == 3:
            u = u.squeeze(1)
        elif u.dim() == 2 and u.size(0) == 1:
            u = u.squeeze(0).unsqueeze(0)
        u = self.global_mlp(u)

        # Concatenate pooled + global features
        combined = torch.cat([x_pooled, u], dim=1)
        out = self.final_mlp(combined)
        return out


class MaterialsGraphSAGE(nn.Module):
    def __init__(self, 
                 node_features,
                 edge_features,
                 global_features,
                 hidden_dim=128,
                 num_conv_layers=4,
                 dropout=0.3):
        super(MaterialsGraphSAGE, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout

        # Embeddings for node and edge features
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim) if edge_features > 0 else None

        # GraphSAGE Convolutions
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.conv_layers.append(
                SAGEConv(hidden_dim, hidden_dim, aggr='mean')
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # Global (graph-level) feature processor
        self.global_mlp = nn.Sequential(
            nn.Linear(global_features, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout)
        )

        # Final regression head (predicting 2 targets)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch

        # Node embedding
        x = F.leaky_relu(self.node_embedding(x), negative_slope=0.01)

        # Edge embedding (GraphSAGE doesn't use edge_attr directly, but we can incorporate it if needed)
        if edge_attr is not None and self.edge_embedding:
            # Optional: You can concatenate edge features to node features or use them separately
            pass  # GraphSAGE typically ignores edge_attr, but extensions can be added

        # Graph convolutions
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x, edge_index)
            x = F.leaky_relu(x, negative_slope=0.01)
            x = self.bn_layers[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        x_pooled = global_mean_pool(x, batch)

        # Global features
        if u.dim() == 3:
            u = u.squeeze(1)
        elif u.dim() == 2 and u.size(0) == 1:
            u = u.squeeze(0).unsqueeze(0)
        u = self.global_mlp(u)

        # Concatenate pooled + global features
        combined = torch.cat([x_pooled, u], dim=1)
        out = self.final_mlp(combined)
        return out


def train_model(model, train_loader, val_loader, num_epochs=300, lr=0.001, patience=25):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-5)  # Reduced weight decay
    # Improved learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=7, min_lr=1e-6)
    # Use per-target MSE so we can weight band-gap loss if needed in callers
    criterion = nn.MSELoss(reduction='mean')

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, num_train_batches = 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            targets = batch.y
            if targets.dim() == 1:
                targets = targets.view(out.size(0), 2)
            loss = criterion(out, targets)
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}. Skipping batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            num_train_batches += 1

        # Validation
        model.eval()
        val_loss, num_val_batches = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                targets = batch.y
                if targets.dim() == 1:
                    targets = targets.view(out.size(0), 2)
                loss = criterion(out, targets)
                if torch.isnan(loss):
                    print(f"NaN loss detected in validation at epoch {epoch}. Skipping batch.")
                    continue
                val_loss += loss.item()
                num_val_batches += 1

        if num_train_batches > 0 and num_val_batches > 0:
            avg_train_loss = train_loss / num_train_batches
            avg_val_loss = val_loss / num_val_batches
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Step scheduler and log learning rate
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}, Learning Rate: {current_lr:.6f}")

            # Early stopping
            if avg_val_loss < best_val_loss - 0.001:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 50 == 0 and len(train_losses) > 0:
            print(f'Epoch {epoch:03d}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    return train_losses, val_losses


def evaluate_model(model, loader, model_name, split='test', y_scaler=None, featurizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_preds, all_targets = [], []
    all_formulas = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            targets = batch.y
            if targets.dim() == 1:
                targets = targets.view(out.size(0), 2)
            all_preds.append(out.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            # Collect formula from individual graphs
            for data in batch.to_data_list():
                all_formulas.append(getattr(data, 'formula', 'unknown'))

    preds, targets = np.vstack(all_preds), np.vstack(all_targets)
    
    # Inverse transform if scaler provided
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds)
        targets = y_scaler.inverse_transform(targets)
    
    formation_mae = mean_absolute_error(targets[:, 0], preds[:, 0])
    formation_mse = mean_squared_error(targets[:, 0], preds[:, 0])
    formation_rmse = np.sqrt(formation_mse)
    formation_r2 = r2_score(targets[:, 0], preds[:, 0])
    bandgap_mae = mean_absolute_error(targets[:, 1], preds[:, 1])
    bandgap_mse = mean_squared_error(targets[:, 1], preds[:, 1])
    bandgap_rmse = np.sqrt(bandgap_mse)
    bandgap_r2 = r2_score(targets[:, 1], preds[:, 1])

    metrics = {
        'formation_mae': formation_mae,
        'formation_mse': formation_mse,
        'formation_rmse': formation_rmse,
        'formation_r2': formation_r2,
        'bandgap_mae': bandgap_mae,
        'bandgap_mse': bandgap_mse,
        'bandgap_rmse': bandgap_rmse,
        'bandgap_r2': bandgap_r2
    }

    print(f"\n{split.capitalize()} Results for {model_name}:")
    print(f"Formation Energy - MAE: {formation_mae:.4f}, MSE: {formation_mse:.4f}, RMSE: {formation_rmse:.4f}, R2: {formation_r2:.4f}")
    print(f"Band Gap - MAE: {bandgap_mae:.4f}, MSE: {bandgap_mse:.4f}, RMSE: {bandgap_rmse:.4f}, R2: {bandgap_r2:.4f}")

    # Save predictions to CSV (include featurizer in filename if provided)
    fea_str = f"_{featurizer}" if featurizer else ""
    pred_filename = f'{split}_predictions_{model_name}{fea_str}.csv'
    df = pd.DataFrame({
        'formula': all_formulas,
        'true_formation_energy': targets[:, 0],
        'pred_formation_energy': preds[:, 0],
        'true_band_gap': targets[:, 1],
        'pred_band_gap': preds[:, 1]
    })
    df.to_csv(pred_filename, index=False)
    print(f"{split.capitalize()} predictions saved to {pred_filename}")

    # Save metrics to CSV (include featurizer in filename if provided)
    metrics_filename = f'metrics_{model_name}{fea_str}_{split}.csv'
    metrics_df = pd.DataFrame([{
        'model': model_name,
        'split': split,
        'featurizer': featurizer if featurizer else '',
        **metrics
    }])
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"{split.capitalize()} metrics saved to {metrics_filename}")

    return preds, targets, metrics


class CompositionMLP(nn.Module):
    """Simple MLP that uses global (CBFV + structural) features to predict targets.
    This often complements graph-level models for properties like band gap.
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super(CompositionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, data):
        # data.u is expected to be (batch_size, feat_dim) or (batch_size, 1, feat_dim)
        u = data.u
        if u.dim() == 3:
            u = u.squeeze(1)
        return self.net(u)


def train_composition_mlp(model, train_loader, val_loader, num_epochs=200, lr=1e-3, patience=15, bandgap_weight=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6)
    mse = nn.MSELoss(reduction='mean')

    best_val = float('inf')
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            targets = batch.y
            if targets.dim() == 1:
                targets = targets.view(out.size(0), 2)
            loss0 = mse(out[:, 0], targets[:, 0])
            loss1 = mse(out[:, 1], targets[:, 1])
            loss = loss0 + bandgap_weight * loss1
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1

        # validation
        model.eval()
        val_loss, m = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                targets = batch.y
                if targets.dim() == 1:
                    targets = targets.view(out.size(0), 2)
                loss0 = mse(out[:, 0], targets[:, 0])
                loss1 = mse(out[:, 1], targets[:, 1])
                loss = loss0 + bandgap_weight * loss1
                val_loss += loss.item()
                m += 1

        if n > 0 and m > 0:
            avg_val = val_loss / m
            scheduler.step(avg_val)
            if avg_val < best_val - 1e-4:
                best_val = avg_val
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                break

    return model


def plot_training_curves(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title(f'{model_name} Training Curves'); plt.legend(); plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-50:], label='Train Loss (Last 50)', color='blue')
    plt.plot(val_losses[-50:], label='Val Loss (Last 50)', color='red')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title(f'{model_name} Training Curves (Final Epochs)'); plt.legend()

    plt.tight_layout()
    filename = f"outputs/training_curves_{model_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {filename}")


def plot_predictions(preds, targets, model_name, featurizer=None):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(targets[:, 0], preds[:, 0], alpha=0.6, color='blue')
    plt.plot([targets[:, 0].min(), targets[:, 0].max()],
             [targets[:, 0].min(), targets[:, 0].max()], 'r--', lw=2)
    plt.xlabel('Actual Formation Energy'); plt.ylabel('Predicted Formation Energy')
    title = f'{model_name} Formation Energy'
    if featurizer:
        title += f'\n({featurizer} features)'
    plt.title(title); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(targets[:, 1], preds[:, 1], alpha=0.6, color='green')
    plt.plot([targets[:, 1].min(), targets[:, 1].max()],
             [targets[:, 1].min(), targets[:, 1].max()], 'r--', lw=2)
    plt.xlabel('Actual Band Gap'); plt.ylabel('Predicted Band Gap')
    title = f'{model_name} Band Gap'
    if featurizer:
        title += f'\n({featurizer} features)'
    plt.title(title); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Create a more organized filename structure
    model_type = model_name.split('_')[0] if '_' in model_name else model_name  # Extract GNN/GCN/GraphSAGE
    featurizer_str = featurizer if featurizer else 'current'
    filename = f"outputs/predictions_{model_type}_{featurizer_str}.png"
    
    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Test prediction plots saved to {filename}")

def generate_all_featurizer_predictions(model_class, model_name, test_loader, y_scaler=None):
    """Generate prediction plots for all featurizers."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for featurizer in ['jarvis', 'magpie', 'oliynyk']:
        # Extract the base model name without architecture type
        base_model_name = model_name.split('_')[-1] if '_' in model_name else model_name
        model_path = f'models/{base_model_name}_{featurizer}.pth'
        feature_info_path = f'feature_info_{base_model_name}_{featurizer}.json'
        
        if os.path.exists(model_path) and os.path.exists(feature_info_path):
            print(f"\nGenerating predictions for {model_name} with {featurizer} featurizer...")
            
            # Load feature info to get dimensions
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
            
            # Create model with correct dimensions
            if model_name == 'composition_mlp':
                model = CompositionMLP(input_dim=feature_info['global_input_dim'])
            else:
                model = model_class(
                    node_features=feature_info['node_input_dim'],
                    edge_features=feature_info['edge_input_dim'],
                    global_features=feature_info['global_input_dim'],
                        hidden_dim=128,  # Match the dimensions used in training
                        num_conv_layers=4,  # Match the number of layers used in training
                        dropout=0.3  # Match the dropout used in training
                )
            
                # Load saved weights
                model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            model.eval()
            
            # Load data for this featurizer
            dataset, _, _ = process_csv_CBFV('Perovskite_data_cleaned.csv', CBFV_preset=featurizer)
            _, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
            featurizer_test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
            
            # Generate predictions
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in featurizer_test_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    targets = batch.y
                    if targets.dim() == 1:
                        targets = targets.view(out.size(0), 2)
                    all_preds.append(out.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
            
            preds, targets = np.vstack(all_preds), np.vstack(all_targets)
            
            # Inverse transform if scaler provided
            if y_scaler is not None:
                preds = y_scaler.inverse_transform(preds)
                targets = y_scaler.inverse_transform(targets)
                
            # Use evaluate_model to save predictions and metrics with consistent filenames
            # We will create a temporary DataLoader-like object by reusing featurizer_test_loader
            # Evaluate and save using evaluate_model (it will write CSVs including featurizer)
            # NOTE: evaluate_model expects a loader; pass featurizer_test_loader and model_name that includes base model
            evaluate_model(model, featurizer_test_loader, model_name, split='test', y_scaler=y_scaler, featurizer=featurizer)
            
            # Calculate and print metrics
            formation_mae = mean_absolute_error(targets[:, 0], preds[:, 0])
            formation_r2 = r2_score(targets[:, 0], preds[:, 0])
            bandgap_mae = mean_absolute_error(targets[:, 1], preds[:, 1])
            bandgap_r2 = r2_score(targets[:, 1], preds[:, 1])
            
            print(f"\nMetrics for {model_name} with {featurizer} features:")
            print(f"Formation Energy - MAE: {formation_mae:.4f}, R2: {formation_r2:.4f}")
            print(f"Band Gap - MAE: {bandgap_mae:.4f}, R2: {bandgap_r2:.4f}")


def get_selected_features(preset):
    """Get predefined feature selections for each preset"""
    feature_selections = {
        'jarvis': [
    'avg_Z', 'avg_row', 'avg_coulmn', 'avg_atom_mass', 'avg_atom_rad', 'avg_voro_coord',
    'avg_X', 'avg_elec_aff', 'avg_first_ion_en',
    'avg_nsvalence', 'avg_npvalence', 'avg_ndvalence', 'avg_nfvalence',
    'avg_nsunfill', 'avg_npunfill', 'avg_ndunfill', 'avg_nfunfill',
    'avg_mp', 'avg_bp', 'avg_hfus', 'avg_polzbl', 'avg_therm_cond',
    'avg_min_oxid_s', 'avg_max_oxid_s',
    'avg_is_transition_metal', 'avg_is_metalloid', 'avg_is_halogen',
    'dev_X', 'dev_elec_aff', 'dev_first_ion_en', 'dev_atom_mass', 'dev_atom_rad',
    'dev_mp', 'dev_bp', 'dev_polzbl', 'dev_voro_coord',
    'range_X', 'range_elec_aff', 'range_first_ion_en', 'range_atom_mass', 'range_atom_rad'
],
        'magpie': ['avg_Number', 'avg_MendeleevNumber', 'avg_AtomicWeight', 'avg_MeltingT',
 'avg_Column', 'avg_Row', 'avg_CovalentRadius', 'avg_Electronegativity',
 'avg_NsValence', 'avg_NpValence', 'avg_NdValence', 'avg_NValence',
 'avg_NsUnfilled', 'avg_NpUnfilled', 'avg_NdUnfilled', 'avg_NUnfilled',
 'avg_GSvolume_pa', 'avg_GSmagmom', 'avg_SpaceGroupNumber','dev_Electronegativity', 'dev_CovalentRadius', 'dev_NValence', 
 'dev_MendeleevNumber', 'dev_Number'],
        'oliynyk':[
    'avg_Atomic_Number', 'avg_Atomic_Weight', 'avg_Atomic_Radius',
    'avg_Covalent_Radius', 'avg_ionic_radius', 'avg_crystal_radius',
    'avg_Pauling_Electronegativity', 'avg_metallic_valence',
    'avg_number_of_valence_electrons', 'avg_valence_s', 'avg_valence_p',
    'avg_valence_d', 'avg_valence_f', 'avg_Number_of_unfilled_s_valence_electrons',
    'avg_Number_of_unfilled_p_valence_electrons', 'avg_Number_of_unfilled_d_valence_electrons',
    'avg_Number_of_unfilled_f_valence_electrons', 'avg_1st_ionization_potential_(kJ/mol)',
    'avg_polarizability(A^3)', 'avg_Melting_point_(K)', 'avg_Boiling_Point_(K)',
    'avg_heat_of_fusion_(kJ/mol)_', 'avg_heat_of_vaporization_(kJ/mol)_',
    'avg_heat_atomization(kJ/mol)', 'avg_Cohesive_energy',
    'dev_Pauling_Electronegativity', 'dev_Atomic_Radius', 'dev_Covalent_Radius',
    'dev_ionic_radius', 'dev_number_of_valence_electrons', 'dev_valence_d',
    'dev_1st_ionization_potential_(kJ/mol)', 'dev_polarizability(A^3)',
    'dev_heat_of_fusion_(kJ/mol)_', 'dev_Cohesive_energy'
]
        # ,
        # 'mat2vec': [
        #     'avg_Number', 'avg_atomic_mass', 'avg_period', 'avg_group',
        #     'avg_block', 'avg_electronegativity'
        # ]
    }
    return feature_selections.get(preset, [])

def train_models(csv_file, CBFV_preset='oliynyk', model_params=None, training_params=None):
    """
    Train MaterialsGNN (GAT), MaterialsGCN (GCN), MaterialsGraphSAGE models with specific CBFV features.
    """
    # Create necessary directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Set default parameters
    if model_params is None:
        model_params = {
            'hidden_dim': 128,
            'num_conv_layers': 4,
            'dropout': 0.3
        }
    
    if training_params is None:
        training_params = {
            'num_epochs': 200,
            'lr': 0.0005,
            'batch_size': 32,
            'patience': 20
        }

    print("="*60)
    print("TRAINING GAT, GCN, AND GRAPHSAGE MODELS")
    print("="*60)
    print(f"CBFV preset: {CBFV_preset}")
    
    # Process data
    graphs, featurizer, y_scaler = process_csv_CBFV(csv_file, CBFV_preset=CBFV_preset)

    if not graphs:
        print("No graphs generated. Exiting.")
        return

    print(f"Selected CBFV features: {featurizer.get_feature_dimension()}")
    if len(graphs) < 10:
        print("Warning: Very small dataset. Consider using more data.")

    # Split dataset
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.2, random_state=42)

    def collate_fn(batch):
        return Batch.from_data_list(batch)

    batch_size = min(training_params['batch_size'], len(train_graphs) // 2)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Dataset sizes - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Infer dimensions
    node_features = graphs[0].x.shape[1]
    edge_features = graphs[0].edge_attr.shape[1] if graphs[0].edge_attr.numel() > 0 else 0
    global_features = graphs[0].u.shape[1]

    print(f"Model input dimensions:")
    print(f"- Node features: {node_features}")
    print(f"- Edge features: {edge_features}")
    print(f"- Global features: {global_features}")

    # Train a composition-only MLP to complement GNNs for band-gap predictions
    print("\nTraining composition-only MLP (uses global CBFV + structural features)...")
    comp_model = CompositionMLP(input_dim=global_features, hidden_dim=256, dropout=0.2)
    # Train composition MLP with higher weight on band-gap to improve that target
    bandgap_weight = training_params.get('bandgap_weight', 2.0) if training_params else 2.0
    comp_model = train_composition_mlp(comp_model, train_loader, val_loader,
                          num_epochs=training_params.get('num_epochs', 200),
                          lr=training_params.get('lr', 1e-3),
                          patience=training_params.get('patience', 15),
                          bandgap_weight=bandgap_weight)
    # Evaluate composition model
    print("\nEvaluating composition MLP on test set...")
    comp_test_preds, comp_test_targets, comp_metrics = evaluate_model(comp_model, test_loader, 'composition_mlp', split='test', y_scaler=y_scaler)

    # Train and evaluate all GNN models and create an ensemble with composition MLP
    models = [
        ('materials_gnn', MaterialsGNN, 'GAT'),
        ('materials_gcn', MaterialsGCN, 'GCN'),
        ('materials_graphsage', MaterialsGraphSAGE, 'GraphSAGE')
    ]

    for model_name, model_class, display_name in models:
        print(f"\nTraining {display_name} model...")
        print("-" * 40)
        
        model = model_class(
            node_features=node_features,
            edge_features=edge_features,
            global_features=global_features,
            **model_params
        )
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            num_epochs=training_params['num_epochs'], 
            lr=training_params['lr'],
            patience=training_params['patience']
        )

        plot_training_curves(train_losses, val_losses, model_name)

        print(f"\nEvaluating {display_name} model on training set...")
        train_preds, train_targets, train_metrics = evaluate_model(model, train_loader, model_name, split='train', y_scaler=y_scaler, featurizer=CBFV_preset)

        print(f"\nEvaluating {display_name} model on test set...")
        test_preds, test_targets, test_metrics = evaluate_model(model, test_loader, model_name, split='test', y_scaler=y_scaler, featurizer=CBFV_preset)

        # Save prediction plots for current model and featurizer combination
        plot_predictions(test_preds, test_targets, f"{display_name}_{model_name}", CBFV_preset)

        # Generate prediction plots for all models with all featurizers
        print(f"\nGenerating prediction plots for {display_name} model with all featurizers...")
        generate_all_featurizer_predictions(model_class, f"{display_name}_{model_name}", test_loader, y_scaler)

        # Ensemble: simple average of GNN and composition MLP for band-gap, keep formation energy from GNN
        # Load composition predictions for the same test set order
        # comp_test_preds contains both targets; use column 1 for band-gap
        try:
            comp_preds = comp_test_preds
        except NameError:
            comp_preds = None

        if comp_preds is not None:
            # Ensure shapes match
            if comp_preds.shape[0] == test_preds.shape[0]:
                ensembled = test_preds.copy()
                # Weighted average: give higher weight to composition model for band-gap
                w_comp = 0.6
                w_gnn = 0.4
                ensembled[:, 1] = w_comp * comp_preds[:, 1] + w_gnn * test_preds[:, 1]
                # Evaluate ensembled predictions
                if y_scaler is not None:
                    ensembled_inv = y_scaler.inverse_transform(ensembled)
                    targets_inv = y_scaler.inverse_transform(test_targets)
                else:
                    ensembled_inv = ensembled
                    targets_inv = test_targets

                bandgap_mae = mean_absolute_error(targets_inv[:, 1], ensembled_inv[:, 1])
                bandgap_mse = mean_squared_error(targets_inv[:, 1], ensembled_inv[:, 1])
                bandgap_rmse = np.sqrt(bandgap_mse)
                bandgap_r2 = r2_score(targets_inv[:, 1], ensembled_inv[:, 1])

                print(f"\nEnsembled Band Gap Results for {model_name}: MAE: {bandgap_mae:.4f}, RMSE: {bandgap_rmse:.4f}, R2: {bandgap_r2:.4f}")
                # Save ensembled predictions
                df_ens = pd.DataFrame({
                    'true_band_gap': targets_inv[:, 1],
                    'pred_band_gap_ensembled': ensembled_inv[:, 1],
                    'pred_band_gap_gnn': test_preds[:, 1],
                    'pred_band_gap_comp': comp_preds[:, 1]
                })
                df_ens.to_csv(f'test_predictions_{model_name}_ensembled.csv', index=False)
                print(f"Ensembled test predictions saved to test_predictions_{model_name}_ensembled.csv")

        # Save model
        model_filename = f'models/{model_name}_{CBFV_preset}.pth'
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to '{model_filename}'")

        # Save feature info
        feature_info = {
            'preset': CBFV_preset,
            'selected_features': featurizer.get_feature_names(),
            'total_features': len(featurizer.get_all_feature_names()),
            'model_params': model_params,
            'training_params': training_params,
            'node_features': node_features,
            'edge_features': edge_features,
            'global_features': global_features
        }
        with open(f'feature_info_{model_name}_{CBFV_preset}.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"Feature info saved to 'feature_info_{model_name}_{CBFV_preset}.json'")
        
        # Generate predictions for all featurizers
        print(f"\nGenerating prediction plots for all featurizers for {model_name}...")
        generate_all_featurizer_predictions(model_class, model_name, test_loader, y_scaler)

def get_available_feature_sets():
    """Return available CBFV feature sets with descriptions"""
    return {
        'jarvis': 'JARVIS features (physical and chemical properties)',
        'magpie': 'Magpie features (statistical and electronic properties)',
        'oliynyk': 'Oliynyk features (atomic and electronic properties)',
        'mat2vec': 'Mat2Vec features (learned materials embeddings)'
    }

if __name__ == "__main__":
    csv_file = "Perovskite_data_cleaned.csv"
    model_params = {
        'hidden_dim': 256,
        'num_conv_layers': 6,
        'dropout': 0.2
    }
    training_params = {
        'num_epochs': 300,
        'lr': 0.001,
        'batch_size': 64,
        'patience': 25
    }

    # Training and generating predictions for all featurizers
    for featurizer in ['oliynyk', 'magpie', 'jarvis']:
        print(f"\n{'='*60}")
        print(f"Training models with {featurizer.upper()} featurizer")
        print(f"{'='*60}")
        
        train_models(
            csv_file=csv_file,
            CBFV_preset=featurizer,
            model_params=model_params,
            training_params=training_params
        )

    print("\nAvailable CBFV feature sets:")
    for preset, desc in get_available_feature_sets().items():
        print(f"- {preset}: {desc}")