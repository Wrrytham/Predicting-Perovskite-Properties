import torch
from torch_geometric.data import Data
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import defaultdict
from CBFV.composition import generate_features
from contextlib import redirect_stdout
import io

class CBFVFeaturizer:
    """
    Wrapper for CBFV library implementation with feature selection
    """
    def __init__(self, preset, selected_features):
        """
        Initialize with different property sets and feature selection
        
        Args:
            preset: 'oliynyk', 'jarvis', 'magpie', etc.
            selected_features: list of feature names to use
        """
        self.preset = preset
        self.feature_set = preset
        
        # Generate dummy features to get all available feature names
        with redirect_stdout(io.StringIO()):
            dummy_df = pd.DataFrame({"formula": ["CaTiO3"], "target": [0]})
            X, y, formulas, skipped = generate_features(dummy_df, elem_prop=self.feature_set)
        
        self.all_feature_labels = list(X.columns)
        
        # Validate selected features
        invalid_features = [f for f in selected_features if f not in self.all_feature_labels]
        if invalid_features:
            print(f"Warning: Invalid features ignored: {invalid_features}")
        
        self.selected_features = [f for f in selected_features if f in self.all_feature_labels]
        
        if not self.selected_features:
            raise ValueError("No valid features selected.")
        
        # Create feature index mapping
        self.feature_indices = [self.all_feature_labels.index(f) for f in self.selected_features]
        
        print(f"Initialized CBFV featurizer with {preset} feature set")
        print(f"Total available features: {len(self.all_feature_labels)}")
        print(f"Selected features: {len(self.selected_features)}")
        print(f"Selected feature names: {self.selected_features[:10]}{'...' if len(self.selected_features) > 10 else ''}")
    
    def featurize_composition(self, composition_dict):
        """
        Create CBFV features from composition dictionary
        
        Args:
            composition_dict: dict like {'Ca': 1, 'Ti': 1, 'O': 3}
            
        Returns:
            numpy array of selected CBFV features
        """
        try:
            formula = ''.join([f"{el}{comp if comp > 1 else ''}" for el, comp in composition_dict.items()])
            df = pd.DataFrame({"formula": [formula], "target": [0]})
            with redirect_stdout(io.StringIO()):
                X, y, formulas, skipped = generate_features(df, elem_prop=self.feature_set)
            
            # Select only the specified features
            all_features = X.iloc[0].values
            selected_features = all_features[self.feature_indices]
            
            return np.array(selected_features, dtype=np.float32)
        except Exception as e:
            print(f"Error featurizing composition {composition_dict}: {e}")
            return np.zeros(len(self.selected_features), dtype=np.float32)
    
    def get_feature_names(self):
        """Get names of selected CBFV features"""
        return self.selected_features
    
    def get_all_feature_names(self):
        """Get names of all available CBFV features"""
        return self.all_feature_labels
    
    def get_feature_dimension(self):
        """Get the dimension of selected CBFV features"""
        return len(self.selected_features)
    
    def get_feature_info(self):
        """Get detailed feature information"""
        return {
            'preset': self.preset,
            'total_available': len(self.all_feature_labels),
            'selected': len(self.selected_features),
            'selected_names': self.selected_features,
            'all_names': self.all_feature_labels
        }

def build_graph_CBFV(row, featurizer):
    """
    Build graph using CBFV featurization for nodes and edges
    """
    # --- Parse composition ---
    composition_str = row['composition']
    comp = {}
    expanded_atoms = []
    for token in composition_str.split():
        el = ''.join([c for c in token if c.isalpha()])
        num = ''.join([c for c in token if c.isdigit()])
        num = int(num) if num else 1
        comp[el] = num
        expanded_atoms.extend([el] * num)
    total_atoms = len(expanded_atoms)
    fractions = {el: count / total_atoms for el, count in comp.items()}
    center_atom = max(fractions, key=fractions.get)
    center_idx = expanded_atoms.index(center_atom)
    
    # --- CBFV Node features ---
    nodes = []
    for el in expanded_atoms:
        single_elem_comp = {el: 1}
        elem_CBFV = featurizer.featurize_composition(single_elem_comp)
        elem_CBFV = np.nan_to_num(elem_CBFV, nan=0.0, posinf=0.0, neginf=0.0)
        elem_features = np.concatenate([elem_CBFV, [fractions[el]]])
        nodes.append(elem_features.tolist())
    x = torch.tensor(nodes, dtype=torch.float)
    
    # --- Edge construction with CBFV edge features ---
    edge_index = []
    edge_attr = []
    for i, el in enumerate(expanded_atoms):
        if i == center_idx: 
            continue
        center_el = expanded_atoms[center_idx]
        edge_comp = {center_el: 1, el: 1}
        edge_CBFV = featurizer.featurize_composition(edge_comp)
        edge_CBFV = np.nan_to_num(edge_CBFV, nan=0.0, posinf=0.0, neginf=0.0)
        structural_features = [
            row['a_edge (angstrom)'], 
            row['alpha_ang (deg)'],
            fractions[el],
            fractions[center_el]
        ]
        structural_features = np.nan_to_num(np.array(structural_features), nan=0.0, posinf=0.0, neginf=0.0)
        # Use selected CBFV features for edges (limit to prevent too large edge features)
        CBFV_limit = min(20, len(edge_CBFV))
        edge_feat = np.concatenate([edge_CBFV[:CBFV_limit], structural_features])
        edge_index.append([center_idx, i])
        edge_index.append([i, center_idx])
        edge_attr.append(edge_feat.tolist())
        edge_attr.append(edge_feat.tolist())
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, min(20, featurizer.get_feature_dimension()) + 4), dtype=torch.float)
    
    # --- Global features (composition CBFV + crystal features) ---
    crystal_systems = ['cubic','tetragonal','orthorhombic','monoclinic','triclinic','hexagonal','trigonal']
    crystal_sys_enc = [1.0 if row['crystal_system'] == cs else 0.0 for cs in crystal_systems]
    material_props = [
        row['density (g/cc)'],
        row['total_magnetisation (bohr)'],
        row['volume (cubic-angstrom)']
    ]
    material_props = np.nan_to_num(np.array(material_props), nan=0.0, posinf=0.0, neginf=0.0)
    CBFV_features = featurizer.featurize_composition(comp)
    CBFV_features = np.nan_to_num(CBFV_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Use selected CBFV features for global features (limit to prevent too large global features)
    global_CBFV_limit = min(50, len(CBFV_features))
    global_feat = np.concatenate([CBFV_features[:global_CBFV_limit], material_props, crystal_sys_enc])
    global_feat = np.nan_to_num(global_feat, nan=0.0, posinf=0.0, neginf=0.0)
    global_feat = torch.tensor(global_feat, dtype=torch.float)
    
    # --- Targets ---
    y_values = [row['formation_energy (eV/atom)'], row['band_gap (eV)']]
    y_values = np.nan_to_num(np.array(y_values), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.tensor(y_values, dtype=torch.float)
    
    data = Data(
        x=torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        edge_index=edge_index,
        edge_attr=torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0),
        u=global_feat.unsqueeze(0),
        y=y
    )
    data.formula = row.get('formula', 'unknown')
    data.atom_list = expanded_atoms
    data.center_idx = center_idx
    return data


def process_csv_CBFV(csv_file_path, CBFV_preset='oliynyk'):
    """
    Process CSV using CBFV featurization
    """
    # Get predefined feature selections based on preset
    preset_features =  {
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
    
    # Get the appropriate feature set for the preset
    selected_features = preset_features.get(CBFV_preset, preset_features['oliynyk'])
    
    # Initialize featurizer with selected features
    featurizer = CBFVFeaturizer(preset=CBFV_preset, selected_features=selected_features)
    df = pd.read_csv(csv_file_path)
    df = df[df['band_gap (eV)'] != 0]
    
    # Define critical columns that must not have NaNs
    critical_cols = [
        'composition', 'crystal_system', 'formation_energy (eV/atom)', 'band_gap (eV)',
        'a_edge (angstrom)', 'alpha_ang (deg)', 'density (g/cc)', 
        'total_magnetisation (bohr)', 'volume (cubic-angstrom)'
    ]
    # Log and save dropped rows
    dropped_rows = df[df[critical_cols].isna().any(axis=1)][['formula']]
    if not dropped_rows.empty:
        print("Dropped rows with NaNs in critical columns:")
        print(dropped_rows)
        dropped_rows.to_csv('dropped_rows.csv', index=False)
        print("Dropped rows saved to 'dropped_rows.csv'")
    
    # Drop rows with NaNs in critical columns
    initial_len = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows due to NaN values in critical columns: {critical_cols}")
    
    structural_cols = [
        'a_edge (angstrom)', 'b_edge (angstrom)', 'c_edge (angstrom)',
        'alpha_ang (deg)', 'beta_ang (deg)', 'gamma_ang (deg)',
        'density (g/cc)', 'total_magnetisation (bohr)', 'volume (cubic-angstrom)'
    ]
    for col in structural_cols:
        if col in df.columns:
            # Fill remaining NaNs with median (for non-critical cols like b_edge, c_edge, etc.)
            median = df[col].median()
            df[col] = df[col].fillna(median)
            # Clip outliers (>3 std) to median
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(mean - 3*std, mean + 3*std)
    
    scaler = StandardScaler()
    df[structural_cols] = scaler.fit_transform(df[structural_cols])
    
    print("Sample processed data:")
    print(df.head())
    print(f"Selected CBFV feature dimension: {featurizer.get_feature_dimension()}")
    
    graphs = []
    failed_count = 0
    for idx, row in df.iterrows():
        try:
            graph = build_graph_CBFV(row, featurizer)
            graphs.append(graph)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            failed_count += 1
            continue
    
    print(f"Successfully processed {len(graphs)} materials, failed: {failed_count}")
    
    if not graphs:
        return graphs, featurizer, None
    
    # Global normalization for better results
    # Scale node features (x)
    all_x = np.vstack([g.x.numpy() for g in graphs])
    all_x = np.nan_to_num(all_x, nan=0.0, posinf=0.0, neginf=0.0)
    x_scaler = StandardScaler().fit(all_x)
    for g in graphs:
        x = g.x.numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        g.x = torch.tensor(x_scaler.transform(x), dtype=torch.float)
    
    # Scale edge attributes (if present)
    edge_attrs = [g.edge_attr.numpy() for g in graphs if g.edge_attr.numel() > 0]
    if edge_attrs:
        all_edge_attr = np.vstack(edge_attrs)
        all_edge_attr = np.nan_to_num(all_edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        edge_scaler = StandardScaler().fit(all_edge_attr)
        for g in graphs:
            if g.edge_attr.numel() > 0:
                edge_attr = g.edge_attr.numpy()
                edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
                g.edge_attr = torch.tensor(edge_scaler.transform(edge_attr), dtype=torch.float)
    
    # Scale global features (u)
    all_u = np.vstack([g.u.numpy() for g in graphs])  # g.u is already (1, n_features)
    if all_u.ndim == 1:  # Handle case where all_u is 1D (e.g., single sample)
        all_u = all_u.reshape(1, -1)
    print(f"all_u shape before scaling: {all_u.shape}")  # Debug print
    all_u = np.nan_to_num(all_u, nan=0.0, posinf=0.0, neginf=0.0)
    u_scaler = StandardScaler().fit(all_u)
    for g in graphs:
        u = g.u.numpy()
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        transformed_u = u_scaler.transform(u)
        g.u = torch.tensor(transformed_u, dtype=torch.float)
    
    # Scale targets (y) for stability
    all_y = np.vstack([g.y.numpy() for g in graphs])
    print(f"all_y shape before scaling: {all_y.shape}")  # Debug print
    all_y = np.nan_to_num(all_y, nan=0.0, posinf=0.0, neginf=0.0)
    y_scaler = StandardScaler().fit(all_y)
    for g in graphs:
        y_reshaped = g.y.numpy().reshape(1, -1)  # Reshape to (1, 2) for StandardScaler
        y_reshaped = np.nan_to_num(y_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
        transformed_y = y_scaler.transform(y_reshaped)
        g.y = torch.tensor(transformed_y, dtype=torch.float)
    
    # Check for remaining NaNs after scaling
    has_nan = any(torch.isnan(g.x).any() or torch.isnan(g.edge_attr).any() or torch.isnan(g.u).any() or torch.isnan(g.y).any() for g in graphs)
    if has_nan:
        print("Warning: NaNs still present after processing. Model may fail.")
    
    return graphs, featurizer, y_scaler


def plot_graph_with_CBFV(graph, featurizer, filename="outputs/graph_CBFV_detailed.png"):
    """
    Visualize a PyG graph with selected CBFV features
    """
    G = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"])
    labels = {}
    feature_names = featurizer.get_feature_names()
    
    for i, el in enumerate(graph.atom_list):
        CBFV_vals = graph.x[i][:-1].tolist()
        fraction = graph.x[i][-1].item()
        label_parts = [f"{el}"]
        label_parts.append(f"f={fraction:.2f}")
        if len(CBFV_vals) >= 3:
            label_parts.append(f"F1={CBFV_vals[0]:.2f}")
            label_parts.append(f"F2={CBFV_vals[1]:.2f}")
            label_parts.append(f"F3={CBFV_vals[2]:.2f}")
        labels[i] = "\n".join(label_parts)
    
    pos = nx.spring_layout(G, seed=42)
    fig = plt.figure(figsize=(20, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    nx.draw(G, pos, with_labels=False, node_size=3000,
            node_color="lightblue", edgecolors="black", ax=ax1)
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight="bold", ax=ax1)
    nx.draw_networkx_nodes(G, pos, nodelist=[graph.center_idx], 
                           node_color="orange", node_size=3500, edgecolors="black", ax=ax1)
    ax1.set_title("Material Graph with Selected CBFV Features", fontsize=12, fontweight="bold")
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.axis('off')
    global_features = graph.u.squeeze().tolist()
    CBFV_global = global_features[:-10]
    material_props = global_features[-10:-7]
    crystal_encoding = global_features[-7:]
    crystal_system = "unknown"
    if 1.0 in crystal_encoding:
        crystal_system = ['cubic','tetragonal','orthorhombic','monoclinic','triclinic','hexagonal','trigonal'][crystal_encoding.index(1.0)]
    targets = graph.y.tolist()
    
    global_text = f"""SELECTED CBFV + MATERIAL FEATURES

CBFV Feature Set: {featurizer.preset}
Total Available Features: {len(featurizer.get_all_feature_names())}
Selected Features: {featurizer.get_feature_dimension()}

Key Selected CBFV Features (first 5):"""
    for i in range(min(5, len(CBFV_global))):
        if i < len(feature_names):
            global_text += f"\n• {feature_names[i][:20]}: {CBFV_global[i]:.3f}"
    
    global_text += f"""

Material Properties:
• Density: {material_props[0]:.3f} g/cc
• Magnetization: {material_props[1]:.3f} μB
• Volume: {material_props[2]:.2f} Å³
• Crystal System: {crystal_system}

Target Properties:
• Formation Energy: {targets[0]:.4f} eV/atom
• Band Gap: {targets[1]:.4f} eV

Graph Statistics:
• Total Atoms: {len(graph.atom_list)}
• Center Atom: {graph.atom_list[graph.center_idx]}
• Node Feature Dim: {graph.x.shape[1]}
• Edge Feature Dim: {graph.edge_attr.shape[1] if graph.edge_attr.numel() > 0 else 0}
• Global Feature Dim: {graph.u.shape[1]}"""
    
    ax2.text(0.05, 0.95, global_text, transform=ax2.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax2.set_title("Selected CBFV Features & Properties", fontsize=12, fontweight="bold")
    
    # Feature visualization
    ax3 = plt.subplot(2, 3, 3)
    num_features_to_show = min(15, len(CBFV_global))
    CBFV_subset = CBFV_global[:num_features_to_show]
    feature_names_short = []
    for i in range(num_features_to_show):
        if i < len(feature_names):
            name = feature_names[i]
            if len(name) > 15:
                name = name[:12] + "..."
            feature_names_short.append(name)
        else:
            feature_names_short.append(f"F{i+1}")
    
    bars = ax3.barh(range(len(CBFV_subset)), CBFV_subset, color='skyblue')
    ax3.set_yticks(range(len(CBFV_subset)))
    ax3.set_yticklabels(feature_names_short, fontsize=8)
    ax3.set_xlabel('Feature Value')
    ax3.set_title('Selected CBFV Features', fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    # Composition pie chart
    ax4 = plt.subplot(2, 3, 4)
    unique_elements = list(set(graph.atom_list))
    element_counts = [graph.atom_list.count(el) for el in unique_elements]
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_elements)))
    wedges, texts, autotexts = ax4.pie(element_counts, labels=unique_elements, 
                                       autopct='%1.1f%%', colors=colors)
    ax4.set_title('Composition', fontsize=12, fontweight="bold")
    
    # Feature distribution
    ax5 = plt.subplot(2, 3, 5)
    if graph.x.shape[0] > 1:
        first_feature_values = graph.x[:, 0].numpy()
        ax5.hist(first_feature_values, bins=min(10, len(first_feature_values)), 
                alpha=0.7, color='green', edgecolor='black')
        ax5.set_xlabel('Feature Value')
        ax5.set_ylabel('Count')
        feature_name = feature_names[0] if len(feature_names) > 0 else "Feature 1"
        ax5.set_title(f'Distribution of {feature_name[:20]}', fontsize=10, fontweight="bold")
    else:
        ax5.text(0.5, 0.5, 'Single atom\nmaterial', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Node Feature Distribution', fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph with selected CBFV features saved to {filename}")


if __name__ == "__main__":
    print("CBFV Featurization Example")
    print("=" * 60)
    
    csv_file = "Perovskite_data_cleaned.csv"
    
    try:
        graphs, featurizer, y_scaler = process_csv_CBFV(
            csv_file, 
            CBFV_preset="oliynyk"
        )
        
        if graphs:
            plot_graph_with_CBFV(graphs[0], featurizer, "selected_features_example.png")
            
    except FileNotFoundError:
        print(f"CSV file '{csv_file}' not found. Creating example...")
        selected_features = [
            'avg_Atomic_Number', 'avg_Atomic_Weight', 'avg_Period', 'avg_group',
            'avg_families', 'avg_Metal', 'avg_Nonmetal', 'avg_Metalliod',
            'avg_Mendeleev_Number', 'avg_l_quantum_number', 'avg_Atomic_Radius',
            'avg_Miracle_Radius_[pm]', 'avg_Covalent_Radius', 'avg_Zunger_radii_sum',
            'avg_ionic_radius', 'avg_crystal_radius', 'avg_Pauling_Electronegativity',
            'avg_MB_electonegativity', 'avg_Gordy_electonegativity', 'avg_Mulliken_EN',
            'avg_Allred-Rockow_electronegativity', 'avg_metallic_valence',
            'avg_number_of_valence_electrons', 'avg_gilmor_number_of_valence_electron',
            'avg_valence_s', 'avg_valence_p', 'avg_valence_d', 'avg_valence_f',
            'avg_Number_of_unfilled_s_valence_electrons',
            'avg_Number_of_unfilled_p_valence_electrons',
            'avg_Number_of_unfilled_d_valence_electrons',
            'avg_Number_of_unfilled_f_valence_electrons', 'avg_outer_shell_electrons',
            'avg_1st_ionization_potential_(kJ/mol)', 'avg_polarizability(A^3)',
            'avg_Melting_point_(K)', 'avg_Boiling_Point_(K)', 'avg_Density_(g/mL)',
            'avg_specific_heat_(J/g_K)_', 'avg_heat_of_fusion_(kJ/mol)_',
            'avg_heat_of_vaporization_(kJ/mol)_', 'avg_thermal_conductivity_(W/(m_K))_',
            'avg_heat_atomization(kJ/mol)', 'avg_Cohesive_energy',
            'dev_Atomic_Number', 'dev_Atomic_Weight', 'dev_Period', 'dev_group',
            'dev_families', 'dev_Metal', 'dev_Nonmetal', 'dev_Metalliod',
            'dev_Mendeleev_Number', 'dev_l_quantum_number', 'dev_Atomic_Radius',
            'dev_Miracle_Radius_[pm]', 'dev_Covalent_Radius', 'dev_Zunger_radii_sum',
            'dev_ionic_radius', 'dev_crystal_radius', 'dev_Pauling_Electronegativity',
            'dev_MB_electonegativity', 'dev_Gordy_electonegativity', 'dev_Mulliken_EN',
            'dev_Allred-Rockow_electronegativity', 'dev_metallic_valence',
            'dev_number_of_valence_electrons', 'dev_gilmor_number_of_valence_electron',
            'dev_valence_s', 'dev_valence_p', 'dev_valence_d', 'dev_valence_f',
            'dev_Number_of_unfilled_s_valence_electrons',
            'dev_Number_of_unfilled_p_valence_electrons',
            'dev_Number_of_unfilled_d_valence_electrons',
            'dev_Number_of_unfilled_f_valence_electrons', 'dev_outer_shell_electrons',
            'dev_1st_ionization_potential_(kJ/mol)', 'dev_polarizability(A^3)',
            'dev_Melting_point_(K)', 'dev_Boiling_Point_(K)', 'dev_Density_(g/mL)',
            'dev_specific_heat_(J/g_K)_', 'dev_heat_of_fusion_(kJ/mol)_',
            'dev_heat_of_vaporization_(kJ/mol)_', 'dev_thermal_conductivity_(W/(m_K))_',
            'dev_heat_atomization(kJ/mol)', 'dev_Cohesive_energy',
            'range_Atomic_Number', 'range_Atomic_Weight', 'range_Period', 'range_group',
            'range_families', 'range_Metal', 'range_Nonmetal', 'range_Metalliod',
            'range_Mendeleev_Number', 'range_l_quantum_number', 'range_Atomic_Radius',
            'range_Miracle_Radius_[pm]', 'range_Covalent_Radius', 'range_Zunger_radii_sum',
            'range_ionic_radius', 'range_crystal_radius', 'range_Pauling_Electronegativity',
            'range_MB_electonegativity', 'range_Gordy_electonegativity', 'range_Mulliken_EN',
            'range_Allred-Rockow_electronegativity', 'range_metallic_valence',
            'range_number_of_valence_electrons', 'range_gilmor_number_of_valence_electron',
            'range_valence_s', 'range_valence_p', 'range_valence_d', 'range_valence_f',
            'range_Number_of_unfilled_s_valence_electrons',
            'range_Number_of_unfilled_p_valence_electrons',
            'range_Number_of_unfilled_d_valence_electrons',
            'range_Number_of_unfilled_f_valence_electrons', 'range_outer_shell_electrons',
            'range_1st_ionization_potential_(kJ/mol)', 'range_polarizability(A^3)',
            'range_Melting_point_(K)', 'range_Boiling_Point_(K)', 'range_Density_(g/mL)',
            'range_specific_heat_(J/g_K)_', 'range_heat_of_fusion_(kJ/mol)_',
            'range_heat_of_vaporization_(kJ/mol)_', 'range_thermal_conductivity_(W/(m_K))_',
            'range_heat_atomization(kJ/mol)', 'range_Cohesive_energy'
        ]
        featurizer = CBFVFeaturizer(preset='oliynyk', selected_features=selected_features)
        example_comp = {'Ca': 1, 'Ti': 1, 'O': 3}
        features = featurizer.featurize_composition(example_comp)
        print(f"Example features shape: {features.shape}")
        print(f"Selected features: {featurizer.get_feature_names()}")