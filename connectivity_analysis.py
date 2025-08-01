import pandas as pd
import glob
import os
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple


# Data Loading Functions
def get_all_subjects(base_path: str = 'shared/data/RBC/PNC_CPAC/cpac_RBCv0') -> List[str]:
    """
    Get all valid subject IDs from the dataset.
    
    Args:
        base_path (str): Path to the cpac_RBCv0 directory
    
    Returns:
        List[str]: Sorted list of subject IDs
    """
    subjects = []
    
    for item in os.listdir(base_path):
        if item.startswith('sub-') and os.path.isdir(os.path.join(base_path, item)):
            # Check if subject has ses-PNC1/func folder
            func_path = os.path.join(base_path, item, 'ses-PNC1', 'func')
            if os.path.exists(func_path):
                subjects.append(item)
    
    return sorted(subjects)


def get_subject_files(subject_id: str, 
                     base_path: str = 'shared/data/RBC/PNC_CPAC/cpac_RBCv0') -> Dict[str, str]:
    """
    Get all connectivity TSV files for a subject, organized by atlas.
    
    Args:
        subject_id (str): Subject ID (e.g., 'sub-1317462')
        base_path (str): Path to the dataset
    
    Returns:
        Dict[str, str]: Dictionary mapping atlas names to file paths
    """
    subject_path = os.path.join(base_path, subject_id, 'ses-PNC1', 'func')
    
    if not os.path.exists(subject_path):
        return {}
    
    # Find all matching TSV files
    file_pattern = '*rest*36Parameter_desc*PearsonNilearn_correlations*.tsv'
    tsv_files = glob.glob(os.path.join(subject_path, file_pattern))
    
    atlas_files = {}
    
    for file_path in tsv_files:
        atlas_name = extract_atlas_name(file_path)
        if atlas_name:
            atlas_files[atlas_name] = file_path
    
    return atlas_files


def get_available_atlases(base_path: str = 'shared/data/RBC/PNC_CPAC/cpac_RBCv0',
                         sample_size: int = 10) -> List[str]:
    """
    Get list of available atlases by sampling subjects.
    
    Args:
        base_path (str): Path to the dataset
        sample_size (int): Number of subjects to sample for atlas discovery
    
    Returns:
        List[str]: List of unique atlas names found across sampled subjects
    """
    subjects = get_all_subjects(base_path)
    atlases = set()
    
    # Sample subjects to find available atlases
    sample_subjects = subjects[:sample_size] if len(subjects) > sample_size else subjects
    
    for subject_id in sample_subjects:
        subject_files = get_subject_files(subject_id, base_path)
        atlases.update(subject_files.keys())
    
    return sorted(list(atlases))


def extract_atlas_name(file_path: str) -> Optional[str]:
    """Extract atlas name from filename."""
    filename = os.path.basename(file_path)
    
    if 'atlas-' in filename and '_space' in filename:
        start_idx = filename.find('atlas-') + len('atlas-')
        end_idx = filename.find('_space', start_idx)
        return filename[start_idx:end_idx]
    
    return None


def make_square_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Convert non-square matrix to square by finding and cropping around diagonal.
    
    Args:
        matrix (np.ndarray): Input matrix
    
    Returns:
        np.ndarray: Square matrix
    """
    if matrix.shape[0] == matrix.shape[1]:
        return matrix
    
    min_dim = min(matrix.shape)
    
    # Check for diagonal of 1s starting from column 1 (common pattern)
    if matrix.shape[1] > matrix.shape[0]:
        # Check diagonal starting from (0,1)
        diagonal_ones = True
        for i in range(min_dim):
            if i + 1 < matrix.shape[1] and abs(matrix[i, i + 1] - 1.0) > 0.1:
                diagonal_ones = False
                break
        
        if diagonal_ones:
            return matrix[:, 1:min_dim+1]
    
    # Default: crop to minimum dimension
    return matrix[:min_dim, :min_dim]


def load_connectivity_matrix(file_path: str) -> Optional[np.ndarray]:
    """
    Load and preprocess a connectivity matrix from TSV file.
    
    Args:
        file_path (str): Path to the TSV file
    
    Returns:
        Optional[np.ndarray]: Connectivity matrix or None if loading failed
    """
    try:
        data = pd.read_csv(file_path, sep='\t')
        
        # Drop first column if it appears to be an index/label column
        first_col = data.iloc[:, 0]
        if (first_col.dtype == 'object' or 
            (first_col.dtype in ['int64', 'float64'] and 
             len(first_col.unique()) == len(first_col))):
            matrix = data.iloc[:, 1:].values
        else:
            matrix = data.values
        
        # Handle non-square matrices by cropping
        matrix = make_square_matrix(matrix)
        
        return matrix
        
    except Exception:
        return None


# Network Processing Functions
def preprocess_matrix(matrix: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Preprocess connectivity matrix for network analysis.
    
    Args:
        matrix (np.ndarray): Raw connectivity matrix
        threshold (float): Correlation threshold for edge creation
    
    Returns:
        np.ndarray: Preprocessed matrix ready for graph creation
    """
    # Remove diagonal (self-connections)
    matrix_copy = matrix.copy()
    np.fill_diagonal(matrix_copy, 0)
    
    # Apply threshold to keep only strong connections
    matrix_thresholded = np.where(np.abs(matrix_copy) > threshold, matrix_copy, 0)
    
    # Use absolute values for graph creation (NetworkX handles all non-zero as edges)
    matrix_abs = np.abs(matrix_thresholded)
    
    # Min-max normalize
    if matrix_abs.max() > matrix_abs.min():
        matrix_normalized = ((matrix_abs - matrix_abs.min()) / 
                           (matrix_abs.max() - matrix_abs.min()))
    else:
        matrix_normalized = matrix_abs
    
    return matrix_normalized


def compute_network_measures(matrix: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute specific network measures from connectivity matrix.
    
    Args:
        matrix (np.ndarray): Preprocessed connectivity matrix
        threshold (float): Threshold used for preprocessing
    
    Returns:
        Dict: Dictionary of network measures
    """
    # Create NetworkX graph
    G = nx.from_numpy_array(matrix)
    
    # Basic graph properties for reference
    num_components = nx.number_connected_components(G)
    
    # Get largest component for path length calculation
    components = list(nx.connected_components(G))
    if components:
        largest_component = max(components, key=len)
    
    # 1. Average clustering coefficient
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0.0
    
    # 2. Average shortest path length (for largest component)
    if num_components > 1:
        try:
            subgraph = G.subgraph(largest_component)
            avg_shortest_path = nx.average_shortest_path_length(subgraph)
        except:
            avg_shortest_path = 0.0
    else:
        avg_shortest_path = 0.0  # No meaningful path length
    
    # 3. Average rich-clubness
    try:
        # Compute rich club coefficient once for all k values
        rc_coeff = nx.rich_club_coefficient(G, normalized=False)
        
        if rc_coeff:
            # Extract coefficients for all meaningful k values
            rich_club_coeffs = []
            for k, coeff in rc_coeff.items():
                if k > 0:  # Only meaningful degree thresholds
                    rich_club_coeffs.append(coeff)
            
            # Calculate average rich club coefficient across all k values
            if rich_club_coeffs:
                avg_rich_club = np.mean(rich_club_coeffs)
            else:
                avg_rich_club = 0.0
        else:
            avg_rich_club = 0.0
    except:
        avg_rich_club = 0.0
    
    return {
        'avg_clustering': avg_clustering,
        'avg_shortest_path': avg_shortest_path,
        'avg_rich_club': avg_rich_club
    }


def process_subject_atlas(subject_id: str, atlas_name: str, file_path: str, 
                         threshold: float = 0.5) -> Optional[Dict]:
    """
    Process a single subject-atlas combination.
    
    Args:
        subject_id (str): Subject ID
        atlas_name (str): Atlas name
        file_path (str): Path to connectivity file
        threshold (float): Correlation threshold
    
    Returns:
        Optional[Dict]: Analysis results or None if failed
    """
    try:
        # Load connectivity matrix
        matrix = load_connectivity_matrix(file_path)
        if matrix is None:
            return None
        
        # Preprocess matrix
        processed_matrix = preprocess_matrix(matrix, threshold)
        
        # Compute network measures
        measures = compute_network_measures(processed_matrix, threshold)
        
        # Add metadata
        result = {
            'subject_id': subject_id,
            **measures
        }
        
        return result
        
    except Exception:
        return None


# Main Pipeline Function
def process_all_subjects(atlas_name: str, 
                        threshold: float = 0.5,
                        base_path: str = 'shared/data/RBC/PNC_CPAC/cpac_RBCv0',
                        save_results: bool = True,
                        output_file: str = 'connectivity_results.csv') -> pd.DataFrame:
    """
    Process all subjects and compute network measures for a specific atlas.
    
    Args:
        atlas_name (str): Name of the atlas to analyze (required parameter)
        threshold (float): Correlation threshold for analysis
        base_path (str): Path to the dataset
        save_results (bool): Whether to save results to CSV
        output_file (str): Output filename for results
    
    Returns:
        pd.DataFrame: Results dataframe with subject_id as first column
    """
    subjects = get_all_subjects(base_path)
    print(f"Processing {len(subjects)} subjects for atlas: {atlas_name}")
    
    all_results = []
    failed_subjects = []
    
    for i, subject_id in enumerate(subjects):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(subjects)} subjects")
        
        # Get available files for this subject
        subject_files = get_subject_files(subject_id, base_path)
        
        if not subject_files:
            failed_subjects.append((subject_id, "No files found"))
            continue
        
        # Process only the specified atlas
        if atlas_name not in subject_files:
            failed_subjects.append((subject_id, f"Atlas {atlas_name} not found"))
            continue
        
        result = process_subject_atlas(subject_id, atlas_name, subject_files[atlas_name], threshold)
        if result:
            all_results.append(result)
        else:
            failed_subjects.append((subject_id, f"Processing failed for atlas {atlas_name}"))
    
    # Convert to DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        # Ensure subject_id is the first column
        cols = ['subject_id'] + [col for col in df.columns if col != 'subject_id']
        df = df[cols]
    else:
        df = pd.DataFrame()
    
    print(f"\nCompleted: {len(all_results)} successful analyses")
    print(f"Failed: {len(failed_subjects)} cases")
    
    if failed_subjects and len(failed_subjects) <= 20:
        print("Failed cases:", failed_subjects)
    elif failed_subjects:
        print(f"Failed cases (showing first 10): {failed_subjects[:10]}")
    
    # Save results
    if save_results and not df.empty:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return df

# Main analysis function
def main():
    """
    Main function to process all subjects for Schaefer2018p1000n17 atlas.
    """
    atlas_name = 'Schaefer2018p1000n17'
    
    print("=" * 60)
    print(f"CONNECTIVITY ANALYSIS FOR {atlas_name}")
    print("=" * 60)
    
    # Process all subjects for the specified atlas
    results_df = process_all_subjects(
        atlas_name=atlas_name,
        threshold=0.5,
        save_results=True,
        output_file=f'connectivity_results_{atlas_name}.csv'
    )
    
    if not results_df.empty:
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total subjects processed: {len(results_df)}")
        print(f"Atlas: {atlas_name}")
        print(f"Threshold: 0.5")
        
        # Display summary statistics
        print(f"\nNetwork Measures Summary:")
        print(f"Average Clustering:")
        print(f"  Mean: {results_df['avg_clustering'].mean():.4f}")
        print(f"  Std:  {results_df['avg_clustering'].std():.4f}")
        print(f"  Range: [{results_df['avg_clustering'].min():.4f}, {results_df['avg_clustering'].max():.4f}]")
        
        print(f"\nAverage Shortest Path Length:")
        print(f"  Mean: {results_df['avg_shortest_path'].mean():.4f}")
        print(f"  Std:  {results_df['avg_shortest_path'].std():.4f}")
        print(f"  Range: [{results_df['avg_shortest_path'].min():.4f}, {results_df['avg_shortest_path'].max():.4f}]")
        
        print(f"\nAverage Rich Club Coefficient:")
        print(f"  Mean: {results_df['avg_rich_club'].mean():.4f}")
        print(f"  Std:  {results_df['avg_rich_club'].std():.4f}")
        print(f"  Range: [{results_df['avg_rich_club'].min():.4f}, {results_df['avg_rich_club'].max():.4f}]")
        
        print(f"\nResults saved to: connectivity_results_{atlas_name}.csv")
        print("="*60)
        
        return results_df
    else:
        print("No results generated. Please check your data and atlas name.")
        return None


# Example usage:
if __name__ == "__main__":
    # Run the main analysis
    results = main()