# pandaprot/core.py
"""
Enhanced core functionality for PandaProt with additional interaction types.
"""

import os
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Structure, Model, Chain, Residue, Atom
from Bio.PDB.vectors import Vector
import py3Dmol
import logging as logger
import networkx as nx
import matplotlib
# Use 'Agg' backend for non-GUI environments
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
import time


# Import all interaction modules
from .interactions import (
    hydrogen_bonds,
    ionic,
    hydrophobic,
    pi_interactions,
    salt_bridges,
    cation_pi,
    ch_pi,
    disulfide,
    sulfur_aromatic,
    water_mediated,
    metal_coordination,
    halogen_bonds,
    amide_aromatic,
    van_der_waals,
    amide_amide
)
from .visualization import plot3d, network
from .reports import generator
#from pandaprot.visualization.export_vis import export_visualization_scripts


class PandaProt:
    """
    PandaProt: A comprehensive tool for mapping and visualizing 
    interactions at protein interfaces.
    """
    
    def __init__(self, pdb_file: str, chains: Optional[List[str]] = None):
        """
        Initialize PandaProt with a PDB file and optional chain specifications.
        
        Args:
            pdb_file: Path to PDB file
            chains: Optional list of chains to analyze (e.g., ['A', 'B'])
        """
        self.pdb_file = pdb_file
        self.chains = chains
        self.structure = None
        self.interactions = {}
        self.parser = PDBParser(QUIET=True)
        self._load_structure()
        
    def _load_structure(self):
        """Load the PDB structure using BioPython."""
        try:
            self.structure = self.parser.get_structure('complex', self.pdb_file)
            print(f"Successfully loaded structure from {self.pdb_file}")
            
            # If chains are specified, validate they exist
            if self.chains:
                available_chains = [chain.id for chain in self.structure[0]]
                for chain in self.chains:
                    if chain not in available_chains:
                        raise ValueError(f"Chain {chain} not found in structure. "
                                        f"Available chains: {', '.join(available_chains)}")
        except Exception as e:
            raise ValueError(f"Failed to load PDB file: {e}")
    
    def map_interactions(self, distance_cutoff: float = 4.5, include_intrachain: bool = False):
        """
        Map all types of interactions between specified chains.
        
        Args:
            distance_cutoff: Maximum distance cutoff for interaction detection
            include_intrachain: Whether to include interactions within the same chain
            
        Returns:
            Dictionary containing all detected interactions
        """
        # Get atoms and residues by chain
        atoms_by_chain = self._get_atoms_by_chain()
        residues_by_chain = self._get_residues_by_chain()
        
        # Map standard interaction types
        self.interactions['hydrogen_bonds'] = hydrogen_bonds.find_hydrogen_bonds(
            atoms_by_chain, include_intrachain=include_intrachain
        )
        
        self.interactions['ionic_interactions'] = ionic.find_ionic_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['hydrophobic_interactions'] = hydrophobic.find_hydrophobic_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['pi_stacking'] = pi_interactions.find_pi_stacking(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['pi_cation'] = pi_interactions.find_pi_cation(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['salt_bridges'] = salt_bridges.find_salt_bridges(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        # Map enhanced interaction types
        self.interactions['cation_pi'] = cation_pi.find_cation_pi_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['ch_pi'] = ch_pi.find_ch_pi_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['disulfide_bridges'] = disulfide.find_disulfide_bridges(
            residues_by_chain, include_intrachain=include_intrachain
        )
        
        self.interactions['sulfur_aromatic'] = sulfur_aromatic.find_sulfur_aromatic_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['water_mediated'] = water_mediated.find_water_mediated_interactions(
            self.structure, chains=self.chains, include_intrachain=include_intrachain
        )
        
        self.interactions['metal_coordination'] = metal_coordination.find_metal_coordination(
            self.structure, chains=self.chains, include_intrachain=include_intrachain
        )
        
        self.interactions['halogen_bonds'] = halogen_bonds.find_halogen_bonds(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['amide_aromatic'] = amide_aromatic.find_amide_aromatic_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        self.interactions['van_der_waals'] = van_der_waals.find_van_der_waals_interactions(
            residues_by_chain, include_intrachain=include_intrachain
        )
        
        self.interactions['amide_amide'] = amide_amide.find_amide_amide_interactions(
            residues_by_chain, distance_cutoff=distance_cutoff, include_intrachain=include_intrachain
        )
        
        # Print summary
        total_interactions = sum(len(interactions) for interactions in self.interactions.values())
        print(f"Found {total_interactions} interactions:")
        for interaction_type, interactions in self.interactions.items():
            print(f"  - {interaction_type}: {len(interactions)}")
        
        return self.interactions
    
    def filter_interactions(self, interaction_types: Optional[List[str]] = None,
                          chains: Optional[List[str]] = None,
                          residues: Optional[List[str]] = None,
                          distance_range: Optional[Tuple[float, float]] = None):
        """
        Filter interactions based on specified criteria.
        
        Args:
            interaction_types: Types of interactions to include
            chains: Chains to include
            residues: Residues to include (format: 'A:ASP32', where A is chain ID)
            distance_range: Range of distances to include (min, max)
            
        Returns:
            Dictionary containing filtered interactions
        """
        if not self.interactions:
            print("No interactions to filter. Run map_interactions() first.")
            return {}
        
        filtered_interactions = {}
        
        # Filter by interaction type
        if interaction_types:
            for interaction_type in interaction_types:
                if interaction_type in self.interactions:
                    filtered_interactions[interaction_type] = self.interactions[interaction_type]
        else:
            filtered_interactions = self.interactions.copy()
        
        # Filter by chain
        if chains:
            for interaction_type, interactions in list(filtered_interactions.items()):
                filtered = []
                for interaction in interactions:
                    # Different interaction types have different field names
                    chain1 = interaction.get('chain1', 
                                          interaction.get('donor_chain',
                                                       interaction.get('positive_chain',
                                                                    interaction.get('aromatic_chain',
                                                                                 interaction.get('sulfur_chain', '')))))
                    
                    chain2 = interaction.get('chain2', 
                                          interaction.get('acceptor_chain',
                                                       interaction.get('negative_chain',
                                                                    interaction.get('pi_chain',
                                                                                 interaction.get('cationic_chain', '')))))
                    
                    if chain1 in chains or chain2 in chains:
                        filtered.append(interaction)
                
                filtered_interactions[interaction_type] = filtered
        
        # Filter by residue
        if residues:
            residue_specs = []
            for res_spec in residues:
                if ':' in res_spec:
                    chain, res = res_spec.split(':')
                    residue_specs.append((chain, res))
                else:
                    # If no chain specified, just use the residue
                    residue_specs.append((None, res_spec))
            
            for interaction_type, interactions in list(filtered_interactions.items()):
                filtered = []
                for interaction in interactions:
                    # Extract residue information from interaction
                    res1 = interaction.get('residue1', 
                                        interaction.get('donor_residue',
                                                     interaction.get('positive_residue',
                                                                  interaction.get('aromatic_residue',
                                                                               interaction.get('sulfur_residue', '')))))
                    
                    chain1 = interaction.get('chain1', 
                                          interaction.get('donor_chain',
                                                       interaction.get('positive_chain',
                                                                    interaction.get('aromatic_chain',
                                                                                 interaction.get('sulfur_chain', '')))))
                    
                    res2 = interaction.get('residue2', 
                                        interaction.get('acceptor_residue',
                                                     interaction.get('negative_residue',
                                                                  interaction.get('pi_residue',
                                                                               interaction.get('cationic_residue', '')))))
                    
                    chain2 = interaction.get('chain2', 
                                          interaction.get('acceptor_chain',
                                                       interaction.get('negative_chain',
                                                                    interaction.get('pi_chain',
                                                                                 interaction.get('cationic_chain', '')))))
                    
                    # Check if either residue matches the specifications
                    for chain_spec, res_spec in residue_specs:
                        if (chain_spec is None or chain_spec == chain1) and (res_spec in res1):
                            filtered.append(interaction)
                            break
                        elif (chain_spec is None or chain_spec == chain2) and (res_spec in res2):
                            filtered.append(interaction)
                            break
                
                filtered_interactions[interaction_type] = filtered
        
        # Filter by distance
        if distance_range:
            min_dist, max_dist = distance_range
            for interaction_type, interactions in list(filtered_interactions.items()):
                filtered = []
                for interaction in interactions:
                    dist = interaction.get('distance', 0)
                    if min_dist <= dist <= max_dist:
                        filtered.append(interaction)
                
                filtered_interactions[interaction_type] = filtered
        
        # Print summary of filtered interactions
        total_filtered = sum(len(interactions) for interactions in filtered_interactions.values())
        print(f"Filtered to {total_filtered} interactions:")
        for interaction_type, interactions in filtered_interactions.items():
            print(f"  - {interaction_type}: {len(interactions)}")
        
        return filtered_interactions
    
    def _get_atoms_by_chain(self) -> Dict[str, List[Atom]]:
        """Get all atoms organized by chain."""
        atoms_by_chain = {}
        
        for model in self.structure:
            for chain in model:
                # Skip if chains are specified and this chain is not in the list
                if self.chains and chain.id not in self.chains:
                    continue
                    
                atoms_by_chain[chain.id] = []
                for residue in chain:
                    # Skip hetero-atoms and water
                    if residue.id[0] != ' ':
                        continue
                        
                    for atom in residue:
                        atoms_by_chain[chain.id].append(atom)
        
        return atoms_by_chain
    
    def _get_residues_by_chain(self) -> Dict[str, List[Residue]]:
        """Get all residues organized by chain."""
        residues_by_chain = {}
        
        for model in self.structure:
            for chain in model:
                # Skip if chains are specified and this chain is not in the list
                if self.chains and chain.id not in self.chains:
                    continue
                    
                residues_by_chain[chain.id] = []
                for residue in chain:
                    # Skip hetero-atoms and water
                    if residue.id[0] != ' ':
                        continue
                        
                    residues_by_chain[chain.id].append(residue)
        
        return residues_by_chain
    
    def visualize_3d(self, output_file: Optional[str] = None, interaction_types: Optional[List[str]] = None):
        """
        Generate 3D visualization of the complex with interactions highlighted.
        
        Args:
            output_file: Optional output file to save the visualization
            interaction_types: Types of interactions to visualize (default: all)
            
        Returns:
            Path to the saved HTML file
        """
        # Set default output file if not provided
        if not output_file:
            output_file = "pandaprot_visualization.html"
            
        # Filter interactions if specified
        interactions_to_visualize = self.interactions
        if interaction_types:
            interactions_to_visualize = {k: v for k, v in self.interactions.items() if k in interaction_types}
        
        # Call the updated visualization function
        html_file = plot3d.create_pandaprot_3d_viz(
            self.pdb_file,  # This can be a file path
            interactions_to_visualize,  # This is the interactions dictionary
            output_file  # This is where to save the output
        )
        
        print(f"3D visualization saved to {html_file}")
        return html_file
    
    def generate_report(self, output_file: Optional[str] = None, interaction_types=None):
        """
        Generate a detailed report of all interactions.
        Args:
            output_file: Optional output file to save the report
            interaction_types: Optional list of interaction types to include in the report
        """
        # If you want to filter interactions based on types:
        filtered_interactions = self.interactions
        if interaction_types:
            # Implement filtering logic here if needed
            # For example: filtered_interactions = [i for i in self.interactions if i.type in interaction_types]
            pass
            
        report_df = generator.create_interaction_report(filtered_interactions)
        if output_file and report_df is not None:
            report_df.to_csv(output_file, index=False)
            print(f"Interaction report saved to {output_file}")
        return report_df
    
    def get_interaction_statistics(self):
        """
        Get statistics about interactions.
        
        Returns:
            Dictionary containing interaction statistics
        """
        if not self.interactions:
            print("No interactions to analyze. Run map_interactions() first.")
            return {}
        
        stats = {
            'total_interactions': sum(len(interactions) for interactions in self.interactions.values()),
            'by_type': {k: len(v) for k, v in self.interactions.items()},
            'avg_distances': {},
            'residue_frequencies': {}
        }
        
        # Calculate average distances for each interaction type
        for interaction_type, interactions in self.interactions.items():
            if interactions:
                distances = [interaction['distance'] for interaction in interactions 
                           if 'distance' in interaction]
                if distances:
                    stats['avg_distances'][interaction_type] = sum(distances) / len(distances)
        
        # Calculate residue frequencies in interactions
        residue_counts = {}
        
        for interaction_type, interactions in self.interactions.items():
            for interaction in interactions:
                # Extract residue info based on interaction type
                res1 = interaction.get('residue1', 
                                     interaction.get('donor_residue',
                                                  interaction.get('positive_residue',
                                                               interaction.get('aromatic_residue',
                                                                            interaction.get('sulfur_residue', '')))))
                
                chain1 = interaction.get('chain1', 
                                       interaction.get('donor_chain',
                                                    interaction.get('positive_chain',
                                                                 interaction.get('aromatic_chain',
                                                                              interaction.get('sulfur_chain', '')))))
                
                res2 = interaction.get('residue2', 
                                     interaction.get('acceptor_residue',
                                                  interaction.get('negative_residue',
                                                               interaction.get('pi_residue',
                                                                            interaction.get('cationic_residue', '')))))
                
                chain2 = interaction.get('chain2', 
                                       interaction.get('acceptor_chain',
                                                    interaction.get('negative_chain',
                                                                 interaction.get('pi_chain',
                                                                              interaction.get('cationic_chain', '')))))
                
                # Count residues
                if res1 and chain1:
                    key = f"{chain1}:{res1}"
                    residue_counts[key] = residue_counts.get(key, 0) + 1
                
                if res2 and chain2:
                    key = f"{chain2}:{res2}"
                    residue_counts[key] = residue_counts.get(key, 0) + 1
        
        # Sort residues by frequency
        stats['residue_frequencies'] = dict(
            sorted(residue_counts.items(), key=lambda x: x[1], reverse=True)
        )
        
        return stats
    
    def sanitize_gml_attributes(self, graph):
        for node, attrs in graph.nodes(data=True):
            for key, value in attrs.items():
                if isinstance(value, (np.generic,)):
                    graph.nodes[node][key] = value.item()
        for u, v, attrs in graph.edges(data=True):
            for key, value in attrs.items():
                if isinstance(value, (np.generic,)):
                    graph[u][v][key] = value.item()

    def create_interaction_network(self, output_file: Optional[str] = None,
                              interaction_types: Optional[List[str]] = None):
        """
        Create a network visualization of interactions.
        Args:
            output_file: Optional output file to save the network visualization
            interaction_types: Types of interactions to include in the network
        """
        if not self.interactions:
            print("No interactions to visualize. Run map_interactions() first.")
            return
            
        if interaction_types:
            filtered_interactions = {
                k: v for k, v in self.interactions.items() if k in interaction_types
            }
        else:
            filtered_interactions = self.interactions
            
        # Get the graph and figure objects
        network_graph, fig = network.create_interaction_network(self.structure, filtered_interactions)
        
        # Handle PNG output with proper extension handling
        if output_file:
            # Create standardized output filename for PNG
            if output_file.endswith('.png'):
                output_png = output_file
            elif output_file.endswith('.html'):
                output_png = output_file.replace('.html', '.png')
            else:
                output_png = f"{output_file}.png"
                
            # Save the figure and immediately close it
            fig.savefig(output_png, dpi=300, bbox_inches='tight')
            plt.close(fig)  # CRITICAL: Clean up matplotlib's state
            logger.info(f"Network visualization saved to {output_png}")
        
        # Handle GML output with proper extension handling
        if output_file:
            self.sanitize_gml_attributes(network_graph)
            
            # Create standardized output filename for GML
            if output_file.endswith('.gml'):
                output_gml = output_file
            elif output_file.endswith(('.html', '.png')):
                output_gml = output_file.replace('.html', '.gml').replace('.png', '.gml')
            else:
                output_gml = f"{output_file}.gml"
                
            nx.write_gml(network_graph, output_gml)
            logger.info(f"Network graph saved to {output_gml}")
            
        # Return the network graph object
        return network_graph

    # In your analyzer class in core.py
    def export_visualization_scripts(self, output_prefix: str, 
                              interaction_types: Optional[List[str]] = None):
        """
        Export interaction visualization scripts for various molecular viewers.
        
        Args:
            output_prefix: Prefix for output files (without extension)
            interaction_types: Types of interactions to include in visualizations
        
        Returns:
            Dictionary mapping program names to generated filenames
        """
        import os
        from Bio.PDB import PDBIO
        
        if not self.structure or not self.interactions:
            print("No structure or interactions to visualize. Run map_interactions() first.")
            return {}
        
        # Filter interactions if needed
        if interaction_types:
            filtered_interactions = {
                k: v for k, v in self.interactions.items() if k in interaction_types
            }
        else:
            filtered_interactions = self.interactions
        
        # Define output files
        output_files = {}
        
        # Create PDB file for reference (needed by all visualization programs)
        pdb_file = f"{output_prefix}.pdb"
        
        # Export PDB if needed
        if not os.path.exists(pdb_file):
            io = PDBIO()
            io.set_structure(self.structure)
            io.save(pdb_file)
            logger.info(f"Saved structure to {pdb_file}")
        
        # Export PyMOL script
        pymol_file = f"{output_prefix}.pml"
        self._export_pymol_script(pymol_file, pdb_file, filtered_interactions)
        output_files["PyMOL"] = pymol_file
        
        # Export Chimera script
        chimera_file = f"{output_prefix}.cmd"
        self._export_chimera_script(chimera_file, pdb_file, filtered_interactions)
        output_files["Chimera"] = chimera_file
        
        # Export VMD script
        vmd_file = f"{output_prefix}.tcl"
        self._export_vmd_script(vmd_file, pdb_file, filtered_interactions)
        output_files["VMD"] = vmd_file
        
        # Export Molstar state file
        molstar_file = f"{output_prefix}.molj"
        self._export_molstar_state(molstar_file, pdb_file, filtered_interactions)
        output_files["Molstar"] = molstar_file
        
        logger.info(f"Exported visualization scripts: {', '.join(output_files.values())}")
        return output_files

    def _export_pymol_script(self, output_file: str, pdb_file: str, 
                        interactions: Dict[str, List[Dict]]):
        """Generate PyMOL script (.pml) to visualize interactions."""
        # Define colors for different interaction types
        interaction_colors = {
            'hydrogen_bonds': 'blue',
            'ionic_interactions': 'red',
            'salt_bridges': 'yellow',
            'hydrophobic_interactions': 'orange',
            'pi_stacking': 'purple',
            'pi_cation': 'green',
            'disulfide': 'gold'
        }
        
        with open(output_file, 'w') as f:
            # Header and load structure
            f.write(f"# PyMOL script for visualizing protein interactions\n")
            f.write(f"# Generated by PandaProt\n\n")
            
            # Load the PDB structure
            rel_path = os.path.relpath(pdb_file, os.path.dirname(output_file))
            f.write(f"load {rel_path}, protein\n")
            
            # Set up initial view
            f.write("hide everything\n")
            f.write("show cartoon\n")
            f.write("color gray80, protein\n")
            f.write("set cartoon_transparency, 0.5\n\n")
            
            # Setup groups for organization
            f.write("group Interactions\n")
            
            # Process each interaction type
            for interaction_type, interactions_list in interactions.items():
                if not interactions_list:
                    continue
                    
                color = interaction_colors.get(interaction_type, "gray50")
                pymol_name = interaction_type.replace("_", "")
                
                # Create a group for this interaction type
                f.write(f"group {pymol_name}, {pymol_name}_*\n")
                
                # Process each interaction
                for i, interaction in enumerate(interactions_list, 1):
                    try:
                        # Extract residue info based on interaction type
                        if interaction_type == 'hydrogen_bonds':
                            chain1 = interaction.get('donor_chain')
                            res1 = interaction.get('donor_residue', '').split()[-1] if ' ' in interaction.get('donor_residue', '') else interaction.get('donor_residue', '')
                            atom1 = interaction.get('donor_atom')
                            
                            chain2 = interaction.get('acceptor_chain')
                            res2 = interaction.get('acceptor_residue', '').split()[-1] if ' ' in interaction.get('acceptor_residue', '') else interaction.get('acceptor_residue', '')
                            atom2 = interaction.get('acceptor_atom')
                            
                            # Skip if missing data
                            if not all([chain1, res1, atom1, chain2, res2, atom2]):
                                continue
                                
                            sel1 = f"chain {chain1} and resi {res1} and name {atom1}"
                            sel2 = f"chain {chain2} and resi {res2} and name {atom2}"
                            
                        elif interaction_type in ['ionic_interactions', 'salt_bridges']:
                            if all(k in interaction for k in ['positive_residue', 'positive_chain']):
                                chain1 = interaction.get('positive_chain')
                                res1 = interaction.get('positive_residue', '').split()[-1] if ' ' in interaction.get('positive_residue', '') else interaction.get('positive_residue', '')
                                atom1 = interaction.get('positive_atom', '*')
                                
                                chain2 = interaction.get('negative_chain')
                                res2 = interaction.get('negative_residue', '').split()[-1] if ' ' in interaction.get('negative_residue', '') else interaction.get('negative_residue', '')
                                atom2 = interaction.get('negative_atom', '*')
                            else:
                                # Fallback to generic naming
                                chain1 = interaction.get('chain1')
                                res1 = interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                                atom1 = interaction.get('atom1', '*')
                                
                                chain2 = interaction.get('chain2')
                                res2 = interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                                atom2 = interaction.get('atom2', '*')
                                
                            # Skip if missing data
                            if not all([chain1, res1, chain2, res2]):
                                continue
                                
                            sel1 = f"chain {chain1} and resi {res1}"
                            if atom1 != '*':
                                sel1 += f" and name {atom1}"
                                
                            sel2 = f"chain {chain2} and resi {res2}"
                            if atom2 != '*':
                                sel2 += f" and name {atom2}"
                        
                        else:
                            # Generic handling for other interaction types
                            chain1 = interaction.get('chain1', interaction.get('residue1_chain'))
                            res1 = interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                            
                            chain2 = interaction.get('chain2', interaction.get('residue2_chain'))
                            res2 = interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                            
                            # Skip if missing data
                            if not all([chain1, res1, chain2, res2]):
                                continue
                                
                            sel1 = f"chain {chain1} and resi {res1}"
                            sel2 = f"chain {chain2} and resi {res2}"
                        
                        # Create a unique object name for this interaction
                        obj_name = f"{pymol_name}_{i:03d}"
                        
                        # Create PyMOL distance object
                        f.write(f"distance {obj_name}, ({sel1}), ({sel2})\n")
                        f.write(f"color {color}, {obj_name}\n")
                        
                        # Show relevant residues
                        f.write(f"show sticks, ({sel1}) or ({sel2})\n")
                    
                    except Exception as e:
                        logger.warning(f"Error processing interaction for PyMOL: {e}")
                        continue
                
                # Set properties for this interaction type
                f.write(f"set dash_gap, 0.3, {pymol_name}_*\n")
                f.write(f"set dash_width, 2.0, {pymol_name}_*\n")
                f.write(f"set dash_radius, 0.1, {pymol_name}_*\n\n")
            
            # Final view settings
            f.write("# Final view settings\n")
            f.write("zoom\n")
            f.write("set ray_shadows, 0\n")
            f.write("set depth_cue, 1\n")
            f.write("set valence, 0\n")
            f.write("bg_color white\n\n")
            
            # Add a legend
            f.write("# Create legend\n")
            f.write("set label_color, black\n")
            y_pos = 0
            for interaction_type, color in interaction_colors.items():
                if interaction_type in interactions and interactions[interaction_type]:
                    nice_name = interaction_type.replace('_', ' ').title()
                    f.write(f"pseudoatom leg_{interaction_type}, pos=[10, {y_pos}, 0]\n")
                    f.write(f"color {color}, leg_{interaction_type}\n")
                    f.write(f"label leg_{interaction_type}, \"{nice_name}\"\n")
                    y_pos -= 2
            
            logger.info(f"Saved PyMOL script to {output_file}")

    def _export_chimera_script(self, output_file: str, pdb_file: str, 
                            interactions: Dict[str, List[Dict]]):
        """Generate UCSF Chimera script (.cmd) to visualize interactions."""
        # Define colors for different interaction types
        interaction_colors = {
            'hydrogen_bonds': '0,0,255',  # blue
            'ionic_interactions': '255,0,0',  # red
            'salt_bridges': '255,255,0',  # yellow
            'hydrophobic_interactions': '255,165,0',  # orange
            'pi_stacking': '128,0,128',  # purple
            'pi_cation': '0,128,0',  # green
            'disulfide': '218,165,32'  # gold
        }
        
        with open(output_file, 'w') as f:
            # Header
            f.write("# UCSF Chimera script for visualizing protein interactions\n")
            f.write("# Generated by PandaProt\n\n")
            
            # Open the PDB file
            rel_path = os.path.relpath(pdb_file, os.path.dirname(output_file))
            f.write(f"open {rel_path}\n")
            
            # Initial display settings
            f.write("~display\n")  # hide everything
            f.write("cartoon\n")   # show cartoons
            f.write("color gray cartoon\n")  # color cartoons gray
            f.write("transparency 50 cartoon\n\n")  # make cartoons transparent
            
            # Process each interaction
            for interaction_type, interactions_list in interactions.items():
                if not interactions_list:
                    continue
                    
                color = interaction_colors.get(interaction_type, "128,128,128")  # default gray
                
                f.write(f"# {interaction_type.replace('_', ' ').title()}\n")
                
                for i, interaction in enumerate(interactions_list):
                    try:
                        # Extract residue info based on interaction type
                        if interaction_type == 'hydrogen_bonds':
                            chain1 = interaction.get('donor_chain')
                            res1 = interaction.get('donor_residue', '').split()[-1] if ' ' in interaction.get('donor_residue', '') else interaction.get('donor_residue', '')
                            atom1 = interaction.get('donor_atom')
                            
                            chain2 = interaction.get('acceptor_chain')
                            res2 = interaction.get('acceptor_residue', '').split()[-1] if ' ' in interaction.get('acceptor_residue', '') else interaction.get('acceptor_residue', '')
                            atom2 = interaction.get('acceptor_atom')
                            
                            # Skip if missing data
                            if not all([chain1, res1, atom1, chain2, res2, atom2]):
                                continue
                                
                            sel1 = f"#{chain1}:{res1}@{atom1}"
                            sel2 = f"#{chain2}:{res2}@{atom2}"
                            
                        elif interaction_type in ['ionic_interactions', 'salt_bridges']:
                            if all(k in interaction for k in ['positive_residue', 'positive_chain']):
                                chain1 = interaction.get('positive_chain')
                                res1 = interaction.get('positive_residue', '').split()[-1] if ' ' in interaction.get('positive_residue', '') else interaction.get('positive_residue', '')
                                atom1 = interaction.get('positive_atom', '')
                                
                                chain2 = interaction.get('negative_chain')
                                res2 = interaction.get('negative_residue', '').split()[-1] if ' ' in interaction.get('negative_residue', '') else interaction.get('negative_residue', '')
                                atom2 = interaction.get('negative_atom', '')
                            else:
                                # Fallback to generic naming
                                chain1 = interaction.get('chain1')
                                res1 = interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                                atom1 = interaction.get('atom1', '')
                                
                                chain2 = interaction.get('chain2')
                                res2 = interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                                atom2 = interaction.get('atom2', '')
                                
                            # Skip if missing data
                            if not all([chain1, res1, chain2, res2]):
                                continue
                                
                            sel1 = f"#{chain1}:{res1}"
                            if atom1:
                                sel1 += f"@{atom1}"
                                
                            sel2 = f"#{chain2}:{res2}"
                            if atom2:
                                sel2 += f"@{atom2}"
                        
                        else:
                            # Generic handling for other interaction types
                            chain1 = interaction.get('chain1', interaction.get('residue1_chain'))
                            res1 = interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                            
                            chain2 = interaction.get('chain2', interaction.get('residue2_chain'))
                            res2 = interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                            
                            # Skip if missing data
                            if not all([chain1, res1, chain2, res2]):
                                continue
                                
                            sel1 = f"#{chain1}:{res1}"
                            sel2 = f"#{chain2}:{res2}"
                        
                        # Display the interacting residues as sticks
                        f.write(f"display {sel1},{sel2}\n")
                        f.write(f"show {sel1},{sel2}\n")
                        
                        # Add a distance monitor between them
                        f.write(f"distance {sel1} {sel2}\n")
                        
                    except Exception as e:
                        logger.warning(f"Error processing interaction for Chimera: {e}")
                        continue
                
                # Set color for all distance monitors of this type
                f.write(f"color {color} pseudobonds\n\n")
            
            # Final settings
            f.write("# Final view settings\n")
            f.write("focus\n")
            f.write("set bg_color white\n")
            
            logger.info(f"Saved Chimera script to {output_file}")

    def _export_vmd_script(self, output_file: str, pdb_file: str, 
                        interactions: Dict[str, List[Dict]]):
        """Generate VMD script (.tcl) to visualize interactions."""
        # Define colors for different interaction types (VMD RGB: 0-1 range)
        interaction_colors = {
            'hydrogen_bonds': '0.0 0.0 1.0',  # blue
            'ionic_interactions': '1.0 0.0 0.0',  # red
            'salt_bridges': '1.0 1.0 0.0',  # yellow
            'hydrophobic_interactions': '1.0 0.65 0.0',  # orange
            'pi_stacking': '0.5 0.0 0.5',  # purple
            'pi_cation': '0.0 0.5 0.0',  # green
            'disulfide': '0.85 0.65 0.13'  # gold
        }
        
        with open(output_file, 'w') as f:
            # Header
            f.write("# VMD script for visualizing protein interactions\n")
            f.write("# Generated by PandaProt\n\n")
            
            # Open the PDB file
            rel_path = os.path.relpath(pdb_file, os.path.dirname(output_file))
            f.write(f"mol new {rel_path}\n")
            
            # Initial display settings
            f.write("mol delrep 0 top\n")  # delete default representation
            f.write("mol representation NewCartoon\n")
            f.write("mol color ColorID 8\n")  # gray
            f.write("mol selection \"all\"\n")
            f.write("mol material Transparent\n")
            f.write("mol addrep top\n\n")
            
            # Setup graphics object for labels
            f.write("# Create a graphics object for labels\n")
            f.write("draw color black\n")
            f.write("draw materials off\n")
            
            # Handle each interaction type
            interaction_count = 0
            
            for interaction_type, interactions_list in interactions.items():
                if not interactions_list:
                    continue
                    
                color = interaction_colors.get(interaction_type, "0.5 0.5 0.5")  # default gray
                nice_name = interaction_type.replace('_', ' ').title()
                
                f.write(f"# {nice_name}\n")
                
                # Draw a legend item
                f.write(f"draw color {color}\n")
                f.write(f"draw text {{30 {interaction_count * 1.2 + 2} 0}} \"{nice_name}\"\n")
                interaction_count += 1
                
                for i, interaction in enumerate(interactions_list):
                    try:
                        # Extract residue info based on interaction type
                        if interaction_type == 'hydrogen_bonds':
                            chain1 = interaction.get('donor_chain')
                            res1 = interaction.get('donor_residue', '').split()[-1] if ' ' in interaction.get('donor_residue', '') else interaction.get('donor_residue', '')
                            atom1 = interaction.get('donor_atom')
                            
                            chain2 = interaction.get('acceptor_chain')
                            res2 = interaction.get('acceptor_residue', '').split()[-1] if ' ' in interaction.get('acceptor_residue', '') else interaction.get('acceptor_residue', '')
                            atom2 = interaction.get('acceptor_atom')
                            
                            # Skip if missing data
                            if not all([chain1, res1, atom1, chain2, res2, atom2]):
                                continue
                                
                            sel1 = f"chain {chain1} and resid {res1} and name {atom1}"
                            sel2 = f"chain {chain2} and resid {res2} and name {atom2}"
                            
                        elif interaction_type in ['ionic_interactions', 'salt_bridges']:
                            if all(k in interaction for k in ['positive_residue', 'positive_chain']):
                                chain1 = interaction.get('positive_chain')
                                res1 = interaction.get('positive_residue', '').split()[-1] if ' ' in interaction.get('positive_residue', '') else interaction.get('positive_residue', '')
                                atom1 = interaction.get('positive_atom')
                                
                                chain2 = interaction.get('negative_chain')
                                res2 = interaction.get('negative_residue', '').split()[-1] if ' ' in interaction.get('negative_residue', '') else interaction.get('negative_residue', '')
                                atom2 = interaction.get('negative_atom')
                            else:
                                # Fallback to generic naming
                                chain1 = interaction.get('chain1')
                                res1 = interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                                atom1 = interaction.get('atom1')
                                
                                chain2 = interaction.get('chain2')
                                res2 = interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                                atom2 = interaction.get('atom2')
                                
                            # Skip if missing data
                            if not all([chain1, res1, chain2, res2]):
                                continue
                            
                            # Use all atoms if specific atoms not provided
                            if not atom1:
                                sel1 = f"chain {chain1} and resid {res1}"
                            else:
                                sel1 = f"chain {chain1} and resid {res1} and name {atom1}"
                                
                            if not atom2:
                                sel2 = f"chain {chain2} and resid {res2}"
                            else:
                                sel2 = f"chain {chain2} and resid {res2} and name {atom2}"
                        
                        else:
                            # Generic handling for other interaction types
                            chain1 = interaction.get('chain1', interaction.get('residue1_chain'))
                            res1 = interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                            
                            chain2 = interaction.get('chain2', interaction.get('residue2_chain'))
                            res2 = interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                            
                            # Skip if missing data
                            if not all([chain1, res1, chain2, res2]):
                                continue
                                
                            sel1 = f"chain {chain1} and resid {res1}"
                            sel2 = f"chain {chain2} and resid {res2}"
                        
                        # Show the interacting residues as sticks
                        f.write(f"# Show residues involved in interaction {i+1}\n")
                        f.write(f"mol representation Licorice 0.3 12.0 12.0\n")
                        f.write(f"mol color Element\n")
                        f.write(f"mol selection \"({sel1}) or ({sel2})\"\n")
                        f.write(f"mol material Opaque\n")
                        f.write(f"mol addrep top\n")
                        
                        # Create a dashed line for the interaction
                        f.write(f"set sel1 [atomselect top \"{sel1}\"]\n")
                        f.write(f"set sel2 [atomselect top \"{sel2}\"]\n")
                        f.write(f"set coord1 [lindex [$sel1 get {{{atom1 if atom1 else 'x y z'}}}] 0]\n")
                        f.write(f"set coord2 [lindex [$sel2 get {{{atom2 if atom2 else 'x y z'}}}] 0]\n")
                        f.write(f"draw color {color}\n")
                        f.write(f"draw line $coord1 $coord2 style dashed width 3\n")
                        f.write(f"$sel1 delete\n")
                        f.write(f"$sel2 delete\n\n")
                        
                    except Exception as e:
                        logger.warning(f"Error processing interaction for VMD: {e}")
                        continue
            
            # Final view settings
            f.write("# Final view settings\n")
            f.write("color Display Background white\n")
            f.write("display projection Orthographic\n")
            f.write("display depthcue off\n")
            f.write("display nearclip set 0.01\n")
            f.write("axes location Off\n")
            f.write("stage location Off\n")
            f.write("display resize 800 800\n")
            f.write("display resetview\n")
            
            logger.info(f"Saved VMD script to {output_file}")

    def _export_molstar_state(self, output_file: str, pdb_file: str, 
                         interactions: Dict[str, List[Dict]]):
        """
        Generate Molstar state file (.molj) to visualize interactions.
        
        This creates a file that can be directly loaded into the Molstar web viewer
        at https://molstar.org/viewer/
        """
        import json
        import os
        
        # Define colors for different interaction types (hexadecimal integers)
        interaction_colors = {
            'hydrogen_bonds': 0x0000FF,  # blue
            'ionic_interactions': 0xFF0000,  # red
            'salt_bridges': 0xFFFF00,  # yellow
            'hydrophobic_interactions': 0xFFA500,  # orange
            'pi_stacking': 0x800080,  # purple
            'pi_cation': 0x008000,  # green
            'disulfide': 0xDAA520,  # gold
            'water_mediated': 0x00BFFF,  # deep sky blue
            'ch_pi': 0x20B2AA,  # light sea green
            'halogen_bond': 0x00CED1,  # dark turquoise
            'van_der_waals': 0x708090,  # slate gray
            'amide_aromatic': 0x9370DB,  # medium purple
            'amide_amide': 0xFF69B4   # hot pink
        }
        
        # Get the PDB file basename for reference in the state
        pdb_basename = os.path.basename(pdb_file)
        
        # Create the base Molstar state structure
        state = {
            "timestamp": int(time.time() * 1000),  # Current time in milliseconds
            "format": {
                "name": "molstar-state",
                "version": "3"
            },
            "state": {
                "viewport": {
                    "width": 800,
                    "height": 600
                },
                "camera": {
                    "mode": "perspective",
                    "fov": 45,
                    "position": {"x": 0, "y": 0, "z": 100}
                },
                "theme": {
                    "name": "light"
                },
                "settings": {
                    "rendering": {
                        "quality": "auto",
                        "background": {
                            "color": 0xFFFFFF  # White background
                        }
                    }
                },
                "snapshots": [],
                "components": {
                    "root": {
                        "type": "root",
                        "transforms": [],
                        "cell": {"state": {}, "params": {}, "reprList": [], "current": -1},
                        "assignments": [],
                        "children": [
                            {
                                "type": "data-component",
                                "dataId": "0",
                                "label": "Main Structure",
                                "params": {},
                                "cell": {
                                    "state": {},
                                    "params": {},
                                    "reprList": [
                                        {
                                            "id": "cartoon-representation",
                                            "type": "representation",
                                            "params": {
                                                "type": "cartoon",
                                                "params": {
                                                    "quality": "auto",
                                                    "alpha": 0.7
                                                }
                                            }
                                        }
                                    ],
                                    "current": 0
                                },
                                "transforms": []
                            }
                        ]
                    }
                },
                "behaviors": {},
                "plugin": {}
            },
            "data": [
                {
                    "id": "0",
                    "kind": "model",
                    "format": "pdb",
                    "source": {
                        "name": pdb_basename,
                        "type": "file"
                    }
                }
            ],
            "annotations": {
                "groups": [],
                "annotations": []
            }
        }
        
        # Create a group for each interaction type
        group_index = 0
        annotation_index = 0
        
        for interaction_type, interactions_list in interactions.items():
            if not interactions_list:
                continue
                
            # Get color for this interaction type
            color = interaction_colors.get(interaction_type, 0x808080)  # Default gray
            nice_name = interaction_type.replace('_', ' ').title()
            
            # Add a group for this interaction type
            group = {
                "id": f"group-{group_index}",
                "color": color,
                "label": nice_name,
                "type": "group"
            }
            state["annotations"]["groups"].append(group)
            
            # Add each interaction as an annotation
            for interaction in interactions_list:
                try:
                    # Extract residue info based on interaction type
                    if interaction_type == 'hydrogen_bonds':
                        chain1 = interaction.get('donor_chain')
                        res1 = interaction.get('donor_residue', '').split()[-1] if ' ' in interaction.get('donor_residue', '') else interaction.get('donor_residue', '')
                        atom1 = interaction.get('donor_atom')
                        
                        chain2 = interaction.get('acceptor_chain')
                        res2 = interaction.get('acceptor_residue', '').split()[-1] if ' ' in interaction.get('acceptor_residue', '') else interaction.get('acceptor_residue', '')
                        atom2 = interaction.get('acceptor_atom')
                        
                        # Skip if missing data
                        if not all([chain1, res1, atom1, chain2, res2, atom2]):
                            continue
                        
                        target1 = {
                            "labelAsId": False, 
                            "entityId": "0", 
                            "chainId": chain1,
                            "residueNumber": int(res1) if res1.isdigit() else res1, 
                            "atomName": atom1
                        }
                        
                        target2 = {
                            "labelAsId": False, 
                            "entityId": "0", 
                            "chainId": chain2,
                            "residueNumber": int(res2) if res2.isdigit() else res2, 
                            "atomName": atom2
                        }
                        
                    elif interaction_type in ['ionic_interactions', 'salt_bridges']:
                        # Similar extraction as in other functions
                        if all(k in interaction for k in ['positive_residue', 'positive_chain']):
                            chain1 = interaction.get('positive_chain')
                            res1 = interaction.get('positive_residue', '').split()[-1] if ' ' in interaction.get('positive_residue', '') else interaction.get('positive_residue', '')
                            atom1 = interaction.get('positive_atom')
                            
                            chain2 = interaction.get('negative_chain')
                            res2 = interaction.get('negative_residue', '').split()[-1] if ' ' in interaction.get('negative_residue', '') else interaction.get('negative_residue', '')
                            atom2 = interaction.get('negative_atom')
                        else:
                            # Fallback to generic naming
                            chain1 = interaction.get('chain1')
                            res1 = interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                            atom1 = interaction.get('atom1')
                            
                            chain2 = interaction.get('chain2')
                            res2 = interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                            atom2 = interaction.get('atom2')
                        
                        # Skip if missing essential data
                        if not all([chain1, res1, chain2, res2]):
                            continue
                        
                        # Use CA atoms if specific atoms not provided
                        if not atom1:
                            atom1 = "CA"
                        if not atom2:
                            atom2 = "CA"
                        
                        target1 = {
                            "labelAsId": False, 
                            "entityId": "0", 
                            "chainId": chain1,
                            "residueNumber": int(res1) if res1.isdigit() else res1, 
                            "atomName": atom1
                        }
                        
                        target2 = {
                            "labelAsId": False, 
                            "entityId": "0", 
                            "chainId": chain2,
                            "residueNumber": int(res2) if res2.isdigit() else res2, 
                            "atomName": atom2
                        }
                    
                    else:
                        # Generic handling for other interaction types
                        chain1 = interaction.get('chain1', interaction.get('residue1_chain'))
                        res1 = interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                        atom1 = interaction.get('atom1')
                        
                        chain2 = interaction.get('chain2', interaction.get('residue2_chain'))
                        res2 = interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                        atom2 = interaction.get('atom2')
                        
                        # Skip if missing essential data
                        if not all([chain1, res1, chain2, res2]):
                            continue
                        
                        # Use CA atoms if specific atoms not provided
                        if not atom1:
                            atom1 = "CA"
                        if not atom2:
                            atom2 = "CA"
                        
                        target1 = {
                            "labelAsId": False, 
                            "entityId": "0", 
                            "chainId": chain1,
                            "residueNumber": int(res1) if res1.isdigit() else res1, 
                            "atomName": atom1
                        }
                        
                        target2 = {
                            "labelAsId": False, 
                            "entityId": "0", 
                            "chainId": chain2,
                            "residueNumber": int(res2) if res2.isdigit() else res2, 
                            "atomName": atom2
                        }
                    
                    # Create the distance annotation
                    annotation = {
                        "id": f"annotation-{annotation_index}",
                        "type": "distance",
                        "targets": [target1, target2],
                        "color": color,
                        "label": f"{nice_name} {chain1}:{res1}-{chain2}:{res2}",
                        "groupId": f"group-{group_index}"
                    }
                    
                    state["annotations"]["annotations"].append(annotation)
                    annotation_index += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing interaction for Molstar: {e}")
                    continue
            
            # Increment group index for the next type
            group_index += 1
        
        # Also add a representation to highlight all interacting residues
        all_interacting_residues = set()
        
        # Collect all residues involved in interactions
        for interaction_type, interactions_list in interactions.items():
            for interaction in interactions_list:
                try:
                    # Extract residue info by interaction type (simplified for clarity)
                    if interaction_type == 'hydrogen_bonds':
                        chain1, res1 = interaction.get('donor_chain'), interaction.get('donor_residue', '').split()[-1] if ' ' in interaction.get('donor_residue', '') else interaction.get('donor_residue', '')
                        chain2, res2 = interaction.get('acceptor_chain'), interaction.get('acceptor_residue', '').split()[-1] if ' ' in interaction.get('acceptor_residue', '') else interaction.get('acceptor_residue', '')
                    elif interaction_type in ['ionic_interactions', 'salt_bridges']:
                        if all(k in interaction for k in ['positive_residue', 'positive_chain']):
                            chain1, res1 = interaction.get('positive_chain'), interaction.get('positive_residue', '').split()[-1] if ' ' in interaction.get('positive_residue', '') else interaction.get('positive_residue', '')
                            chain2, res2 = interaction.get('negative_chain'), interaction.get('negative_residue', '').split()[-1] if ' ' in interaction.get('negative_residue', '') else interaction.get('negative_residue', '')
                        else:
                            chain1, res1 = interaction.get('chain1'), interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                            chain2, res2 = interaction.get('chain2'), interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                    else:
                        chain1, res1 = interaction.get('chain1', interaction.get('residue1_chain')), interaction.get('residue1', '').split()[-1] if ' ' in interaction.get('residue1', '') else interaction.get('residue1', '')
                        chain2, res2 = interaction.get('chain2', interaction.get('residue2_chain')), interaction.get('residue2', '').split()[-1] if ' ' in interaction.get('residue2', '') else interaction.get('residue2', '')
                    
                    # Add to set if we have valid data
                    if chain1 and res1:
                        all_interacting_residues.add((chain1, res1))
                    if chain2 and res2:
                        all_interacting_residues.add((chain2, res2))
                        
                except Exception:
                    continue
        
        # Add a ball-and-stick representation for interacting residues
        if all_interacting_residues:
            # Create a Molstar selection expression for all interacting residues
            selection_parts = []
            for chain, res in all_interacting_residues:
                res_num = res if res.isdigit() else f"'{res}'"  # Handle non-numeric residue IDs
                selection_parts.append(f"(chain {chain} and residue {res_num})")
            
            selection_expression = " or ".join(selection_parts)
            
            # Add a new representation for these residues
            state["state"]["components"]["root"]["children"][0]["cell"]["reprList"].append({
                "id": "interacting-residues",
                "type": "representation",
                "params": {
                    "type": "ball-and-stick",
                    "params": {
                        "quality": "auto",
                        "alpha": 1.0,
                        "aspectRatio": 2.0,
                        "bondSpacing": 1.0,
                        "bondOrder": True
                    }
                }
            })
        
        # Write the state file
        try:
            with open(output_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved Molstar state file to {output_file}")
        except Exception as e:
            logger.error(f"Error saving Molstar state file: {e}")
        
        return output_file