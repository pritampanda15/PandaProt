# pandaprot/reports/generator.py
"""
Module for generating detailed reports of protein interactions.
"""

from typing import Dict, List, Optional
import pandas as pd
from Bio.PDB import Structure


def create_interaction_report(interactions: Dict[str, List[Dict]]) -> Optional[pd.DataFrame]:
    """
    Create a detailed report of all interactions.
    
    Args:
        interactions: Dictionary of interactions
        
    Returns:
        Pandas DataFrame containing the report, or None if no interactions
    """
    # Create dataframes for each interaction type
    dfs = []
    
    # Process hydrogen bonds
    if 'hydrogen_bonds' in interactions and interactions['hydrogen_bonds']:
        h_bonds = []
        for hb in interactions['hydrogen_bonds']:
            h_bonds.append({
                'Interaction_Type': 'Hydrogen Bond',
                'Chain1': hb['donor_chain'],
                'Residue1': hb['donor_residue'],
                'Atom1': hb['donor_atom'],
                'Chain2': hb['acceptor_chain'],
                'Residue2': hb['acceptor_residue'],
                'Atom2': hb['acceptor_atom'],
                'Distance_Å': round(hb['distance'], 2)
            })
        if h_bonds:
            dfs.append(pd.DataFrame(h_bonds))
    
    # Process ionic interactions
    if 'ionic_interactions' in interactions and interactions['ionic_interactions']:
        ionic = []
        for ion in interactions['ionic_interactions']:
            ionic.append({
                'Interaction_Type': 'Ionic Interaction',
                'Chain1': ion.get('positive_chain', ion.get('chain1', 'Unknown')),
                'Residue1': ion.get('positive_residue', ion.get('residue1', 'Unknown')),
                'Atom1': ion.get('positive_atom', ion.get('atom1', 'Unknown')),
                'Chain2': ion.get('negative_chain', ion.get('chain2', 'Unknown')),
                'Residue2': ion.get('negative_residue', ion.get('residue2', 'Unknown')),
                'Atom2': ion.get('negative_atom', ion.get('atom2', 'Unknown')),
                'Distance_Å': round(ion['distance'], 2)
            })
        if ionic:
            dfs.append(pd.DataFrame(ionic))
    
    # Process hydrophobic interactions
    if 'hydrophobic_interactions' in interactions and interactions['hydrophobic_interactions']:
        hydrophobic = []
        for hp in interactions['hydrophobic_interactions']:
            hydrophobic.append({
                'Interaction_Type': 'Hydrophobic',
                'Chain1': hp['chain1'],
                'Residue1': hp['residue1'],
                'Atom1': hp['atom1'],
                'Chain2': hp['chain2'],
                'Residue2': hp['residue2'],
                'Atom2': hp['atom2'],
                'Distance_Å': round(hp['distance'], 2)
            })
        if hydrophobic:
            dfs.append(pd.DataFrame(hydrophobic))
    
    # Process pi stacking
    if 'pi_stacking' in interactions and interactions['pi_stacking']:
        pi_stacking = []
        for ps in interactions['pi_stacking']:
            pi_stacking.append({
                'Interaction_Type': f"Pi-Pi Stacking ({ps.get('stacking_type', 'Unknown')})",
                'Chain1': ps['aromatic_chain'] if 'aromatic_chain' in ps else ps['chain1'],
                'Residue1': ps['aromatic_residue'] if 'aromatic_residue' in ps else ps['residue1'],
                'Ring1': ps.get('ring1', 'Unknown'),
                'Chain2': ps['aromatic_chain2'] if 'aromatic_chain2' in ps else ps['chain2'],
                'Residue2': ps['aromatic_residue2'] if 'aromatic_residue2' in ps else ps['residue2'],
                'Ring2': ps.get('ring2', 'Unknown'),
                'Distance_Å': round(ps['distance'], 2),
                'Angle_degrees': round(ps.get('angle', 0), 2)
            })
        if pi_stacking:
            dfs.append(pd.DataFrame(pi_stacking))
    
    # Process pi cation
    if 'pi_cation' in interactions and interactions['pi_cation']:
        pi_cation = []
        for pc in interactions['pi_cation']:
            pi_cation.append({
                'Interaction_Type': 'Pi-Cation',
                'Chain1': pc['aromatic_chain'],
                'Residue1': pc['aromatic_residue'],
                'Ring': pc.get('ring', 'Unknown'),
                'Chain2': pc['cationic_chain'],
                'Residue2': pc['cationic_residue'],
                'Cationic_Atom': pc['cationic_atom'],
                'Distance_Å': round(pc['distance'], 2)
            })
        if pi_cation:
            dfs.append(pd.DataFrame(pi_cation))
    
    # Process salt bridges
    if 'salt_bridges' in interactions and interactions['salt_bridges']:
        salt_bridges = []
        for sb in interactions['salt_bridges']:
            salt_bridges.append({
                'Interaction_Type': 'Salt Bridge',
                'Chain1': sb['acidic_chain'],
                'Residue1': sb['acidic_residue'],
                'Atom1': sb['acidic_atom'],
                'Chain2': sb['basic_chain'],
                'Residue2': sb['basic_residue'],
                'Atom2': sb['basic_atom'],
                'Distance_Å': round(sb['distance'], 2)
            })
        if salt_bridges:
            dfs.append(pd.DataFrame(salt_bridges))
    # # Process other interactions
    #     if 'amide_amide' in interactions and interactions['amide_amide']:
    #         amide_amide = []
    #         for aa in interactions['amide_amide']:
    #             amide_amide.append({
    #                 'Interaction_Type': 'Amide-Amide',
    #                 'Chain1': aa['chain1'],
    #                 'Residue1': aa['residue1'],
    #                 'Atom1': aa['atom1'],
    #                 'Chain2': aa['chain2'],
    #                 'Residue2': aa['residue2'],
    #                 'Atom2': aa['atom2'],
    #                 'Distance_Å': round(aa['distance'], 2)
    #             })
    #         if amide_amide:
    #             dfs.append(pd.DataFrame(amide_amide))
    #     if [amide_aromatic]:
    #     amide_aromatic = []
    #     for aa in interactions['amide_aromatic']:
    #         amide_aromatic.append({
    #             'Interaction_Type': 'Amide-Aromatic',
    #             'Chain1': aa['chain1'],
    #             'Residue1': aa['residue1'],
    #             'Atom1': aa['atom1'],
    #             'Chain2': aa['chain2'],
    #             'Residue2': aa['residue2'],
    #             'Atom2': aa['atom2'],
    #             'Distance_Å': round(aa['distance'], 2)
    #         })          
    #     if amide_aromatic:
    #         dfs.append(pd.DataFrame(amide_aromatic))
    # if 'ch_pi' in interactions and interactions['ch_pi']:
    #     ch_pi = []
    #     for cp in interactions['ch_pi']:
    #         ch_pi.append({
    #             'Interaction_Type': 'C-H...Pi',
    #             'Chain1': cp['ch_chain'],
    #             'Residue1': cp['ch_residue'],
    #             'Atom1': cp['ch_atom'],
    #             'Chain2': cp['pi_chain'],
    #             'Residue2': cp['pi_residue'],
    #             'Ring': cp.get('ring', 'Unknown'),
    #             'Distance_Å': round(cp['distance'], 2)
    #         })
    #     if ch_pi:
    #         dfs.append(pd.DataFrame(ch_pi))
    if dfs:
        report_df = pd.concat(dfs, ignore_index=True)
        return report_df
    else:
        print("No interactions found to report.")
        return None


def create_residue_summary(interactions: Dict[str, List[Dict]]) -> Optional[pd.DataFrame]:
    """
    Create a summary of interactions by residue.
    
    Args:
        interactions: Dictionary of interactions
        
    Returns:
        Pandas DataFrame containing the summary by residue
    """
    # Create a report dataframe first
    report_df = create_interaction_report(interactions)
    
    if report_df is None:
        return None
    
    # Create residue summaries
    residue_summary = {}
    
    # Process Chain1-Residue1 pairs
    for _, row in report_df.iterrows():
        key1 = f"{row['Chain1']}:{row['Residue1']}"
        
        if key1 not in residue_summary:
            residue_summary[key1] = {
                'Chain': row['Chain1'],
                'Residue': row['Residue1'],
                'H_Bonds': 0,
                'Ionic': 0,
                'Hydrophobic': 0,
                'Pi_Stacking': 0,
                'Pi_Cation': 0,
                'Salt_Bridge': 0,
                'Total': 0
            }
        
        # Increment counts based on interaction type
        if 'Hydrogen Bond' in row['Interaction_Type']:
            residue_summary[key1]['H_Bonds'] += 1
        elif 'Ionic' in row['Interaction_Type']:
            residue_summary[key1]['Ionic'] += 1
        elif 'Hydrophobic' in row['Interaction_Type']:
            residue_summary[key1]['Hydrophobic'] += 1
        elif 'Pi-Pi' in row['Interaction_Type']:
            residue_summary[key1]['Pi_Stacking'] += 1
        elif 'Pi-Cation' in row['Interaction_Type']:
            residue_summary[key1]['Pi_Cation'] += 1
        elif 'Salt Bridge' in row['Interaction_Type']:
            residue_summary[key1]['Salt_Bridge'] += 1
            
        residue_summary[key1]['Total'] += 1
        
        # Process Chain2-Residue2 pairs
        key2 = f"{row['Chain2']}:{row['Residue2']}"
        
        if key2 not in residue_summary:
            residue_summary[key2] = {
                'Chain': row['Chain2'],
                'Residue': row['Residue2'],
                'H_Bonds': 0,
                'Ionic': 0,
                'Hydrophobic': 0,
                'Pi_Stacking': 0,
                'Pi_Cation': 0,
                'Salt_Bridge': 0,
                'Total': 0
            }
        
        # Increment counts based on interaction type (same as above)
        if 'Hydrogen Bond' in row['Interaction_Type']:
            residue_summary[key2]['H_Bonds'] += 1
        elif 'Ionic' in row['Interaction_Type']:
            residue_summary[key2]['Ionic'] += 1
        elif 'Hydrophobic' in row['Interaction_Type']:
            residue_summary[key2]['Hydrophobic'] += 1
        elif 'Pi-Pi' in row['Interaction_Type']:
            residue_summary[key2]['Pi_Stacking'] += 1
        elif 'Pi-Cation' in row['Interaction_Type']:
            residue_summary[key2]['Pi_Cation'] += 1
        elif 'Salt Bridge' in row['Interaction_Type']:
            residue_summary[key2]['Salt_Bridge'] += 1
            
        residue_summary[key2]['Total'] += 1
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(list(residue_summary.values()))
    
    # Sort by total interactions
    summary_df = summary_df.sort_values('Total', ascending=False)
    
    return summary_df


def export_to_csv(df: pd.DataFrame, output_file: str) -> None:
    """
    Export a DataFrame to CSV.
    
    Args:
        df: DataFrame to export
        output_file: Output file path
    """
    df.to_csv(output_file, index=False)
    print(f"Report exported to {output_file}")


def export_to_excel(df: pd.DataFrame, output_file: str) -> None:
    """
    Export a DataFrame to Excel.
    
    Args:
        df: DataFrame to export
        output_file: Output file path
    """
    df.to_excel(output_file, index=False)
    print(f"Report exported to {output_file}")