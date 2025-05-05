# pandaprot/main.py
"""
Enhanced command-line interface for PandaProt with additional interaction types.
"""

import argparse
import os
from typing import List, Optional

from .core import PandaProt


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='PandaProt: Comprehensive Protein Interaction Mapper')
    parser.add_argument('pdb_file', help='Path to PDB file')
    
    # Basic options
    parser.add_argument('--chains', nargs='+', help='Chains to analyze (e.g., A B)')
    parser.add_argument('--3d-plot', action='store_true', dest='plot_3d', 
                      help='Generate 3D visualization')
    parser.add_argument('--export-vis', action='store_true',
                   help='Export visualization files for PyMOL, Chimera, VMD, and Molstar')
    parser.add_argument('--report', action='store_true', 
                      help='Generate detailed interaction report')
    parser.add_argument('--network', action='store_true',
                      help='Generate interaction network visualization')
    parser.add_argument('--output', '-o', help='Output file prefix')
    
    # Interaction filtering options
    parser.add_argument('--distance-cutoff', type=float, default=4.5,
                      help='Distance cutoff for interaction detection (default: 4.5Å)')
    parser.add_argument('--include-intrachain', action='store_true',
                      help='Include intra-chain interactions')
    
    # Interaction type selection
    interaction_group = parser.add_argument_group('Interaction Types')
    interaction_group.add_argument('--all-interactions', action='store_true',
                                 help='Map all interaction types (default)')
    
    # Standard interactions
    interaction_group.add_argument('--standard-only', action='store_true',
                                 help='Map only standard interactions (H-bonds, ionic, hydrophobic, pi-stacking, salt bridges)')
    interaction_group.add_argument('--hydrogen-bonds', action='store_true',
                                 help='Map hydrogen bonds')
    interaction_group.add_argument('--ionic', action='store_true',
                                 help='Map ionic interactions')
    interaction_group.add_argument('--hydrophobic', action='store_true',
                                 help='Map hydrophobic interactions')
    interaction_group.add_argument('--pi-stacking', action='store_true',
                                 help='Map pi-pi stacking interactions')
    interaction_group.add_argument('--pi-cation', action='store_true',
                                 help='Map pi-cation interactions')
    interaction_group.add_argument('--salt-bridges', action='store_true',
                                 help='Map salt bridges')
    
    # Enhanced interactions
    interaction_group.add_argument('--cation-pi', action='store_true',
                                 help='Map cation-pi interactions')
    interaction_group.add_argument('--ch-pi', action='store_true',
                                 help='Map CH-pi interactions')
    interaction_group.add_argument('--disulfide', action='store_true',
                                 help='Map disulfide bridges')
    interaction_group.add_argument('--sulfur-aromatic', action='store_true',
                                 help='Map sulfur-aromatic interactions')
    interaction_group.add_argument('--water-mediated', action='store_true',
                                 help='Map water-mediated interactions')
    interaction_group.add_argument('--metal-coordination', action='store_true',
                                 help='Map metal-coordinated bonds')
    interaction_group.add_argument('--halogen-bonds', action='store_true',
                                 help='Map halogen bonds')
    interaction_group.add_argument('--amide-aromatic', action='store_true',
                                 help='Map amide-aromatic interactions')
    interaction_group.add_argument('--van-der-waals', action='store_true',
                                 help='Map van der Waals interactions')
    interaction_group.add_argument('--amide-amide', action='store_true',
                                 help='Map amide-amide hydrogen bonds')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--statistics', action='store_true',
                              help='Generate interaction statistics')
    analysis_group.add_argument('--residue-summary', action='store_true',
                              help='Generate residue interaction summary')
    
    args = parser.parse_args()
    
    # Initialize PandaProt
    analyzer = PandaProt(args.pdb_file, args.chains)
    
    # Determine which interaction types to map
    interaction_types = []
    
    if args.all_interactions or (not args.standard_only and 
                               not args.hydrogen_bonds and
                               not args.ionic and
                               not args.hydrophobic and
                               not args.pi_stacking and
                               not args.pi_cation and
                               not args.salt_bridges and
                               not args.cation_pi and
                               not args.ch_pi and
                               not args.disulfide and
                               not args.sulfur_aromatic and
                               not args.water_mediated and
                               not args.metal_coordination and
                               not args.halogen_bonds and
                               not args.amide_aromatic and
                               not args.van_der_waals and
                               not args.amide_amide):
        # Default: map all interactions
        print("Mapping all interaction types...")
        interactions = analyzer.map_interactions(
            distance_cutoff=args.distance_cutoff,
            include_intrachain=args.include_intrachain
        )
    else:
        # Map selected interaction types
        if args.standard_only:
            interaction_types = ['hydrogen_bonds', 'ionic_interactions', 'hydrophobic_interactions',
                                'pi_stacking', 'pi_cation', 'salt_bridges']
        else:
            if args.hydrogen_bonds:
                interaction_types.append('hydrogen_bonds')
            if args.ionic:
                interaction_types.append('ionic_interactions')
            if args.hydrophobic:
                interaction_types.append('hydrophobic_interactions')
            if args.pi_stacking:
                interaction_types.append('pi_stacking')
            if args.pi_cation:
                interaction_types.append('pi_cation')
            if args.salt_bridges:
                interaction_types.append('salt_bridges')
            if args.cation_pi:
                interaction_types.append('cation_pi')
            if args.ch_pi:
                interaction_types.append('ch_pi')
            if args.disulfide:
                interaction_types.append('disulfide_bridges')
            if args.sulfur_aromatic:
                interaction_types.append('sulfur_aromatic')
            if args.water_mediated:
                interaction_types.append('water_mediated')
            if args.metal_coordination:
                interaction_types.append('metal_coordination')
            if args.halogen_bonds:
                interaction_types.append('halogen_bonds')
            if args.amide_aromatic:
                interaction_types.append('amide_aromatic')
            if args.van_der_waals:
                interaction_types.append('van_der_waals')
            if args.amide_amide:
                interaction_types.append('amide_amide')
        
        print(f"Mapping selected interaction types: {', '.join(interaction_types)}")
        interactions = analyzer.map_interactions(
            distance_cutoff=args.distance_cutoff,
            include_intrachain=args.include_intrachain
        )
    
    # Generate outputs based on arguments
    if args.plot_3d:
        output_file = f"{args.output}_3d.html" if args.output else "pandaprot_3d.html"
        analyzer.visualize_3d(output_file, interaction_types)
    
    if args.report:
        output_file = f"{args.output}_report.csv" if args.output else "pandaprot_report.csv"
        analyzer.generate_report(output_file, interaction_types)
    
    if args.network:
        output_file = f"{args.output}_network.png" if args.output else "pandaprot_network.png"
        analyzer.create_interaction_network(output_file, interaction_types)
    
    # In your main script
    if args.export_vis:
        output_prefix = args.output.replace('.html', '').replace('.pdb', '') if args.output else 'pandaprot'
        vis_files = analyzer.export_visualization_scripts(output_prefix, interaction_types)
        print("\nExported visualization files:")
        for program, filename in vis_files.items():
            print(f" - {program}: {filename}")
    
    # Print summary of exported files
    print("\nExported visualization files:")
    for program, filename in vis_files.items():
        print(f" - {program}: {filename}")
    
    if args.statistics:
        stats = analyzer.get_interaction_statistics()
        print("\nInteraction Statistics:")
        print(f"Total interactions: {stats['total_interactions']}")
        
        print("\nInteractions by type:")
        for interaction_type, count in stats['by_type'].items():
            print(f"  - {interaction_type}: {count}")
        
        print("\nAverage distances:")
        for interaction_type, avg_dist in stats['avg_distances'].items():
            print(f"  - {interaction_type}: {avg_dist:.2f}Å")
        
        print("\nTop 10 most interactive residues:")
        for i, (residue, count) in enumerate(list(stats['residue_frequencies'].items())[:10]):
            print(f"  {i+1}. {residue}: {count} interactions")
        
        if args.output:
            stats_file = f"{args.output}_stats.txt"
            with open(stats_file, 'w') as f:
                f.write(f"Interaction Statistics:\n")
                f.write(f"Total interactions: {stats['total_interactions']}\n\n")
                
                f.write("Interactions by type:\n")
                for interaction_type, count in stats['by_type'].items():
                    f.write(f"  - {interaction_type}: {count}\n")
                
                f.write("\nAverage distances:\n")
                for interaction_type, avg_dist in stats['avg_distances'].items():
                    f.write(f"  - {interaction_type}: {avg_dist:.2f}Å\n")
                
                f.write("\nMost interactive residues:\n")
                for residue, count in stats['residue_frequencies'].items():
                    f.write(f"  - {residue}: {count} interactions\n")
            
            print(f"Statistics saved to {stats_file}")
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()