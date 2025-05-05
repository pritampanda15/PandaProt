# PandaProt

A comprehensive Python application for mapping and visualizing molecular interactions at protein/nucleic acid interfaces. A tool for mapping protein-protein, protein-nucleic acid, and antigen-antibody interactions.

<p align="center">
  <img src="https://github.com/pritampanda15/PandaProt/blob/main/logo/logo.png" alt="PandaProt Logo" width="400"> 
</p>
<p align="center">
  <a href="https://pypi.org/project/pandaprot/">
    <img src="https://img.shields.io/pypi/v/pandaprot.svg" alt="PyPI Version">
  </a>
  <a href="https://github.com/pritampanda15/PandaProt/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/pritampanda15/PandaProt" alt="License">
  </a>
  <a href="https://github.com/pritampanda15/PandaProt/stargazers">
    <img src="https://img.shields.io/github/stars/pritampanda15/PandaProt?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/pritampanda15/PandaProt/issues">
    <img src="https://img.shields.io/github/issues/pritampanda15/PandaProt" alt="GitHub Issues">
  </a>
  <a href="https://github.com/pritampanda15/PandaProt/network/members">
    <img src="https://img.shields.io/github/forks/pritampanda15/PandaProt?style=social" alt="GitHub Forks">
  </a>
  <a href="https://pepy.tech/project/PandaProt">
    <img src="https://static.pepy.tech/badge/PandaProt" alt="Downloads">
  </a>
</p>

### Comprehensive Interaction Mapping

PandaProt can detect and analyze a comprehensive set of molecular interactions that occur in protein/DNA structures:

**Standard Interactions:**
- **Hydrogen Bonds**: Between hydrogen donors and acceptors (N-H···O, O-H···O, etc.)
- **Ionic Interactions**: Between oppositely charged residues (Arg/Lys/His with Asp/Glu)
- **Hydrophobic Interactions**: Between non-polar side chains
- **Pi-Pi Stacking**: Between aromatic rings (parallel, T-shaped, offset-stacked)
- **Pi-Cation Interactions**: Between aromatic rings and cationic groups
- **Salt Bridges**: Close-range ionic interactions forming "bridges"

**Enhanced Interactions:**
- **Cation-Pi Interactions**: Between cations and aromatic rings
- **CH-Pi Interactions**: Between CH groups and aromatic rings
- **Disulfide Bridges**: Covalent bonds between cysteine residues
- **Sulfur-Aromatic Interactions**: Between sulfur atoms and aromatic rings
- **Water-Mediated Interactions**: Hydrogen bonds bridged by water molecules
- **Metal-Coordinated Bonds**: Interactions mediated by metal ions
- **Halogen Bonds**: Between halogen atoms and Lewis bases
- **Amide-Aromatic Interactions**: Between amide groups and aromatic rings
- **Van der Waals Interactions**: Non-specific attractive forces
- **Amide-Amide Hydrogen Bonds**: Between backbone or side chain amide groups

### Intuitive Visualization

- **Interactive 3D Visualization**: Color-coded display of interactions in the 3D structure
- **Network Analysis**: Graph representation of interaction networks
- **Detailed Reports**: Comprehensive interaction statistics and measurements

## Installation

```bash
pip install pandaprot
```

## Usage

### Command Line Interface
```bash
pandaprot -h
usage: pandaprot [-h] [--chains CHAINS [CHAINS ...]] [--3d-plot] [--export-vis] [--report] [--network] [--output OUTPUT]
                 [--distance-cutoff DISTANCE_CUTOFF] [--include-intrachain] [--all-interactions] [--standard-only]
                 [--hydrogen-bonds] [--ionic] [--hydrophobic] [--pi-stacking] [--pi-cation] [--salt-bridges] [--cation-pi]
                 [--ch-pi] [--disulfide] [--sulfur-aromatic] [--water-mediated] [--metal-coordination] [--halogen-bonds]
                 [--amide-aromatic] [--van-der-waals] [--amide-amide] [--statistics] [--residue-summary]
                 pdb_file

PandaProt: Comprehensive Protein Interaction Mapper

positional arguments:
  pdb_file              Path to PDB file

options:
  -h, --help            show this help message and exit
  --chains CHAINS [CHAINS ...]
                        Chains to analyze (e.g., A B)
  --3d-plot             Generate 3D visualization
  --export-vis          Export visualization files for PyMOL, Chimera, VMD, and Molstar
  --report              Generate detailed interaction report
  --network             Generate interaction network visualization
  --output, -o OUTPUT   Output file prefix
  --distance-cutoff DISTANCE_CUTOFF
                        Distance cutoff for interaction detection (default: 4.5Å)
  --include-intrachain  Include intra-chain interactions

Interaction Types:
  --all-interactions    Map all interaction types (default)
  --standard-only       Map only standard interactions (H-bonds, ionic, hydrophobic, pi-stacking, salt bridges)
  --hydrogen-bonds      Map hydrogen bonds
  --ionic               Map ionic interactions
  --hydrophobic         Map hydrophobic interactions
  --pi-stacking         Map pi-pi stacking interactions
  --pi-cation           Map pi-cation interactions
  --salt-bridges        Map salt bridges
  --cation-pi           Map cation-pi interactions
  --ch-pi               Map CH-pi interactions
  --disulfide           Map disulfide bridges
  --sulfur-aromatic     Map sulfur-aromatic interactions
  --water-mediated      Map water-mediated interactions
  --metal-coordination  Map metal-coordinated bonds
  --halogen-bonds       Map halogen bonds
  --amide-aromatic      Map amide-aromatic interactions
  --van-der-waals       Map van der Waals interactions
  --amide-amide         Map amide-amide hydrogen bonds

Analysis Options:
  --statistics          Generate interaction statistics
  --residue-summary     Generate residue interaction summary

```

# Full command
```bash
pandaprot 4kpy.pdb --export-vis  --3d-plot --report --network --statistics --residue-summary --chains A B C D E F --output protein-dna-ligand
```
```bash
pandaprot 1BJ1.pdb --3d-plot --export-vis  --report --network --statistics --residue-summary --chains L H V W J K  --output antigen_antibody
```

```bash
# Basic usage - map all interactions
pandaprot example.pdb

# Analyze specific chains
pandaprot example.pdb --chains A B

# Generate 3D visualization
pandaprot example.pdb --3d-plot

# Generate detailed report
pandaprot example.pdb --report

# Generate interaction network
pandaprot example.pdb --network

# Map specific interaction types
pandaprot example.pdb --hydrogen-bonds --ionic --salt-bridges

# Map only standard interactions
pandaprot example.pdb --standard-only

# Include water-mediated and metal-coordinated interactions
pandaprot example.pdb --water-mediated --metal-coordination

# Generate interaction statistics
pandaprot example.pdb --statistics

# Specify output file prefix
pandaprot example.pdb --3d-plot --report --network --output myresults

# Include intra-chain interactions
pandaprot example.pdb --include-intrachain
```

### Python API

```python
from pandaprot import PandaProt

# Initialize with PDB file
analyzer = PandaProt("example.pdb", chains=["A", "B"])

# Map all interactions
interactions = analyzer.map_interactions()

# Map with custom parameters
interactions = analyzer.map_interactions(
    distance_cutoff=4.5,
    include_intrachain=False
)

# Filter interactions
filtered = analyzer.filter_interactions(
    interaction_types=["hydrogen_bonds", "salt_bridges"],
    chains=["A"],
    distance_range=(2.5, 3.5)
)

# Generate 3D visualization
analyzer.visualize_3d("visualization.html")

# Visualize specific interaction types
analyzer.visualize_3d(
    "hbonds_visualization.html", 
    interaction_types=["hydrogen_bonds", "water_mediated"]
)

# Generate detailed report
report = analyzer.generate_report("report.csv")

# Generate report for specific interaction types
report = analyzer.generate_report(
    "ionic_report.csv",
    interaction_types=["ionic_interactions", "salt_bridges"]
)

# Create interaction network
graph, fig = analyzer.create_interaction_network("network.png")

# Get interaction statistics
stats = analyzer.get_interaction_statistics()
print(f"Total interactions: {stats['total_interactions']}")
print(f"Most interactive residue: {list(stats['residue_frequencies'].keys())[0]}")
```

## Example Applications

### Antibody-Antigen Interaction Analysis

```python
# Analyze an antibody-antigen complex
analyzer = PandaProt("antibody_complex.pdb", chains=["H", "L", "A"])

# Map all interactions
interactions = analyzer.map_interactions()

# Find epitope residues (antigen residues interacting with antibody)
epitope_residues = []
for interaction_type, interactions_list in interactions.items():
    for interaction in interactions_list:
        # Extract chain information from interaction
        chain1 = interaction.get('chain1', interaction.get('donor_chain', ''))
        chain2 = interaction.get('chain2', interaction.get('acceptor_chain', ''))
        
        # Extract residue information
        res1 = interaction.get('residue1', interaction.get('donor_residue', ''))
        res2 = interaction.get('residue2', interaction.get('acceptor_residue', ''))
        
        # Check if antigen (chain A) is interacting with antibody (chains H or L)
        if chain1 == 'A' and chain2 in ['H', 'L']:
            epitope_residues.append(f"A:{res1}")
        elif chain2 == 'A' and chain1 in ['H', 'L']:
            epitope_residues.append(f"A:{res2}")

# Get unique epitope residues
epitope_residues = list(set(epitope_residues))
print(f"Epitope residues: {', '.join(epitope_residues)}")
```

### Protein-Protein Interface Analysis

```python
# Analyze a protein-protein complex
analyzer = PandaProt("protein_complex.pdb", chains=["A", "B"])

# Map all interactions
interactions = analyzer.map_interactions()

# Generate statistics
stats = analyzer.get_interaction_statistics()

# Find hotspot residues (highly interactive residues)
hotspots = []
for residue, count in list(stats['residue_frequencies'].items())[:5]:
    hotspots.append((residue, count))

print("Interface hotspot residues:")
for residue, count in hotspots:
    print(f"  - {residue}: {count} interactions")

# Analyze interaction types at the interface
print("\nInteraction composition at the interface:")
total = stats['total_interactions']
for interaction_type, count in stats['by_type'].items():
    if count > 0:
        percentage = (count / total) * 100
        print(f"  - {interaction_type}: {count} ({percentage:.1f}%)")
```

### Drug Binding Site Analysis

```python
# Analyze a protein-ligand complex
analyzer = PandaProt("protein_ligand.pdb", chains=["A", "L"])

# Map interactions with focus on drug binding
interactions = analyzer.map_interactions()

# Create a pharmacophore model based on interactions
pharmacophore_features = {
    'hydrogen_bond_donors': [],
    'hydrogen_bond_acceptors': [],
    'ionic_positive': [],
    'ionic_negative': [],
    'hydrophobic': [],
    'aromatic': []
}

# Extract pharmacophore features from interactions
for interaction_type, interactions_list in interactions.items():
    for interaction in interactions_list:
        # Focus on protein-ligand interactions
        if interaction_type == 'hydrogen_bonds':
            if 'donor_chain' in interaction and interaction['donor_chain'] == 'A':
                pharmacophore_features['hydrogen_bond_acceptors'].append(interaction['donor_residue'])
            elif 'acceptor_chain' in interaction and interaction['acceptor_chain'] == 'A':
                pharmacophore_features['hydrogen_bond_donors'].append(interaction['acceptor_residue'])
        
        elif interaction_type == 'ionic_interactions':
            if 'positive_chain' in interaction and interaction['positive_chain'] == 'A':
                pharmacophore_features['ionic_negative'].append(interaction['positive_residue'])
            elif 'negative_chain' in interaction and interaction['negative_chain'] == 'A':
                pharmacophore_features['ionic_positive'].append(interaction['negative_residue'])
        
        elif interaction_type == 'hydrophobic_interactions':
            if 'chain1' in interaction and interaction['chain1'] == 'A':
                pharmacophore_features['hydrophobic'].append(interaction['residue1'])
            elif 'chain2' in interaction and interaction['chain2'] == 'A':
                pharmacophore_features['hydrophobic'].append(interaction['residue2'])
        
        elif interaction_type in ['pi_stacking', 'pi_cation', 'cation_pi']:
            if 'aromatic_chain' in interaction and interaction['aromatic_chain'] == 'A':
                pharmacophore_features['aromatic'].append(interaction['aromatic_residue'])

print("Pharmacophore features for drug design:")
for feature_type, residues in pharmacophore_features.items():
    if residues:
        print(f"  - {feature_type}: {', '.join(set(residues))}")
```

## Detailed Interaction Types

PandaProt detects and analyzes the following interaction types:

### Standard Interactions

#### Hydrogen Bonds
- **Definition**: Non-covalent interaction between a hydrogen atom covalently bonded to an electronegative donor atom (N, O) and another electronegative acceptor atom
- **Strength**: 2-10 kcal/mol
- **Distance Criteria**: Donor-acceptor distance < 3.5Å
- **Angle Criteria**: Donor-H-acceptor angle > 120°

#### Ionic Interactions
- **Definition**: Electrostatic attraction between oppositely charged groups
- **Strength**: 10-20 kcal/mol
- **Distance Criteria**: < 6.0Å between charged groups
- **Residues Involved**: Positive (Arg, Lys, His) and negative (Asp, Glu)

#### Hydrophobic Interactions
- **Definition**: Clustering of non-polar side chains to exclude water
- **Strength**: 1-2 kcal/mol per methylene group
- **Distance Criteria**: < 5.0Å between carbon atoms
- **Residues Involved**: Ala, Val, Leu, Ile, Met, Phe, Trp, Pro, Tyr

#### Pi-Pi Stacking
- **Definition**: Interaction between aromatic rings
- **Types**: Parallel, T-shaped, offset stacked
- **Strength**: 2-3 kcal/mol
- **Distance Criteria**: < 7.0Å between ring centers
- **Angle Criteria**: Dependent on stacking type
- **Residues Involved**: Phe, Tyr, Trp, His

#### Pi-Cation Interactions
- **Definition**: Attraction between a cation and the face of an aromatic ring
- **Strength**: 3-5 kcal/mol
- **Distance Criteria**: < 6.0Å between cation and ring center
- **Residues Involved**: Cation (Arg, Lys) and aromatic (Phe, Tyr, Trp)

#### Salt Bridges
- **Definition**: Close-range ionic interaction with partial covalent character
- **Strength**: 10-20 kcal/mol
- **Distance Criteria**: < 4.0Å between charged groups
- **Residues Involved**: Acidic (Asp, Glu) and basic (Arg, Lys, His)

### Enhanced Interactions

#### Cation-Pi Interactions
- **Definition**: Electrostatic attraction between a cation and π-electron system
- **Strength**: 2-5 kcal/mol
- **Distance Criteria**: < 6.0Å between cation and ring center
- **Residues Involved**: Cation (Arg, Lys, metal ions) and aromatic (Phe, Tyr, Trp)

#### CH-Pi Interactions
- **Definition**: Weak hydrogen bond between a CH group and an aromatic ring
- **Strength**: 0.5-1.0 kcal/mol
- **Distance Criteria**: < 4.0Å between CH and ring center
- **Angle Criteria**: CH vector pointing toward ring plane

#### Disulfide Bridges
- **Definition**: Covalent bond between sulfur atoms of cysteine residues
- **Strength**: 60-70 kcal/mol (covalent)
- **Distance Criteria**: 2.0-2.2Å between sulfur atoms
- **Residues Involved**: Cys

#### Sulfur-Aromatic Interactions
- **Definition**: Interaction between sulfur atoms and aromatic rings
- **Strength**: 1-2 kcal/mol
- **Distance Criteria**: < 5.5Å between sulfur and ring center
- **Residues Involved**: Sulfur (Met, Cys) and aromatic (Phe, Tyr, Trp)

#### Water-Mediated Interactions
- **Definition**: Hydrogen bonds bridged by water molecules
- **Strength**: Similar to regular hydrogen bonds
- **Distance Criteria**: < 3.5Å between polar atom and water
- **Types**: Water bridges, water networks

#### Metal-Coordinated Bonds
- **Definition**: Coordination between metal ions and electron-rich atoms
- **Strength**: 20-40 kcal/mol
- **Distance Criteria**: Depends on metal (typically 1.8-2.5Å)
- **Coordination**: Tetrahedral, octahedral, square planar
- **Metals**: Zn²⁺, Ca²⁺, Mg²⁺, Fe²⁺/³⁺, etc.

#### Halogen Bonds
- **Definition**: Interaction between halogen atom (Lewis acid) and Lewis base
- **Strength**: 1-5 kcal/mol
- **Distance Criteria**: < 4.0Å between halogen and acceptor
- **Angle Criteria**: C-X···Y angle close to 180°
- **Atoms Involved**: F, Cl, Br, I as donors; O, N, S as acceptors

#### Amide-Aromatic Interactions
- **Definition**: Interaction between amide groups and aromatic rings
- **Strength**: 1-3 kcal/mol
- **Distance Criteria**: < 4.5Å between amide and ring center
- **Residues Involved**: Amide (Asn, Gln, backbone) and aromatic (Phe, Tyr, Trp)

#### Van der Waals Interactions
- **Definition**: Weak non-specific attractive forces between atoms
- **Strength**: 0.1-1.0 kcal/mol per atom pair
- **Distance Criteria**: Sum of van der Waals radii + 40%
- **Involves**: All atom types

#### Amide-Amide Hydrogen Bonds
- **Definition**: Hydrogen bonds between amide groups
- **Strength**: Similar to regular hydrogen bonds
- **Distance Criteria**: < 3.5Å between amide groups
- **Residues Involved**: Asn, Gln, backbone amides

## Technical Details

### Distance Criteria

PandaProt uses carefully calibrated distance criteria for each interaction type:

| Interaction Type | Default Distance Cutoff (Å) |
|------------------|------------------------------|
| Hydrogen Bonds | 3.5 |
| Ionic Interactions | 6.0 |
| Hydrophobic Interactions | 5.0 |
| Pi-Pi Stacking | 7.0 |
| Pi-Cation | 6.0 |
| Salt Bridges | 4.0 |
| Cation-Pi | 6.0 |
| CH-Pi | 4.0 |
| Disulfide Bridges | 2.2 |
| Sulfur-Aromatic | 5.5 |
| Water-Mediated | 3.5 (per hydrogen bond) |
| Metal-Coordination | 3.0 |
| Halogen Bonds | 4.0 |
| Amide-Aromatic | 4.5 |
| Van der Waals | vdW radii sum × 1.4 |
| Amide-Amide | 3.5 |

These distance cutoffs can be customized using the `distance_cutoff` parameter.

### Requirements

- Python 3.7+
- BioPython
- NumPy
- Pandas
- Matplotlib
- py3Dmol
- NetworkX

## Citation

If you use PandaProt in your research, please cite:

```
Pritam Kumar Panda. (2025). 
A comprehensive Python application for mapping and visualizing molecular interactions at protein/nucleic acid interfaces and  tool for mapping protein-protein, protein-nucleic acid, and antigen-antibody interactions. GitHub repository https://github.com/pritampanda15/PandaProt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.