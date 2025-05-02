# PandaProt

A comprehensive Python application for mapping and visualizing molecular interactions at protein interfaces.

## Features

### Comprehensive Interaction Mapping

PandaProt can detect and analyze a comprehensive set of molecular interactions that occur in protein structures:

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
    distance_range=(2.5, 3
