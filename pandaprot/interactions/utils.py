# pandaprot/interactions/utils.py
"""
Utility functions for interaction detection.
"""

from typing import List, Tuple, Optional
import numpy as np
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom 
from Bio.PDB.vectors import Vector


def calculate_distance(atom1: Atom, atom2: Atom) -> float:
    """Calculate distance between two atoms."""
    return atom1 - atom2


def calculate_centroid(coords: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Calculate centroid of a set of coordinates."""
    coords_array = np.array(coords)
    centroid = coords_array.mean(axis=0)
    return tuple(centroid)


def calculate_ring_normal(ring_atoms: List[Atom]) -> Tuple[float, float, float]:
    """
    Calculate the normal vector to a ring of atoms.
    Uses least-squares fitting to a plane.
    """
    # Extract coordinates
    coords = np.array([atom.coord for atom in ring_atoms])
    
    # Center the coordinates
    centroid = coords.mean(axis=0)
    centered_coords = coords - centroid
    
    # Use SVD for plane fitting
    u, s, vh = np.linalg.svd(centered_coords)
    
    # The normal is the right singular vector corresponding to the smallest singular value
    normal = vh[2]
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    return tuple(normal)


def calculate_angle_between_vectors(v1: Tuple[float, float, float], 
                                   v2: Tuple[float, float, float]) -> float:
    """Calculate angle between two vectors in radians."""
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    
    # Normalize vectors
    v1_norm = v1_array / np.linalg.norm(v1_array)
    v2_norm = v2_array / np.linalg.norm(v2_array)
    
    # Calculate dot product and angle
    dot_product = np.dot(v1_norm, v2_norm)
    
    # Clip to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    return np.arccos(dot_product)