#!/usr/bin/python

import collections
import csv
import hashlib
import joblib
import logging
import os
import random
import re
import shutil
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import (
    AllChem,
    ChemicalFeatures,
    EState,
    FilterCatalog,
    MACCSkeys,
    rdFMCS,
    rdMHFPFingerprint,
    rdMolDescriptors,
    rdqueries,
)
from rdkit.Chem.AtomPairs import Pairs, Torsions, Utils
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Geometry import Point3D
from rdkit.RDPaths import RDDataDir
from scipy import cluster, sparse, spatial
from sklearn import ensemble, decomposition, manifold, svm
# import umap
# import tmap
# import networkx as nx

# import molcutter
# import conftools

## Property meaning for function `mol_atomfeatures`
ATOMPROPS = [
    prop
    for props in [
        ["SUM"],
        [
            f"Symbol_{i}"
            for i in [
                "B",
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "I",
                "H",
                "*",
                "other",
            ]
        ],
        [f"Degree_{i}" for i in range(7)],
        [f"Hybrid_SP{i}" for i in ["", "2", "3", "3D", "3D2"]],
        [f"ImpValence_{i}" for i in range(7)],
        [f"Charge_{i}" for i in [-1, 0, 1]],
        [f"Ringsize_{i}" for i in range(3, 9)],
        ["Aromatic"],
        [f"TotalHs_{i}" for i in range(5)],
        ["NumAtoms"],
        [f"Chriality_{i}" for i in ["CW", "CCW", "other"]],
        [
            "GasteigerCharge",
            "GasteigerHCharge",
            "logP",
            "MR",
            "LabuteASA",
            "TPSA",
            "EState",
        ],
    ]
    for prop in props
]

## Encoder of fingerprint method MHFP
MHFP_ENCODER = rdMHFPFingerprint.MHFPEncoder()

## PHARM-like fingerprint features
PHARMS = ["HYD", "AR", "ACC", "DON", "CAT", "ANI", "@CTR"]
PHARMATOMS = ["I", "P", "F", "S", "N", "O", "C"]
with open(os.path.join(RDDataDir, "BaseFeatures.fdef")) as f:
    s = f.read()
for key, val in [("Hydrophobe", "HYD"), ("LumpedHydrophobe", "HYD"), ("Aromatic", "AR"), ("Acceptor", "ACC"), ("Donor", "DON"), ("PosIonizable", "CAT"), ("NegIonizable", "ANI")]:
    s = s.replace(f"Family {key}", f"Family {val}")
s = s[:s.index("# the LigZn binder features were adapted from combichem.fdl")] + s[s.index("# aromatic rings of various sizes:"):] + """
DefineFeature site_center [!1&#0]
  Family @CTR
  Weights 1.0
EndFeature
"""
PHARM_FACTORY = ChemicalFeatures.BuildFeatureFactoryFromString(s)
PHARM_PSEUDOFACTORY = ChemicalFeatures.BuildFeatureFactoryFromString(
    "\n".join(
        [
            f"DefineFeature {pharm} {abbr}\n  Family {pharm}\n  Weights 1\nEndFeature"
            for pharm, abbr in zip(PHARMS, PHARMATOMS)
        ]
    )
)

## Summary statistical function of similarity matrix
STAT_FUNC_D = {
    "MEAN": lambda S: S.mean(axis=1),
    "MAX": lambda S: S.max(axis=1),
    "MAX5": lambda S: np.partition(S, -min(S.shape[1], 5), axis=1)[:, -min(S.shape[1], 5)],
}

FP_BIT_D = collections.defaultdict(dict)
FP_FUNC_D = {}
FP_MOLFUNC_D = {}


"""
FP_BIT_D : Dict[str, Dict[int, Tuple[str, int]]]
    Saving bit information (frag, diameter) for fingerprint methods.
    Keys are fingerprint methods, values are dictionary of bit information
    Currently supported methods: MORGAN, TOPO, RECAP, HRF, BRICS, RINGS, IOSEQ, IOFRAG, IOSHAPE
FP_FUNC_D : Dict[str, Callable[List[str], [2d-array[float]]]]
    Batch function of fingerprint calculation. Current supported methods: IOSEQ, IOFRAG, IOSHAPE
FP_MOLFUNC_D : Dict[str, Tuple[Callable[Mol, []], Callable[[Mol, int], [List[Tuple[int,...]]]]]
    Molecule-wise fingerprint function loaded.
    Keys are fingerprint methods, values are tuples of 2 functions. The
    first function calculate fingerprint from RDKit molecule, And the
    second get tuples of matched atom index for molecule with specific
    fingerprint bit as input.
"""


def frag_inthash(smi):
    """
    Get hash for fragment SMILES (usually canonical with sites `*`).
    Used for fast fingerprint ID generation for fragments.

    Parameters
    ----------
    smi : str
        Input SMILES.

    Returns
    -------
    inthash : int
        Integer hash of SMILES with 16 digits. First 7 digits for
        structural features of SMILES and last 5 digits as uniform hash
        to avoid collisions.
            0 : Count of total sites
            1 : Length of SMILES (divided by 8).
            2 : Count of aliphatic carbons (divided by 4).
            3 : Count of aromatic carbons (divided by 4).
            4 : Count of double/triple bonds.
            5 : Count of branched chains.
            6 : Count of rings.
            7 ~ 11 : Quasi random hash.
    """
    char_d = collections.Counter(smi.replace("Cl", "Q"))
    inthash = (
        10 ** 11 * min(char_d["*"], 9)
        + 10 ** 10 * min(len(smi) >> 3, 9)
        + 10 ** 9 * min(char_d["C"] >> 2, 9)
        + 10 ** 8 * min(char_d["c"] >> 2, 9)
        + 10 ** 7 * min(char_d["="] + char_d["#"], 9)
        + 10 ** 6 * min(char_d["("], 9)
        + 10 ** 5 * min(sum([char_d[key] for key in "123456789"]) >> 1, 9)
        + int.from_bytes(
            hashlib.blake2s(bytes(smi, "utf-8"), digest_size=3).digest(), "little"
        )
        % 100000
    )
    return inthash


def frag_hashmap(frags):
    """
    Collect fragments into count-based hashing. Used for fingerprint
    conversion.

    Parameters
    ----------
    frags : List[str]
        Input fragments SMILES.

    Returns
    -------
    count_d : Dict[int, Tuple[str, int]]
        Keys are hashed values of fragments, values are tuple of
        (fragment SMILES, counted number of fragment).
    """
    frag_d = collections.Counter(frags)
    count_d = {
        frag_inthash(frag): (frag, count) for frag, count in frag_d.items() if frag
    }
    return count_d


def mol_map4_fp(mol):
    """
    Generate revised MAP4 fingerprint for vectorization.

    Parameters
    ----------
    mol : Mol
        Input molecule.

    Returns
    -------
    bits : 1d-array[uint32]
        Fingerprint bits of MAP4
    counts : 1d-array[uint8]
        Counts of bits of MAP4
    """
    bins = [1, 2, 3, 5, 8, 13, 21, 34]
    D = np.digitize(AllChem.GetDistanceMatrix(mol), bins=bins[1:]).tolist()
    n_atom = len(D)
    atoms_env = [atom.GetSmarts(isomericSmiles=False) for atom in mol.GetAtoms()] + list(MHFP_ENCODER.CreateShinglingFromMol(mol, radius=2, rings=False, kekulize=False, isomeric=False))
    if len(atoms_env) < (3 * n_atom):
        logging.warning(f"Get atom environment error for MAP4: {Chem.MolToSmiles(mol, canonical=False)}")
        return np.array([], dtype=np.uint32), np.array([], dtype=np.uint8)
    hash_env = [int.from_bytes(hashlib.blake2s(bytes(smi, "utf-8"), digest_size=3).digest(), "little") for smi in atoms_env]
    count_d = collections.Counter()
    for i, Di in enumerate(D):
        for j, d in enumerate(Di[:i]):
            for ri, ei in enumerate([i, n_atom + 2 * i, n_atom + 2 * i + 1]):
                for rj, ej in enumerate([j, n_atom + 2 * j, n_atom + 2 * j + 1]):
                    bit = (hash_env[ei] + hash_env[ej]) | ((8 * (ri + rj) + d) << 25)
                    count_d[bit] += 1
                    if bit not in FP_BIT_D["MAP4"]:
                        FP_BIT_D["MAP4"][bit] = (f"{atoms_env[ei]}.{atoms_env[ej]}", bins[d])
    bits, counts = zip(*sorted(count_d.items()))
    bits = np.array(bits, dtype=np.uint32)
    counts = np.minimum(counts, 256).astype(np.uint8)
    return bits, counts


def mol_scaf_fp(mol):
    """
    Calculate scaffold (SCAF) fingerprint for molecule.

    Parameters
    ----------
    mol : Mol
        Input molecule.

    Returns
    -------
    fp_scaf : 1d-array[uint16] of length 30
        Scaffold fingerprint of input molecule.
    """
    ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
    carbons = set()
    sp3rings = set()
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == "C":
            carbons.add(i)
            if (i in ring_atoms) and atom.GetTotalDegree() == 4:
                sp3rings.add(i)
    D = np.digitize(AllChem.GetDistanceMatrix(mol)[:, list(ring_atoms)], bins=[1, 2, 3, 5, 8, 13, 21, 34, 55])
    fp_scaf = np.minimum(np.r_[
        np.bincount(D[list(ring_atoms)].ravel(), minlength=10),
        np.bincount(D[list(ring_atoms.difference(carbons))].ravel(), minlength=10),
        np.bincount(D[list(ring_atoms.difference(sp3rings))].ravel(), minlength=10),
    ], 65535).astype(np.uint16)
    return fp_scaf


def mol_fps(
    mol,
    methods={
        "MORGAN",
        "TOPO",
        "PAIR",
        "TORSION",
        "MAP4",
        "ESTATE",
        "LAYERED",
        "PATTERN",
        "MACCS",
        "ERG",
        "AUTOCORR",
        "MQN",
        "VSA",
        "AVALON",
        "DISTPROP",
        "PHARM",
        "SCAF",
    },
):
    """
    Calculate fingerprints using different methods from molecule.

    Parameters
    ----------
    mol : Mol
        Input molecule.
    methods: List[str]
        Methods to use for vectorization.  Available methods:
            MORGAN: Tuple[1d-array[uint32], 1d-array[uint8]] (sparse of length 2 ** 32)
                Circular (ECFP4), circular of radius <= 2
            TOPO: Tuple[1d-array[uint32], 1d-array[uint8]] (sparse of length 2 ** 32)
                Path-based, n-connected bonds
            PAIR: Tuple[1d-array[uint32], 1d-array[uint8]] (sparse of length 2 ** 23)
                Disconnected pairs of 2 typed atom [{atom type};D{# heavy neighbours}:{# pi bonds}]
            TORSION: Tuple[1d-array[int64], 1d-array[uint8]] (sparse of length 2 ** 36)
                4-atom-path of typed atom [{atom type};D{# heavy neighbours}:{# pi bonds}]
            MAP4: Tuple[1d-array[uint32], 1d-array[uint8]] (sparse of length 2 ** 32)
                Modified MAP4 fingerprint for vectorization, circular of radius <= 2
            LAYERED: 1d-array[uint8] of length 2048
                Path-based,
            PATTERN: 1d-array[uint8] of length 2048
                Substructure keys
            MACCS: 1d-array[uint8] of length 167
                Substructure keys, SMARTS of 166 pre-defined substructures
            ESTATE: 1d-array[float32] of length 158
                Substructure keys, SMARTS of 79 pre-defined atom
                patterns with unweighted / weighted counts
            ERG: 1d-array[float32] of length 315
                Pharmacophore, extended reduced graph
                Similarity search suggested
            AUTOCORR: 1d-array[float32] of length 192
                Graph-based
            MQN: 1d-array[float32] of length 42
                Molecular quantum numbers
            VSA: 1d-array[float32] of length 36
                MOE-type Van der Waals Surface Area
            AVALON: 1d-array[uint8] of length 512
                Path-based
            PAIRPROP: 1d-array[float32] of length 1830
                Quadratic interaction of atom properties with atom
                connectivity matrix.
            DISTPROP: 1d-array[float32] of length 1830
                Quadratic interaction of atom properties with atom
                distance matrix.
            PHARM: 1d-array[uint8] of length 168
                Quasi-3D pharmacophore fingerprint
            SCAF : 1d-array[uint16] of length 30
                Ring-based.

    Returns
    -------
    fp_d : Dict[str, 1d-array or Tuple[1darray[uint32, 1darray[uint8]]]
        Molecule-wise fingerprint dictionary of different methods. Keys
        are methods, and values are fingerprint vectors. If sparse bits,
        using a tuple of 2 arrays of (bits, counts).

    References
    ----------
    * Gregory Landrum
      Fingerprints in the RDKit
      https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf

    * Getting Started with the RDKit in Python
      https://rdkit.readthedocs.io/en/latest/GettingStartedInPython.html

    * Nikolaus Stiefl, Ian A. Watson, Knut Baumann, and Andrea Zaliani
      ErG: 2D Pharmacophore Descriptions for Scaffold Hopping
      https://pubs.acs.org/doi/abs/10.1021/ci050457y

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("c1ccccc1")
    >>> fp_d = mol_fps(mol, methods={"MORGAN", "TOPO"})
    >>> print(fp_d)
    {'MORGAN': (array([  98513984, 2763854213, 3218693969], dtype=uint32), array([6, 6, 6], dtype=uint8)), 'TOPO': (array([ 374073638,  375489799, 1949583554, 3517902689, 3752102730,
       3764713633], dtype=uint32), array([6, 1, 6, 6, 6, 6], dtype=uint8))}
    """
    fp_d = {}
    if "MORGAN" in methods:
        count_d = AllChem.GetMorganFingerprint(mol, 2).GetNonzeroElements()
        fp_d["MORGAN"] = (
            np.fromiter(count_d, np.uint32),
            np.fromiter(count_d.values(), np.uint8),
        )
        bits = [
            bit
            for bit in count_d
            if bit not in FP_BIT_D["MORGAN"] or not FP_BIT_D["MORGAN"][bit][0]
        ]
        if bits:
            bitinfo_d = {}
            AllChem.GetMorganFingerprint(mol, 2, bitInfo=bitinfo_d)
            for bit in bits:
                atom, radius = bitinfo_d[bit][0]
                try:
                    frag = smi_frag_from_circular(mol, atom, radius)
                except RuntimeError:
                    frag = None
                FP_BIT_D["MORGAN"][bit] = (frag, radius * 2)
    if "TOPO" in methods:
        count_d = Chem.rdmolops.UnfoldedRDKFingerprintCountBased(
            mol
        ).GetNonzeroElements()
        fp_d["TOPO"] = (
            np.fromiter(count_d, np.uint32),
            np.fromiter(count_d.values(), np.uint8),
        )
        bits = [
            bit
            for bit in count_d
            if bit not in FP_BIT_D["TOPO"] or not FP_BIT_D["TOPO"][bit][0]
        ]
        if bits:
            bitinfo_d = {}
            Chem.rdmolops.UnfoldedRDKFingerprintCountBased(mol, bitInfo=bitinfo_d)
            for bit in bits:
                path = bitinfo_d[bit][0]
                frag = Chem.MolToSmiles(
                    Chem.PathToSubmol(mol, path), allBondsExplicit=True
                )
                FP_BIT_D["TOPO"][bit] = (frag, len(path))
    if "PAIR" in methods:
        count_d = AllChem.GetAtomPairFingerprint(mol).GetNonzeroElements()
        fp_d["PAIR"] = (
            np.fromiter(count_d, np.uint32),
            np.fromiter(count_d.values(), np.uint8),
        )
    if "TORSION" in methods:
        count_d = AllChem.GetTopologicalTorsionFingerprint(mol).GetNonzeroElements()
        fp_d["TORSION"] = (
            np.fromiter(count_d, np.int64),
            np.fromiter(count_d.values(), np.uint8),
        )
    if "MAP4" in methods:
        fp_d["MAP4"] = mol_map4_fp(mol)
    if "MACCS" in methods:
        fp_d["MACCS"] = np.zeros(0, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMACCSKeysFingerprint(mol), fp_d["MACCS"]
        )
    if "LAYERED" in methods:
        fp_d["LAYERED"] = np.zeros(0, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(
            AllChem.LayeredFingerprint(mol), fp_d["LAYERED"]
        )
    if "PATTERN" in methods:
        fp_d["PATTERN"] = np.zeros(0, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(
            AllChem.PatternFingerprint(mol), fp_d["PATTERN"]
        )
    if "ESTATE" in methods:
        fp_d["ESTATE"] = np.concatenate(FingerprintMol(mol)).astype(np.float32)
    if "ERG" in methods:
        fp_d["ERG"] = AllChem.GetErGFingerprint(mol).astype(np.float32)
    if "AUTOCORR" in methods:
        fp_d["AUTOCORR"] = np.nan_to_num(AllChem.CalcAUTOCORR2D(mol)).astype(np.float32)
    if "MQN" in methods:
        fp_d["MQN"] = np.array(rdMolDescriptors.MQNs_(mol), dtype=np.float32)
    if "VSA" in methods:
        fp_d["VSA"] = np.concatenate(
            [
                rdMolDescriptors.PEOE_VSA_(mol),
                rdMolDescriptors.SlogP_VSA_(mol),
                rdMolDescriptors.SMR_VSA_(mol),
            ]
        ).astype(np.float32)
    if "AVALON" in methods:
        count_d = pyAvalonTools.GetAvalonCountFP(
            Chem.MolToSmiles(mol, canonical=False).replace("~", ""), isSmiles=True
        ).GetNonzeroElements()
        fp_d["AVALON"] = (
            np.fromiter(count_d, np.int16),
            np.fromiter(count_d.values(), np.uint8),
        )
    if ("PAIRPROP" in methods) or ("DISTPROP" in methods):
        K, V = mol_atomfeatures(mol)
        D = AllChem.GetDistanceMatrix(mol).astype(np.float32)
        V[:, 59] = (
            (np.add.outer(V[:, 59], -V[:, 59]) + np.diag(V[:, 59])) / (D + 1) ** 2
        ).sum(axis=1)
        if "PAIRPROP" in methods:
            norm = np.sqrt(K[:, [4]], dtype=np.float32)
            norm[norm == 0] = 1e-16
            G = V.T @ (V - ((D == 1) / norm) @ (V / norm))
            fp_d["PAIRPROP"] = np.r_[
                np.diag(G), spatial.distance.squareform(G, checks=False)
            ]
        if "DISTPROP" in methods:
            norm = max(D.sum() / len(D), 1)
            G = V.T @ (D / norm) @ V
            fp_d["DISTPROP"] = np.r_[
                np.diag(G), spatial.distance.squareform(G, checks=False)
            ]
    if "PHARM" in methods:
        fp_d["PHARM"] = mol_pharmfp(mol)
    if "SCAF" in methods:
        fp_d["SCAF"] = mol_scaf_fp(mol)
    for method, (fp_func, fpmatch_func) in FP_MOLFUNC_D.items():
        if method in methods:
            fp_d[method] = fp_func(mol)
    return fp_d


def mol_3dfps(
    mol, methods={"3DUSRCAT", "3DAUTOCORR", "3DRDF", "3DMORSE", "3DWHIM", "3DGETAWAY", "3DDESC", "3DPHARM"}
):
    """
    Get pharmacophore fingerprint from molecule.

    Parameters
    ----------
    mol : Mol
        Input molecule with conformations.
    methods : List[str]
        Methods to use for vectorization.  Available methods:
            3DUSRCAT: 2d-array[float32] of shape (n_conf, 60)
                Distance statistics of pharmacophore atoms
            3DAUTOCORR: 2d-array[float32] of shape (n_conf, 80)
            3DRDF: 2d-array[float32] of shape (n_conf, 210)
            3DMORSE: 2d-array[float32] of shape (n_conf, 224)
            3DWHIM: 2d-array[float32] of shape (n_conf, 114)
            3DGETAWAY: 2d-array[float32] of shape (n_conf, 273)
            3DDESC: 2d-array[float32] of shape (n_conf, 8)
            3DPHARM: 2d-array[uint8] of shape (n_conf, 168)

    Returns
    -------
    fp_d : Dict[str, 2d-array[float32]]
        3D fingerprint of current molecule.

    References
    ----------
    * Adrian M Schreyer & Tom Blundell
      USRCAT: real-time ultrafast shape recognition with pharmacophoric constraints
      https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-4-27
    * RDKit
      rdkit.Chem.Descriptors3D module
      https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors3D.html

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)cc1")
    >>> conf_ids = AllChem.EmbedMultipleConfs(mol, 2)
    >>> fp_d = mol_3dfps(mol)
    >>> {key: val.shape for key, val in fp_d.items()}
    {'3DUSRCAT': (2, 60), '3DAUTOCORR': (2, 80), '3DRDF': (2, 210), '3DMORSE': (2, 224), '3DWHIM': (2, 114), '3DGETAWAY': (2, 273), '3DDESC': (2, 8)}
    """
    fp_d = {}
    idx_conf = [conf.GetId() for conf in mol.GetConformers()]
    if not idx_conf:
        return fp_d
    if "3DUSRCAT" in methods:
        if mol.GetNumAtoms() >= 3:
            fp_d["3DUSRCAT"] = np.array(
                [rdMolDescriptors.GetUSRCAT(mol, confId=i) for i in idx_conf],
                dtype=np.float32,
            )
    if "3DAUTOCORR" in methods:
        fp_d["3DAUTOCORR"] = np.array(
            [AllChem.CalcAUTOCORR3D(mol, confId=i) for i in idx_conf], dtype=np.float32
        )
    if "3DRDF" in methods:
        fp_d["3DRDF"] = np.array(
            [AllChem.CalcRDF(mol, confId=i) for i in idx_conf], dtype=np.float32
        )
    if "3DMORSE" in methods:
        fp_d["3DMORSE"] = np.array(
            [AllChem.CalcMORSE(mol, confId=i) for i in idx_conf], dtype=np.float32
        )
    if "3DWHIM" in methods:
        fp_d["3DWHIM"] = np.array(
            [AllChem.CalcWHIM(mol, confId=i) for i in idx_conf], dtype=np.float32
        )
    if "3DGETAWAY" in methods:
        fp_d["3DGETAWAY"] = np.nan_to_num(
            np.array(
                [AllChem.CalcGETAWAY(mol, confId=i) for i in idx_conf], dtype=np.float32
            ),
            posinf=0,
            neginf=0,
        )
    if "3DDESC" in methods:
        fp_d["3DDESC"] = np.array(
            [
                [
                    rdMolDescriptors.CalcAsphericity(mol, confId=i),
                    rdMolDescriptors.CalcEccentricity(mol, confId=i),
                    rdMolDescriptors.CalcInertialShapeFactor(mol, confId=i),
                    rdMolDescriptors.CalcNPR1(mol, confId=i),
                    rdMolDescriptors.CalcNPR2(mol, confId=i),
                    rdMolDescriptors.CalcPBF(mol, confId=i),
                    rdMolDescriptors.CalcRadiusOfGyration(mol, confId=i),
                    rdMolDescriptors.CalcSpherocityIndex(mol, confId=i),
                ]
                for i in idx_conf
            ],
            dtype=np.float32,
        )
    if "3DPHARM" in methods:
        if len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(1))) > 0:
            mol_noHs = Chem.RemoveHs(mol, sanitize=True)
        else:
            mol_noHs = mol
        try:
            pharm_d = mol_pharm(mol_noHs)
        except Exception:
            logging.exception("Extract molecule pharmacophore error")
        fp_d["3DPHARM"] = np.vstack([mol_pharmfp(mol_noHs, AllChem.Get3DDistanceMatrix(mol_noHs, i), pharm_d) for i in idx_conf])
    return fp_d


def mol_pharm_fp(mol_p):
    """
    Calculate fingerprint from pharmacophore pseudo molecule with 3D
    conformers.

    Parameters
    ----------
    mol_p : Mol
        Pharmacophore pseudo molecule with 3D conformations.

    Returns
    -------
    fp : 2d-array[uint8] of shape (n_conf, 168)
        3D pharmacophore fingerprint.

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("COc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc4c(c3)OCCO4)CC2)cc1")
    >>> conf_ids = AllChem.EmbedMultipleConfs(mol, 2)
    >>> mol_p = get_pharm_mol(mol)
    >>> fp = mol_pharm_fp(mol_p)
    >>> fp.shape
    (2, 168)
    """
    idx_conf = [conf.GetId() for conf in mol_p.GetConformers()]
    if len(idx_conf) == 0:
        return None
    pharm_d = mol_pharm(mol_p, factory=PHARM_PSEUDOFACTORY)
    fp = np.vstack([mol_pharmfp(mol_p, AllChem.Get3DDistanceMatrix(mol_p, i), pharm_d) for i in idx_conf])
    return fp


def stack_sparse_fp(fps):
    """
    Stack molecule-wise sparse fingerprint records to sparse matrix.

    Parameters
    ----------
    fps : List[Tuple[1d-array[int], 1d-array[int]]]
        Sparse fingerprints, each element is (bits, counts).

    Returns
    -------
    X_fp : sparse.csr_matrix of shape (len(fps), ..)
        Stacked sparse fingerprints matrix.

    See also
    --------
    split_sparse_fp : Split sparse fingerprint matrix to molecule-wise
        records.
    """
    fps_bits = []
    fps_counts = []
    fps_idx = [0]
    if fps is None:
        return None
    for fp in fps:
        if fp is None:
            fps_idx.append(0)
        else:
            fps_idx.append(len(fp[0]))
            fps_bits.append(fp[0])
            fps_counts.append(fp[1])
    fps_idx = np.cumsum(fps_idx)
    if fps_idx[-1] == 0:
        return None
    fps_bits = np.concatenate(fps_bits)
    fps_counts = np.concatenate(fps_counts)
    X_fp = sparse.csr_matrix(
        (fps_counts, fps_bits, fps_idx),
        shape=(len(fps), 1 << int(fps_bits.max()).bit_length()),
    )
    return X_fp


def split_sparse_fp(X):
    """
    Split sparse fingerprint matrix to molecule-wise records.

    Parameters
    ----------
    X : sparse.csr_matrix
        Sparse fingerprints matrix.

    Returns
    -------
    fps : List[Tuple[1d-array[int], 1d-array[int]]] of length X.shape[0]
        Sparse fingerprints, each element is (bits, counts).

    See also
    --------
    stack_sparse_fp : Stack molecule-wise sparse fingerprint records to
        sparse matrix.
    """
    fps = list(
        zip(np.split(X.indices, X.indptr[1:-1]), np.split(X.data, X.indptr[1:-1]))
    )
    return fps


def first_notnone(iterable):
    """
    Get first non-None element from a iterable.

    Parameters
    ----------
    iterable : List[Any]
        An iterable object, usually a list with unknown elements mixed
        with None.

    Returns
    -------
    Any
        First element which is not None. If all None, return None.

    Examples
    --------
    >>> first_notnone(["c", None, "b", 3])
    'c'
    >>> first_notnone([None, np.array([1,2,3]), np.array([2,3,5])])
    array([1, 2, 3])
    >>> first_notnone([None, None, None])
    """
    return next(filter(lambda x: x is not None, iterable), None)


def fpbits_to_method(fpbits, keep_index=False):
    """
    Gather fingerprint bits by methods.

    Parameters
    ----------
    fpbits : List[str]
        Fingerprint bits with format "{METHOD}_{bit}".
    keep_index : bool
        Whether keep original index of fingerprint bits in output.

    Returns
    -------
    method_d : Dict[str, List[int]] or Dict[str, List[Tuple[int, int]]]
        Collected bits gathered by fingerprint methods. Keys are
        fingerprint methods. If keep_index is False, values are bits,
        otherwise values are tuples of (bit, index of `fpbits`).

    See also
    --------
    method_to_fpbits : Expand fingerprint method-bit dictionary to bits.

    Examples
    --------
    >>> fpbits_to_method(["MORGAN_123456", "MORGAN_789", "TOPO_251"])
    {'MORGAN': [123456, 789], 'TOPO': [251]}
    >>> fpbits_to_method(["MORGAN_123456", "MORGAN_789", "TOPO_251"], keep_index=True)
    {'MORGAN': [(123456, 0), (789, 1)], 'TOPO': [(251, 2)]}
    >>> fpbits_to_method(["MORGAN", "TOPO", "3DPHARM_86", "3DPHARM_113"])
    {'MORGAN': [], 'TOPO': [], '3DPHARM': [86, 113]}
    """
    method_d = {}
    for i, fpbit in enumerate(fpbits):
        fpbit = fpbit.strip(" []'\"")
        if fpbit:
            method, _, bit = fpbit.partition("_")
            if method not in method_d:
                method_d[method] = []
            if bit:
                if keep_index:
                    method_d[method].append((int(bit), i))
                else:
                    method_d[method].append(int(bit))
    return method_d


def method_to_fpbits(method_d):
    """
    Expand fingerprint method-bit dictionary to bits.

    Parameters
    ----------
    method_d : Dict[str, List[int]]
        Collected bits gathered by fingerprint methods.

    Returns
    -------
    fpbits : List[str]
        Fingerprint bits with format "{METHOD}_{bit}".

    See also
    --------
    fpbits_to_method : Gather fingerprint bits by methods.

    Examples
    --------
    >>> method_to_fpbits({'MORGAN': [123456, 789], 'TOPO': [251]})
    ['MORGAN_123456', 'MORGAN_789', 'TOPO_251']
    """
    fpbits = [f"{method}_{bit}" for method, bits in method_d.items() for bit in bits]
    return fpbits


def collect_feature_names(model_d, methods=None):
    """
    Collect fingerprint bits used to train from QSAR models for prediction.

    Parameters
    ----------
    model_d : Dict[str, model]
        Saved models for prediction. Keys are names of model, values
        with sklearn-API and attribute `feature_names` or
        `data_manager.columns`
    methods : List[str]
        Available fingerprint methods to use. If not None, put a
        restriction filter on input models.

    Returns
    -------
    model_out_d : Dict[str, model]
        Models with `feature_names`.
    method_d : Dict[str, List[int]]
        Collected bits gathered by fingerprint methods. Keys are
        fingerprint methods, values are bits.
    """
    model_out_d = {}
    method_out_d = {}
    for key, model in model_d.items():
        if hasattr(model, "feature_names"):
            method_d = fpbits_to_method(model.feature_names)
        elif hasattr(model, "data_manager"):
            model.feature_names = model.data_manager.columns
            method_d = fpbits_to_method(model.feature_names)
        else:
            logging.warning(f"No attribute 'feature_names' in model {key}, do not use for prediction")
            continue
        for method, bits in method_d.items():
            if methods and method not in methods:
                logging.warning(f"Detect {method} in model {key}, available methods: {methods}")
                break
            if method in method_out_d:
                method_out_d[method] = sorted(set(method_out_d[method]).union(bits))
            else:
                method_out_d[method] = bits
        else:
            model_out_d[key] = model
    if model_d:
        logging.info(f"Collect {sum(len(val) for val in method_out_d.values())} bits / {len(method_out_d)} fingerprint methods from {len(model_out_d)} / {len(model_d)} models: {list(method_out_d)}")
    return model_out_d, method_out_d


def save_fps_dict(filepath, X_d):
    """
    Save fingerprint dictionary into *.npz file.

    Parameters
    ----------
    X_d : Dict[str, List[Any]]
        Fingerprint dictionary for saved smiles. Keys are fingerprint
        methods, and values are fingerprint matrices.

    See also
    --------
    load_fps_dict : load fingerprint dictionary from compressed *.npz.
    """
    fp_d = {}
    for key, X in X_d.items():
        key = key.lower()
        if X is None:
            continue
        elif type(X) == np.ndarray:
            fp_d[f"fp_{key}"] = X
        elif key == "morgan":
            fp_d.update({
                "indices": X.indices,
                "indptr": X.indptr,
                "format": X.format.encode('ascii'),
                "shape": X.shape,
                "data": X.data
            })
        elif sparse.issparse(X):
            fp_d.update({
                f"fp_{key}_indices": X.indices,
                f"fp_{key}_indptr": X.indptr,
                f"fp_{key}_format": X.format.encode('ascii'),
                f"fp_{key}_shape": X.shape,
                f"fp_{key}_data": X.data
            })
    np.savez_compressed(filepath, **fp_d)


def load_fps_dict(filepath):
    """
    Load fingerprint dictionary from fingerprint *.npz file.

    Parameters
    ----------
    filepath : str
        Fingerprint *.npz file.

    Returns
    -------
    X_d : Dict[str, List[Any]]
        Fingerprint dictionary for saved smiles. Keys are fingerprint
        methods, and values are fingerprint matrices.

    See also
    --------
    save_fps_dict : save fingerprint dictionary into *.npz file.
    """
    fp_d = dict(np.load(filepath))
    X_d = {}
    for key, val in fp_d.items():
        subkeys = key.split("_")
        if len(subkeys) == 2:
            X_d[subkeys[-1].upper()] = val
        elif subkeys[-1] == "format":
            sp_format = getattr(sparse, val.item().decode('ascii') + '_matrix')
            if len(subkeys) == 1:
                X_d["MORGAN"] = sp_format((fp_d['data'], fp_d['indices'], fp_d['indptr']), shape=fp_d['shape'])
            else:
                X_d[subkeys[1].upper()] = sp_format((fp_d[key[:-6] + 'data'], fp_d[key[:-6] + 'indices'], fp_d[key[:-6] + 'indptr']), shape=fp_d[key[:-6] + 'shape'])
    return X_d


def model_init(methods=[], X_df=None, n_jobs=1, random_state=0, select_names=None):
    """
    Initiate QSAR models from saved files or methods.

    Parameters
    ----------
    methods : List[str]
        List of model paths or names.
    X_df : DataFrame
        Input fingerprint matrix.
    n_jobs : int
        The number of threads used for model.
    random_state : int
        Random seed of model.
    select_names : Set[str]
        QSAR names defined by user in predictor string.

    Returns
    -------
    model_d : Dict[str, model]
        Loaded models.
    """
    model_d = {}
    logging.info(f"Begin to initiate models: {methods}")
    for method in methods:
        names = os.path.split(method.strip())[-1].split("_")
        if not names[-1]:
            pass
        elif names[-1] == "model.bz2":
            if names[-2] == "best":
                category, name = "QSAR", names[-3]
            elif names[-2].startswith("fpcoord"):
                category, name = "FPCOORD", names[-2][7:]
            else:
                category, name = "QSAR", names[-2]
            full_name = f"{category}_{name.upper()}"
            if select_names is not None and full_name not in select_names:
                continue
            try:
                model_d[full_name] = joblib.load(method)
            except Exception:
                logging.exception(f"Load model {method} failure")
        elif names[-1] == "PCA":
            model_d["FPCOORD_PCA"] = decomposition.TruncatedSVD(
                n_components=3, random_state=random_state
            )
        elif names[-1] == "ISOMAP":
            model_d["FPCOORD_ISOMAP"] = manifold.Isomap(
                n_neighbors=10, n_components=3, n_jobs=n_jobs
            )
        elif names[-1] == "UMAP":
            model_d["FPCOORD_UMAP"] = umap.UMAP(
                n_components=3, random_state=random_state
            )
        elif names[-1] == "OCSVM":
            model_d["FPCOORD_OCSVM"] = svm.OneClassSVM(gamma="auto")
        elif names[-1] == "ISOFOREST":
            model_d["FPCOORD_ISOFOREST"] = ensemble.IsolationForest(
                1000, n_jobs=n_jobs, random_state=random_state
            )
    if X_df is not None:
        model_fit(model_d, X_df)
    logging.info(f"Initiate models: {list(model_d)}")
    return model_d


def model_fit(model_d, X_df):
    """
    Fitting model

    Parameters
    ----------
    model_d : Dict[str, model]
        Models to fit. keys are names of model, values are models.
    X_df : DataFrame
        Dense fingerprint matrix with columns "{METHOD}_{bit}".
    """
    X_norm = None
    for key, model in model_d.items():
        category, _, name = key.partition("_")
        begin_time = time.time()
        if hasattr(model, "feature_names"):
            continue
        elif category == "FPCOORD":
            if name in ["PCA", "ISOMAP", "UMAP", "OCSVM"]:
                if X_norm is None:
                    logging.info("Starting normalization")
                    X_norm = X_df.values.astype(np.float32)
                    x_mean = np.mean(X_norm, axis=0)
                    x_std = np.std(X_norm, axis=0) + 1e-8
                    X_norm -= x_mean
                    X_norm /= x_std
                    logging.info(
                        f"Normalization finished, time: {time.time() - begin_time:.1f}s. Start model prediction for {X_norm.shape} matrix using methods: "
                        + ", ".join(list(model_d))
                    )
                    begin_time = time.time()
                model.fit(X_norm)
                model.x_mean_ = x_mean
                model.x_std_ = x_std
            elif name == "ISOFOREST":
                model.fit(X_df.values)
            model.feature_names = X_df.columns.tolist()
            logging.info(
                f"Finish model fitting of {key}, time: {time.time() - begin_time:.1f} s"
            )
        elif category == "FACTOR":
            model.fit(X_df, model.Y_)
            model.feature_names = X_df.columns.tolist()
            logging.info(
                f"Finish model fitting of {key}, time: {time.time() - begin_time:.1f} s"
            )
    return model_d


def model_predict(model_d, X_df, batch_size=100000):
    Y_d = {}
    X_norm = None
    for key, model in model_d.items():
        category, _, name = key.partition("_")
        begin_time = time.time()
        if category == "FPCOORD":
            if name in ["PCA", "ISOMAP", "UMAP", "OCSVM"]:
                if X_norm is None:
                    logging.info("Starting normalization")
                    X_norm = X_df[model.feature_names].values.astype(np.float32)
                    X_norm -= model.x_mean_
                    X_norm /= model.x_std_
                    logging.info(
                        f"Normalization finished, time: {time.time() - begin_time:.1f} s. Start model prediction for {X_norm.shape} matrix using methods: "
                        + ", ".join(list(model_d))
                    )
                    begin_time = time.time()
                if name == "OCSVM":
                    Y = model.score_samples(X_norm)
                else:
                    Y = model.transform(X_norm)
            elif name == "ISOFOREST":
                Y = model.score_samples(X_df.loc[:, model.feature_names].values)
        else:
            ## sklearn-like model
            X = X_df.loc[:, model.feature_names].values
            try:
                if len(X) <= batch_size:
                    Y = model.predict_proba(X)
                else:
                    Y = np.hstack([model.predict_proba(X[i:(i + batch_size)]) for i in range(0, len(X), batch_size)])
                if Y.shape[1] == 2:
                    Y = Y[:, 1]
            except (AttributeError, NotImplementedError):
                if len(X) <= batch_size:
                    Y = model.predict(X)
                else:
                    Y = np.hstack([model.predict(X[i:(i + batch_size)]) for i in range(0, len(X), batch_size)])
        Y_d[key] = Y
        logging.info(
            f"Finish model prediction using {key}. Time: {time.time() - begin_time:.1f} s"
        )
    return Y_d


class FingerprintQSAR(object):
    """
    Array representation and property prediction of Morgan fingerprint
    from SMILES using sklearn-API QSAR models.

    Parameters
    ----------
    model_d : Dict[str, model]
        Saved models for prediction. Keys are column names for output,
        values with sklearn-API and method `predict`.
    methods : List[str]
        Fingerprints to use for molecular vectorization. Two kinds of
        elements can be used:
            1. Names of fingerprint methods.
            2. Feature bits of dense matrix for prediction, with format
            "{METHOD}_{bit}".
    smiles : List[str] or Dict[str, List[str]], optional
        Pre-loaded canonical smiles. If a dict, keys are canonical
        smiles, values are enumerated smiles.
    X_d : Dict[str, List[Any]], optional
        Fingerprint dictionary for saved smiles. Keys are
        fingerprint methods, and values are fingerprint matrices for
        `smiles`.
    n_conf : int
        Number of conformers generated for 3D fingerprint and similarity.
    conf_opt : bool
        Whether to run energy minimization to the conformers.
    n_jobs : int
        The number of threads used in conformer generation.
    filepath : str, optional (default=None)
        Working directory for writing temperary files.

    Attributes
    ----------
    methods_ : Dict[str, List[int]]
        Methods to use for vectorization. Keys are methods. Values are
        fingerprint bits used for dense matrix for QSAR prediction.
    model_d_ : Dict[str, sklearn.base.RegressorMixin]
        Saved models for prediction. Keys are column names for output,
        values with sklearn-API and method `predict`.
    mols_ref_ : List[Mol]
        Reference molecules for similarity comparison.
    X_ref_d_ : Dict[str, Dict[str, sparse.csr_matrix[uint8] or ndarray]]
        Reference datasets. Keys are names of reference, values are
        fingerprint matrices as reference for similarity comparison.

    See also
    ----------
    mol_fps
        Calculate fingerprints using different methods from molecule.

    Examples
    --------
    >>> smiles = ["C1CNCC1", "c1ccccc1", "N1CCCC1", "C1CC", "CCCCC"]
    >>> model_fp = FingerprintQSAR(n_conf=1)
    >>> X_d, idx_smi = model_fp.transform(smiles) # Calculate fingerprint matrices
    >>> X_df = model_fp.get_dense(X_d, nonzero=0.01, colinear=0.9) # Get dense fingerprint matrix.
    >>> print(X_df.shape)
    (5, 830)
    >>> bitinfo_df = model_fp.fps_summary(X_df) # Get summary of fingerprint bits.
    >>> print(bitinfo_df)
                              COUNT  NONZERO     MAX_ORDER  EVEN_ORDER  \
    FPBIT_ID
    MORGAN_98513984    6.000000e+00      0.2  6.000000e+00        1.00
    MORGAN_416356657   4.000000e+00      0.4  2.000000e+00        1.00
    MORGAN_725338437   4.000000e+00      0.4  2.000000e+00        1.00
    MORGAN_1173125914  2.000000e+00      0.2  2.000000e+00        1.00
    MORGAN_1289643292  2.000000e+00      0.4  1.000000e+00        0.00
    ...                         ...      ...           ...         ...
    3DGETAWAY_266      5.850000e-01      0.8  1.770000e-01        0.00
    3DGETAWAY_267      4.330000e-01      0.8  1.260000e-01        0.00
    3DGETAWAY_268      1.330000e-01      0.4  8.100000e-02        0.00
    3DGETAWAY_269      3.402823e+38      0.4  3.402823e+38        0.50
    3DGETAWAY_272      3.402823e+38      0.8  3.402823e+38        0.25

                        SMILES_FRAG  DIAMETER
    FPBIT_ID
    MORGAN_98513984     c(:c:*):c:*       2.0
    MORGAN_416356657    C(-C-*)-N-*       2.0
    MORGAN_725338437   C1-C-C-N-C-1       4.0
    MORGAN_1173125914     C(-C)-C-*       2.0
    MORGAN_1289643292   N(-C-*)-C-*       2.0
    ...                         ...       ...
    3DGETAWAY_266              None       NaN
    3DGETAWAY_267              None       NaN
    3DGETAWAY_268              None       NaN
    3DGETAWAY_269              None       NaN
    3DGETAWAY_272              None       NaN
    >>> smiles_ref = ["c1cc(C)ccc1O", "Cc1c(N)cccc1CC"]
    >>> model_fp.set_reference(smiles_ref) # Set reference SMILES for similarity calculation.
    >>> Y_df = model_fp.predict(smiles) # Calculate similarity prediction results.
    >>> print(Y_df.shape)
    (5, 40)
    """

    def __init__(
        self,
        model_d={},
        methods=[
            "MORGAN",
            "TOPO",
            "PAIR",
            "TORSION",
            "MAP4",
            "RECAP",
            "HRF",
            "BRICS",
            "RINGS",
            "ESTATE",
            "LAYERED",
            "PATTERN",
            "MACCS",
            "ERG",
            "AUTOCORR",
            "MQN",
            "VSA",
            "DISTPROP",
            "PHARM",
            "3DPHARM",
            "3DUSRCAT",
            "3DAUTOCORR",
            "3DRDF",
            "3DMORSE",
            "3DWHIM",
            "3DGETAWAY",
        ],
        smiles=[],
        X_d={},
        n_conf=1,
        conf_opt=False,
        n_jobs=1,
        filepath=None,
    ):
        self.model_d_, self.methods_ = collect_feature_names(model_d)
        self.set_methods(methods, update=True)
        self.n_conf_ = n_conf
        self.conf_opt_ = conf_opt
        self.n_jobs_ = n_jobs
        self.filepath_ = filepath
        self.X_ref_d_ = {}
        self.statfunc_ref_d_ = {}
        if self.filepath_:
            self.init_filepath("")
        self.reset(smiles, X_d)
        logging.info(
            f"Initiate fingerprint QSAR with models [{', '.join(self.model_d_)}] and fingerprint methods [{', '.join(self.methods_)}] in filepath {self.filepath_}"
        )

    def reset(self, smiles=[], X_d={}):
        """
        Load pre-calculated fingerprint data of SMILES.

        Parameters
        ----------
        smiles : List[str] or Dict[str, List[str]], optional
            Pre-loaded canonical smiles. If a dict, keys are canonical
            smiles, values are enumerated smiles.
        X_d : Dict[str, List[Any]], optional
            Fingerprint dictionary for saved smiles. Keys are
            fingerprintmethods, and values are list of fingerprints for
            `smiles`.
        """
        self.smi_d_ = {smi: i for i, smi in enumerate(smiles)}
        if type(smiles) == dict:
            self.smi_enum_d_ = self.smi_d_.copy()
            self.smi_enum_d_.update(
                {
                    smi: i
                    for i, smi_enums in enumerate(smiles.values())
                    for smi in smi_enums
                    if smi_enums
                }
            )
        else:
            self.smi_enum_d_ = None
        self.fps_d_ = collections.defaultdict(list)
        for method, X in X_d.items():
            fps = self.split_fps(X)
            self.fps_d_[method] = fps
            if len(fps) != len(smiles):
                logging.warning(
                    f"Number of SMILES and fingerprint {method} mismatch: {len(smiles)} / {len(fps)}"
                )
        if self.filepath_:
            self.init_filepath("input")

    def set_methods(self, methods, update=False):
        """

        """
        if type(methods) == dict:
            method_d = methods
        else:
            method_d = fpbits_to_method(methods)
        if "RECAP" not in FP_MOLFUNC_D:
            FragFingerprint(
                lambda mol: molcutter.cut_mol_by_smarts(
                    mol, "recap", keep_map_num=False
                ),
                assign_name="RECAP",
            )
            FragFingerprint(
                lambda mol: molcutter.cut_mol_by_smarts(
                    mol, "hussain_rea", keep_map_num=False
                ),
                assign_name="HRF",
            )
            FragFingerprint(
                lambda mol: molcutter.cut_mol_by_brics(mol, keep_map_num=False),
                assign_name="BRICS",
            )
            FragFingerprint(
                lambda mol: molcutter.cut_mol_by_rings(mol, keep_map_num=False),
                assign_name="RINGS",
            )
            logging.info(f"Pre-loaded fingerprint methods: {list(FP_MOLFUNC_D)}")
        if update:
            for method, bits in method_d.items():
                if method in self.methods_:
                    self.methods_[method] = sorted(
                        set(self.methods_[method]).union(bits)
                    )
                else:
                    self.methods_[method] = bits
        else:
            self.methods_ = method_d

    def set_reference(
        self,
        ref_input=None,
        methods=None,
        name="SIM",
        statfunc_d=STAT_FUNC_D,
    ):
        """
        Set reference SMILES for similarity calculation.

        Parameters
        ----------
        ref_input : List[str] or List[Mol] or Dict[str, Any]
            Input SMILES or molecules.
        methods : List[str]
            Fingerprint methods to use for reference SMILES.
        ref_sim : Dict[str, ]
            Selected reference when calculating similarity.
        name : str
            Name of reference for subpath and output column.
        statfunc_d : Dict[str, Callable[[2d-array[float]], 1d-array[float]]]
            Statistics of similarity with reference. Keys are statistic
            names, values are functions applied on each input SMILES
            for similarity vector with references.
        """
        self.statfunc_ref_d_[name] = statfunc_d
        self.X_ref_d_[name] = {}
        if type(ref_input) == dict:
            for method, val in ref_input.items():
                self.X_ref_d_[name][method] = val
            self.set_methods(list(ref_input), update=True)
        else:
            val = first_notnone(ref_input)
            if self.filepath_:
                self.init_filepath(name)
            if val is None:
                fps_d = {}
                logging.error("No reference input detected.")
            elif type(val) == str:
                self.X_ref_d_[name], idx_smi = self.transform(
                    ref_input, methods=methods, subpath=name
                )
            elif type(val) == Chem.Mol:
                fps_d = collections.defaultdict(list)
                for mol_p in ref_input:
                    path_ref = os.path.join(
                        self.filepath_, name, name + "_" + mol_p.GetProp("_Name")
                    )
                    if {"ALIGNIT", "ALIGNITR"}.intersection(methods):
                        file_ref = f"{path_ref}_alignit.phar"
                        save_pharm_alignit(file_ref, mol_p)
                        fps_d["ALIGNIT"].append(file_ref)
                    if {"SHAEP", "SHAPE", "ESP"}.intersection(methods):
                        file_ref = f"{path_ref}_shaep.mol2"
                        save_pharm_shaep(file_ref, mol_p)
                        fps_d["SHAEP"].append(file_ref)
                    if {"PHARM", "3DPHARM"}.intersection(methods):
                        fp_pharm = mol_pharm_fp(mol_p)[0]
                        fps_d["PHARM"].append(fp_pharm)
                for method, fps in fps_d.items():
                    self.X_ref_d_[name][method] = self.stack_fps(
                        fps, n_conf=mol_p.GetNumConformers()
                    )
            self.set_methods(methods, update=True)
        logging.info(
            f"Set {len(ref_input)} reference {name} using methods: [{','.join(list(self.X_ref_d_[name]))}]"
        )

    def init_filepath(self, subpath):
        output_path = os.path.join(self.filepath_, subpath)
        if os.path.exists(output_path):
            for i in os.listdir(output_path):
                if i.startswith("."):
                    continue
                path = os.path.join(output_path, i)
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
        else:
            os.mkdir(output_path)
        return output_path

    def stack_fps(self, fps, n_conf=None):
        if n_conf is None:
            n_conf = self.n_conf_
        fp = first_notnone(fps)
        if fp is None:
            return None
        elif type(fp) == tuple:
            X_fp = stack_sparse_fp(fps)
        elif type(fp) == list:
            conf_len = len(fps) * n_conf
            fps_out = [None for i in range(conf_len)]
            for i, fp in zip(range(0, conf_len, n_conf), fps):
                if fp is not None:
                    for j, fp_conf in enumerate(fp):
                        fps_out[i + j] = fp_conf
            X_fp = stack_sparse_fp(fps_out)
        elif type(fp) in {str, Chem.Mol}:
            X_fp = fps
        elif type(fp) == np.ndarray:
            if len(fp.shape) == 1:
                X_fp = np.zeros((len(fps), fp.shape[0]), dtype=fp.dtype)
                for i, fp in enumerate(fps):
                    if fp is not None:
                        X_fp[i] = fp
            elif len(fp.shape) == 2:
                X_fp = np.zeros((len(fps), n_conf, fp.shape[1]), dtype=fp.dtype)
                for i, fp in enumerate(fps):
                    if fp is not None:
                        X_fp[i, : len(fp)] = fp
        return X_fp

    def split_fps(self, X):
        if type(X) == sparse.csr_matrix:
            fps = split_sparse_fp(X)
        elif type(X) == np.ndarray:
            fps = [fp for fp in X]
        elif type(X) == list:
            fps = X.copy()
        return fps

    def get_fps(self, smiles, methods):
        """
        Load pre-calculated fingerprints of SMILES from records.

        Parameters
        ----------
        smiles : List[str]
            Input SMILES
        methods : List[str]
            Fingerprint methods to find from records.

        Returns
        -------
        idx_smi : 1d-array[uint32]
            Index of SMILES in record, if not existed, use -1.
        X_saved : 2d-array[bool] of shape (len(smiles), len(methods))
            Indicator of existence of fingerprints records.
        fps_d : Dict[str, List of length len(smiles)]
            Fingerprint records for given methods. Keys are methods,
            values are fingerprints of SMILES. if not exist for current
            SMILES, use None in list.
        """
        X_saved = np.zeros((len(smiles), len(methods)), dtype=bool)
        if self.smi_enum_d_ is not None:
            smi_d = self.smi_enum_d_
        else:
            smi_d = self.smi_d_
        idx_smi = np.full(len(smiles), -1, dtype=int)
        for i, smi in enumerate(smiles):
            i_smi = smi_d.get(smi)
            if i_smi is not None:
                idx_smi[i] = i_smi
        fps_d = {}
        idx_exist = np.nonzero(idx_smi >= 0)[0]
        for j, method in enumerate(methods):
            fps_out = [None for smi in smiles]
            if method in self.fps_d_:
                fps = self.fps_d_[method]
                gap_size = len(self.smi_d_) - len(fps)
                if gap_size > 0:
                    fps.extend([None for i in range(gap_size)])
                for i, i_smi in zip(idx_exist, idx_smi[idx_exist]):
                    fp = fps[i_smi]
                    if fp is not None:
                        fps_out[i] = fp
                        X_saved[i, j] = True
            else:
                self.fps_d_[method].extend([None for i in self.smi_d_])
            fps_d[method] = fps_out
        return idx_smi, X_saved, fps_d

    def mol_fps(self, mol, methods=None, name=None):
        if methods is None:
            methods = self.methods_.copy()
        methods_3d = {method for method in methods if method.startswith("3D")}
        methods_program = {"ALIGNIT", "SHAPEIT", "SHAEP"}.intersection(methods)
        ## 2D fingerprint calculation
        fp_d = mol_fps(mol, methods)
        if "MCS" in methods:
            ## Maximum common substructure needs to compare molecules.
            fp_d["MCS"] = mol
        if methods_3d or methods_program:
            ## 3D fingerprint calculation
            if mol.GetNumConformers() == 0:
                mol_3d = conftools.genConf(
                    mol,
                    num_conf=self.n_conf_,
                    removeHs=False,
                    rmsd_thresh=-1,
                    conf_opt=self.conf_opt_,
                    sort_confs=False,
                    prune_diverged=False,
                    n_proc=self.n_jobs_,
                )
            else:
                mol_3d = mol
            fp_d.update(mol_3dfps(mol_3d, methods_3d))
            if methods_program.intersection(methods):
                ## Program similarity comparison with file writing
                sdf_path = os.path.join(self.filepath_, f"{name}.sdf")
                sdf_writer = Chem.SDWriter(sdf_path)
                for j, conf in enumerate(mol_3d.GetConformers()):
                    mol_3d.SetProp("_Name", f"conf_{j}")
                    sdf_writer.write(mol_3d, confId=conf.GetId())
                sdf_writer.close()
                if "ALIGNIT" in methods:
                    fp_d["ALIGNIT"] = sdf_path
                if "SHAPEIT" in methods:
                    fp_d["SHAPEIT"] = sdf_path
                if "SHAEP" in methods:
                    mol2_path = os.path.join(self.filepath_, f"{name}.mol2")
                    cmd = f"obabel -isdf {sdf_path} -omol2 -O {mol2_path}"
                    with open(os.devnull, "w") as devnull:
                        subprocess.run(
                            cmd,
                            stdout=devnull,
                            stderr=devnull,
                            shell=True,
                            cwd=self.filepath_,
                        )
                    fp_d["SHAEP"] = mol2_path
        return fp_d

    def transform(self, smiles, methods=None, subpath="input", verbose=True):
        """
        Convert SMILES to fingerprints.

        Parameters
        ----------
        smiles : List[str]
            Input SMILES
        methods : List[str]
            Fingerprint methods.

        Returns
        -------
        X_d : Dict[str, Any]
            Keys are methods, values are fingerprint matrices. There
            are 6 different types:
            sparse.csr_matrix[uint8] of shape (len(smiles), fp_size)
                Sparse fingerprint
            2d-array of shape (len(smiles), fp_size)
                Dense fingerprint
            3d-array of shape (len(smiles), n_conf_, fp_size)
                Dense 3D fingerprint with multiple conformers
            List[sparse.csr_matrix[uint8] of shape (n_conf_, fp_size)] of length len(smiles)
                Sparse 3D fingerprint with multiple conformers
            List[Mol] of length len(smiles)
                Molecule used for similarity comparison
            List[str] of length len(smiles):
                Filename of 3D molecules used for similarity comparison
        idx_smi : 1d-array[int] of length len(smiles)
            Record index of input SMILES, and -1 indicate invalid SMILES.
        """

        if methods is None:
            methods = self.methods_.copy()
        methods_input = {}
        for method in fpbits_to_method(methods):
            method = method.partition(".")[0]
            if method in ["SHAPE", "ESP"]:
                method = "SHAEP"
            if method in ["ALIGNITR"]:
                method = "ALIGNIT"
            if method in ["SHAPEITR"]:
                method = "SHAPEIT"
            methods_input[method] = []
        if verbose:
            begin_time = time.time()
            logging.info(
                f"Start calculate fingerprints of {len(smiles)} SMILES using methods: [{','.join(list(methods_input))}]"
            )
        idx_smi, X_saved, fps_d = self.get_fps(smiles, methods_input)
        idx_mols = np.nonzero(~X_saved.all(axis=1))[0]

        ## if all fingerprints has been loaded, there is no need for calculation.
        if len(idx_mols) == 0:
            X_d = {method: self.stack_fps(fps) for method, fps in fps_d.items()}
            if verbose:
                logging.info("All fingerprints read from saved records.")
            return X_d, idx_smi
        del fps_d
        methods = set()
        for i in idx_mols:
            smi = smiles[i]
            if type(smi) == Chem.Mol:
                mol = Chem.RemoveHs(smi, sanitize=False)
            else:
                mol = Chem.MolFromSmiles(smi)
            if not mol:
                ## for invalid molecules
                idx_smi[i] = -1
                continue
            methods = set()
            i_smi = idx_smi[i]
            if i_smi == -1:
                ## for valid SMILES not in records, canonize it
                smi_canon = Chem.MolToSmiles(mol)
                i_smi = self.smi_d_.get(smi_canon)
                if i_smi is not None:
                    ## If canonized SMILES is in records, read from record
                    idx_smi[i] = i_smi
                    for j, method in enumerate(methods_input):
                        fp = self.fps_d_[method][i_smi]
                        if fp is None:
                            methods.add(method)
                    if not methods:
                        ## if all fps in record
                        continue
                else:
                    ## If canonized SMILES not in records either, add a new record
                    i_smi = len(self.smi_d_)
                    self.smi_d_[smi_canon] = i_smi
                    idx_smi[i] = i_smi
                    if self.smi_enum_d_ is not None:
                        self.smi_enum_d_[smi_canon] = i_smi
                        if type(smi) == str:
                            self.smi_enum_d_[smi] = i_smi
                    for method in methods_input:
                        self.fps_d_[method].append(None)
                        methods.add(method)
            else:
                methods = {
                    method
                    for saved, method in zip(X_saved[i], methods_input)
                    if not saved
                }
            try:
                fp_d = self.mol_fps(
                    mol, methods, name=os.path.join(subpath, f"mol_{i_smi}")
                )
            except Exception:
                logging.exception(f"Error when calculating fingerprint of {i_smi} {self.smi_d_[i_smi]}")
                fp_d = {}
            if "IOSEQ" in methods:
                fp_d["IOSEQ"] = smiles[i]
            if "IOFRAG" in methods:
                fp_d["IOFRAG"] = smiles[i]
            if "IOSHAPE" in methods:
                fp_d["IOSHAPE"] = smiles[i]
            for method, fp in fp_d.items():
                if fp is not None:
                    self.fps_d_[method][i_smi] = fp
        X_d = {}
        for method in methods.intersection(FP_FUNC_D):
            fps_raw = self.fps_d_[method]
            idx = [i for i in idx_smi[idx_mols] if i >= 0]
            fps = FP_FUNC_D[method]([fps_raw[i] for i in idx])
            for i_smi, fp in zip(idx, fps):
                self.fps_d_[method][i_smi] = fp
        for method in methods_input:
            fps = self.fps_d_.get(method)
            if fps is None:
                logging.warning(f"Method {method} has no calculated record for output.")
            else:
                X_d[method] = self.stack_fps([fps[i] if i >= 0 else None for i in idx_smi])
        if verbose:
            logging.info(
                f"Get {len(X_d)} / {len(methods_input)} fingerprints for {len(smiles)} SMILES, time: {time.time() - begin_time:.1f} s"
            )
        return X_d, idx_smi

    def similarity(self, X_d, ref_name="SIM"):
        """
        Get similarity matrices between input fingerprint matrices and
        reference fingerprint matrices saved.

        Parameters
        ----------
        X_d : Dict[str, Any]
            Keys are methods, values are fingerprint matrices. There
            are 6 different types:
            sparse.csr_matrix[uint8] of shape (len(smiles), fp_size)
                Sparse fingerprint
            sparse.csr_matrix[uint8] of shape (len(smiles) * n_conf_, fp_size)
                Sparse 3D fingerprint with multiple conformers
            2d-array of shape (len(smiles), fp_size)
                Dense fingerprint
            3d-array of shape (len(smiles), n_confs_, fp_size)
                Dense 3D fingerprint with multiple conformers
            List[Mol] of length len(smiles)
                Molecule used for similarity comparison
            List[str] of length len(smiles):
                Filename of 3D molecules used for similarity comparison

        Returns
        -------
        S_d : Dict[str, 2d-array[float32]]
            Keys are methods, values are similarity matrices.
        """
        begin_time = time.time()
        S_d = {}
        for method, X_ref in self.X_ref_d_[ref_name].items():
            X = X_d.get(method)
            if X is None:
                pass
            elif method == "MCS":
                S_d[method] = sim_mcs(X, X_ref)
            elif method == "ALIGNIT":
                S = self.sim_alignit(X, X_ref)
                S_d["ALIGNIT"], S_d["ALIGNITR"] = S[:, :, 0], S[:, :, 1]
            elif method == "SHAPEIT":
                S = self.sim_shapeit(X, X_ref)
                S_d["SHAPEIT"], S_d["SHAPEITR"] = S[:, :, 0], S[:, :, 1]
            elif method == "SHAEP":
                S = self.sim_shaep(X, X_ref)
                S_d["SHAEP"], S_d["SHAPE"], S_d["ESP"] = (
                    S[:, :, 0],
                    S[:, :, 1],
                    S[:, :, 2],
                )
            elif method.startswith("3D"):
                if type(X) == np.ndarray:
                    S = similarity(
                        X.reshape(-1, X.shape[-1]),
                        X_ref.reshape(-1, X_ref.shape[-1]),
                        verbose=False,
                    )
                    S_d[method] = S.reshape(X.shape[0], -1, S.shape[1]).max(axis=1)
                elif type(X) == sparse.csr_matrix:
                    S = similarity(X, X_ref, verbose=False)
                    S_d[method] = S.reshape(
                        (X.shape[0] // self.n_conf_, self.n_conf_, -1)
                    ).max(axis=1)
            elif method in ["PAIR", "TORSION"]:
                S_d[method] = similarity(X, X_ref, a=0.5, b=0.5, verbose=False)
            else:
                S_d[method] = similarity(X, X_ref, verbose=False)
        logging.info(
            f"Get {len(S_d)} similarity matrices calculated, time: {time.time() - begin_time:.1f} s"
        )
        return S_d

    def sim_alignit(self, input_files, ref_files):
        S = np.zeros((len(input_files), len(ref_files), 2), dtype=np.float32)
        for j, ref_file in enumerate(ref_files):
            ref_name, ref_type = os.path.splitext(os.path.split(ref_file)[1])
            ref_type = ref_type.upper()[1:]
            output_path = self.init_filepath("alignit")
            output_files = []
            cmds = []
            for input_file in input_files:
                if input_file:
                    name = os.path.splitext(os.path.split(input_file)[1])[0]
                    output_file = os.path.join(output_path, f"{ref_name}_{name}.tab")
                    if os.path.isfile(output_file):
                        output_files.append(output_file)
                        continue
                    cmds.append(
                        " ".join(
                            [
                                "align-it",
                                f"--reference {ref_file}",
                                f"--refType {ref_type}",
                                f"--dbase {input_file}",
                                "--dbType SDF",
                                f"--scores {output_file}",
                                "--best 1",
                                "--rankBy TANIMOTO",
                                "--quiet",
                                "--noHybrid",
                                "--noNormal",
                            ]
                        )
                    )
                    output_files.append(output_file)
                    if len(cmds) == self.n_jobs_:
                        with open(os.devnull, "w") as devnull:
                            subprocess.run(
                                " & ".join(cmds),
                                stdout=devnull,
                                stderr=devnull,
                                shell=True,
                                cwd=self.filepath_,
                            )
                        cmds = []
                else:
                    output_files.append(None)
            else:
                with open(os.devnull, "w") as devnull:
                    subprocess.run(
                        " & ".join(cmds),
                        stdout=devnull,
                        stderr=devnull,
                        shell=True,
                        cwd=self.filepath_,
                    )
            for i, output_file in enumerate(output_files):
                if not os.path.isfile(output_file):
                    logging.info(f"No alignit output: {output_file}")
                    continue
                with open(output_file) as file:
                    txt_frags = file.read().strip().split("\t")
                    if len(txt_frags) >= 9:
                        S[i, j] = (txt_frags[8], txt_frags[9])
        return S

    def sim_shapeit(self, input_files, ref_files):
        S = np.zeros((len(input_files), len(ref_files), 2), dtype=np.float32)
        for j, ref_file in enumerate(ref_files):
            ref_name, ref_type = os.path.splitext(os.path.split(ref_file)[1])
            ref_type = ref_type.upper()[1:]
            output_path = self.init_filepath("shapeit")
            output_files = []
            cmds = []
            for input_file in input_files:
                if input_file:
                    name = os.path.splitext(os.path.split(input_file)[1])[0]
                    output_file = os.path.join(output_path, f"{ref_name}_{name}.tab")
                    if os.path.isfile(output_file):
                        output_files.append(output_file)
                        continue
                    cmds.append(
                        " ".join(
                            [
                                "shape-it",
                                f"--reference {ref_file}",
                                f"--dbase {input_file}",
                                f"--scores {output_file}",
                                "--best 1",
                                "--rankBy TANIMOTO",
                            ]
                        )
                    )
                    output_files.append(output_file)
                    if len(cmds) == self.n_jobs_:
                        with open(os.devnull, "w") as devnull:
                            subprocess.run(
                                " & ".join(cmds),
                                stdout=devnull,
                                stderr=devnull,
                                shell=True,
                                cwd=self.filepath_,
                            )
                        cmds = []
                else:
                    output_files.append(None)
            else:
                with open(os.devnull, "w") as devnull:
                    subprocess.run(
                        " & ".join(cmds),
                        stdout=devnull,
                        stderr=devnull,
                        shell=True,
                        cwd=self.filepath_,
                    )
            for i, output_file in enumerate(output_files):
                if not os.path.isfile(output_file):
                    logging.info(f"No shapeit output: {output_file}")
                    continue
                with open(output_file) as csvfile:
                    reader = csv.DictReader(csvfile, delimiter="\t")
                    for row in reader:
                        S[i, j] = (
                            row["Shape-it::Tanimoto"],
                            row["Shape-it::Tversky_Ref"],
                        )
                        break
        return S

    def sim_shaep(self, input_files, ref_files):
        """
        Parameters
        ----------
        input_files : List[str]
            Filepath of *.mol2 format.
        ref_files : List[str]
            Reference filepath of *.mol2 format.
        """
        S = np.zeros((len(input_files), len(ref_files), 3), dtype=np.float32)
        for j, ref_file in enumerate(ref_files):
            ref_name = os.path.splitext(os.path.split(ref_file)[1])[0]
            output_path = self.init_filepath("shaep")
            output_files = []
            cmds = []
            for input_file in input_files:
                if input_file:
                    name = os.path.splitext(os.path.split(input_file)[1])[0]
                    output_file = os.path.join(output_path, f"{ref_name}_{name}.tab")
                    if os.path.isfile(output_file):
                        output_files.append(output_file)
                        continue
                    cmds.append(
                        " ".join(
                            [
                                f"shaep --input-file {ref_file}",
                                f"--query {input_file}",
                                f"--query {input_file}",
                                f"--output-file {output_file}",
                                "--maxhits 1",
                            ]
                        )
                    )
                    output_files.append(output_file)
                    if len(cmds) == self.n_jobs_:
                        with open(os.devnull, "w") as devnull:
                            subprocess.run(
                                " & ".join(cmds),
                                stdout=devnull,
                                stderr=devnull,
                                shell=True,
                                cwd=self.filepath_,
                            )
                        cmds = []
                else:
                    output_files.append(None)
            else:
                with open(os.devnull, "w") as devnull:
                    subprocess.run(
                        " & ".join(cmds),
                        stdout=devnull,
                        stderr=devnull,
                        shell=True,
                        cwd=self.filepath_,
                    )
            for i, output_file in enumerate(output_files):
                if not os.path.isfile(output_file):
                    continue
                with open(output_file) as csvfile:
                    reader = csv.DictReader(csvfile, delimiter="\t")
                    records = []
                    for row in reader:
                        if row:
                            records.append(
                                [
                                    row["best_similarity"],
                                    row["shape_similarity"],
                                    row["ESP_similarity"],
                                ]
                            )
                    if records:
                        S[i, j] = np.array(records, dtype=np.float32).max(axis=0)
        return S

    def fps_summary(self, X_fp, prop_df=None):
        """
        Get bit summary information of fingerprint matrix.

        Parameters
        ----------
        X_fp : Dict[str, sparse.csr_matrix[uint8] or 2darray[uint8]] or DataFrame
            Fingerprint matrix dictionary or dense fingerprint matrix.
        prop_df : DataFrame, optional
            Property dataframe to summarize bit effects by statistical
            difference between nonzero-bit and zero-bit.

        Returns
        -------
        bitinfo_df : DataFrame of shape (:, 5)
            Information table of fingerprint bits. With index BIT_ID,
            and 5 columns [COUNT, NONZERO, MAX_ORDER, EVEN_ORDER,
            SMILES_FRAGS].
            BIT_ID : index, int
                Fingerprint and bit ID, with format '{METHOD}_{bit}'.
            COUNT : int
                Total counts of fingerprint bit.
            NONZERO : float
                Proportions of existence in molecules of fingerprint
                bit.
            MAX_ORDER : int
                Maximum count of fingerprint bit in a molecule.
            EVEN_ORDER : float
                Proportion of non-zero even count of fingerprint bit.
            SMILES_FRAGS : str
                SMILES / SMARTS representations of fingerprint bit.
            DIAMETER : str
                Bonds diameter of the fingerprint fragments, e.g.
                single atom: diameter 0, two connected atoms: diameter
                1, 4-atom path: diameter 3.
        sim_nonzero_d : Dict[str, 1d-array[float32]]
            Similarity of every fingerprint row with library based on
            nonzero distribution of bits.
        """
        sim_nonzero_d = {}
        if type(X_fp) == dict:
            bitinfo_df_ls = []
            for method, X in X_fp.items():
                if type(X) in [np.ndarray, sparse.csr_matrix]:
                    bitinfo_sub_df, sim_nonzero = fps_summary(X, prop_df)
                    bitinfo_df_ls.append(
                        bitinfo_sub_df.rename(lambda x: f"{method}_{x}")
                    )
                    sim_nonzero_d[method] = sim_nonzero.reshape(X.shape[0], -1).mean(
                        axis=1
                    )
            if bitinfo_df_ls:
                bitinfo_df = pd.concat(bitinfo_df_ls)
            else:
                bitinfo_df = None
        elif type(X_fp) == pd.DataFrame:
            bitinfo_df, sim_nonzero = fps_summary(X_fp.values)
            bitinfo_df.index = X_fp.columns
        if bitinfo_df is not None:
            bitinfo_df["SMILES_FRAG"], bitinfo_df["DIAMETER"] = get_fpbit_info(
                bitinfo_df.index
            )
            bitinfo_df.index.name = "FPBIT_ID"
            logging.info(f"Get summary information of {len(bitinfo_df)} fingerprint bits")
        return bitinfo_df, sim_nonzero_d

    def get_dense(self, X_d, nonzero=-1, colinear=-1):
        """
        Get dense matrix from fingerprint matrix dictionary.

        Parameters
        ----------
        X_d : Dict[str, 2darray or sparse.csr_matrix[uint8]]
            Fingerprint matrix dictionary. Keys are fingerprint methods,
            values are fingerprint matrices. There are 4 different
            types can be converted:
            sparse.csr_matrix[uint8] of shape (n, fp_size)
                Sparse fingerprint
            2d-array of shape (n, fp_size)
                Dense fingerprint
            3d-array of shape (n, n_confs_, fp_size)
                Dense 3D fingerprint with multiple conformers
            List[sparse.csr_matrix[uint8] of shape (n_confs_, fp_size)] of length n
                Sparse 3D fingerprint with multiple conformers
        nonzero : float
            If 0 <= nonzero < 1, create dense matrix from fingerprint
            bits with nonzero frequency above threshold.
            If negative, use pre-setting fingerprint bits.
        colinear : float
            If 0 < colinear <= 1, filter columns by colinearity above
            threshold.

        Returns
        -------
        X_df : DataFrame
            Dense fingerprint matrix with columns "{METHOD}_{bit}".
        """
        X_fp_d = {}
        for method, X in X_d.items():
            if X is None:
                pass
            elif type(X) == list:
                fp = first_notnone(X)
                if sparse.isspmatrix_csr(fp):
                    fps = []
                    for fp in X:
                        if fp is not None:
                            fps.append(
                                split_sparse_fp(fp)[random.randint(0, fp.shape[0] - 1)]
                            )
                        else:
                            fps.append(None)
                    X_fp_d[method] = stack_sparse_fp(fps)
            elif len(X.shape) == 3:
                X_fp_d[method] = X[
                    np.arange(X.shape[0]), np.random.randint(0, X.shape[1], X.shape[0])
                ]
            elif len(X.shape) == 2:
                X_fp_d[method] = X
        if nonzero >= 0:
            bitinfo_df, sim_nonzero_d = self.fps_summary(X_fp_d)
            fpbits = bitinfo_df.index[bitinfo_df["NONZERO"] >= nonzero]
            method_d = fpbits_to_method(fpbits)
        else:
            method_d = self.methods_.copy()
        X_df_ls = []
        for method, bits in method_d.items():
            if not bits:
                continue
            if method in X_fp_d:
                X_dense = get_dense(X_fp_d[method], bits)
            else:
                logging.warning(
                    f"Method {method} with {len(bits)} bits not in fingerprints: {bits[:5]} ..."
                )
                X_dense = np.zeros((len(X_dense), len(bits)), dtype=np.uint8)
            X_df_ls.append(
                pd.DataFrame(X_dense, columns=[f"{method}_{bit}" for bit in bits])
            )
        X_df = pd.concat(X_df_ls, axis=1)
        if 0 < colinear < 1:
            X_df = filter_dense(X_df, freq=nonzero, colinear=colinear)
        logging.info(
            f"Get dense matrix of {X_df.shape[0]} smiles and {X_df.shape[1]} bits"
        )
        return X_df

    def predict(self, smiles, nonzero=-1, colinear=-1, name="", prop_df=None):
        """
        Predict fingerprint-based QSAR and similarity values from SMILES.

        Parameters
        ----------
        smiles : List[str]
            Input SMILES.
        nonzero : float
            If 0 <= nonzero < 1, create dense matrix from fingerprint
            bits with nonzero frequency above threshold.
            If negative, use pre-setting fingerprint bits.
        colinear : float
            If 0 < colinear <= 1, filter columns by colinearity above
            threshold.
        name : str
            Prefix name of file saved.
        prop_df : DataFrame, optional
            Property dataframe to summarize bit effects by statistical
            difference between nonzero-bit and zero-bit. Index are
            NAME of input SMILES.

        Returns
        -------
        Y : DataFrame[float32]
            Predicted QSAR / similarity values. Index are canonical
            SMILES of input.
        """
        begin_time = time.time()
        if self.filepath_:
            save_path = os.path.join(self.filepath_, name)
        X_d, idx_smi = self.transform(smiles)
        if self.filepath_:
            save_fps_dict(f"{save_path}_fps.npz", X_d)
        record_smiles = list(self.smi_d_)
        smiles = [record_smiles[i] for i in idx_smi]
        self.columns_ = []
        Ys = []
        try:
            bitinfo_df, sim_nonzero_d = self.fps_summary(X_d, prop_df=prop_df)
            if self.filepath_ and bitinfo_df is not None:
                bitinfo_df.to_csv(f"{save_path}_bitinfo.csv", float_format="%g")
            if prop_df is not None:
                for col in prop_df.columns:
                    model = ensemble.RandomForestRegressor(n_estimators=100, oob_score=True, max_features="sqrt", random_state=0)
                    model.Y_ = prop_df[col].astype(np.float32).fillna(prop_df[col].mean())
                    self.model_d_[f"FACTOR_{col}"] = model
        except Exception as error:
            logging.exception(f"get bitinfo error {str(error)}")
        if nonzero >= 0 or self.model_d_:
            try:
                if any(self.methods_.values()):
                    X_df = self.get_dense(X_d, nonzero=-1, colinear=-1)
                else:
                    X_df = self.get_dense(X_d, nonzero=nonzero, colinear=colinear)
            except Exception as error:
                logging.exception(f"get dense error {str(error)}")
            if self.filepath_:
                X_df.index = smiles
                X_df.index.name = "SMILES"
                if prop_df is not None:
                    X_df.reset_index().join(prop_df.reset_index()).to_csv(f"{save_path}_fps.csv", float_format="%g", index=False)
                else:
                    X_df.to_csv(f"{save_path}_fps.csv", float_format="%g")
            try:
                model_fit(self.model_d_, X_df)
            except Exception:
                logging.exception("Model fitting error")
            self.set_methods(X_df.columns)
            if self.filepath_ and prop_df is not None and bitinfo_df is not None:
                for col in prop_df.columns:
                    fi = self.model_d_[f"FACTOR_{col}"].feature_importances_
                    bitinfo_df[f"BITFI_{col}"] = pd.Series(X_df.shape[1] * fi / fi.sum(), X_df.columns)
                bitinfo_df.to_csv(f"{save_path}_bitinfo.csv", float_format="%g")
            try:
                Y_d = model_predict(self.model_d_, X_df)
                for key, Y in Y_d.items():
                    Ys.append(Y)
                    if len(Y.shape) == 1:
                        self.columns_.append(key)
                    else:
                        self.columns_.extend([f"{key}{i + 1}" for i in range(Y.shape[1])])
            except Exception:
                logging.exception("Model prediction error")
            del X_df
        for simname, X_ref_d in self.X_ref_d_.items():
            statfunc_d = self.statfunc_ref_d_[simname]
            S_d = self.similarity(X_d, ref_name=simname)
            if self.filepath_:
                np.savez_compressed(f"{save_path}_sim_{simname}.npz", **S_d)
            for method, S in S_d.items():
                for key, func in statfunc_d.items():
                    Ys.append(func(S))
                    self.columns_.append(f"{simname}_{method}_{key}")
        if self.columns_:
            Y_df = pd.DataFrame(
                np.column_stack(Ys), columns=self.columns_, index=smiles, dtype=np.float32
            )
        else:
            Y_df = pd.DataFrame(np.zeros((len(smiles), 0)), index=smiles, dtype=np.float32)
        logging.info(f"Finish vectorize and prediction step, time: {time.time() - begin_time:.1f}s")
        return Y_df


class SmartsFingerprint(object):
    """
    Fingerprinting molecules by SMARTS patterns.

    Parameters
    ----------
    smarts : List[str]
        Input SMARTS for filtering
    names : List[str]
        Name notation for SMARTS pattern
    assign_name : str
        Fingerprint name to assign

    Attributes
    ----------
    catalog_ : FilterCatalog
        SMARTS catalog for filtering.
    submols_ : List[Mol]
        Molecule substructures of SMARTS.
    """

    def __init__(self, smarts, names=None, assign_name=None):
        self.catalog_ = FilterCatalog.FilterCatalog()
        self.submols_ = []
        if not names:
            names = [str(i) for i in range(len(smarts))]
        for i, (sma, name) in enumerate(zip(smarts, names)):
            submol = Chem.MolFromSmarts(sma)
            if submol:
                entry = FilterCatalog.SmartsMatcher(name, submol, 1, 255)
            else:
                entry = FilterCatalog.SmartsMatcher(name, sma, 1, 255)
            self.submols_.append(submol)
            self.catalog_.AddEntry(FilterCatalog.FilterCatalogEntry(f"{i} ?{sma} (name:{name.replace(' ', '_')})", entry))
        logging.info(f"Load {sum(1 for submol in self.submols_ if submol)} valid / {len(self.submols_)} total SMARTS")
        if assign_name:
            self.assign(assign_name)

    def __len__(self):
        return len(self.submols_)

    def get_fp(self, mol):
        """
        Parameters
        ----------
        mol : Mol
            Input molecules to find SMARTS.

        Returns
        -------
        idx : List[int]
            SMARTS index found in SMILES.
        counts : List[int]
            Counts found of SMARTS idx.
        matches : List[Tuple[int, ...]]
            Matches atoms of SMARTS found.
        """
        idx = []
        counts = []
        if mol:
            for entry in self.catalog_.GetMatches(mol):
                idx.append(int(entry.GetDescription().partition(' ')[0]))
                counts.append(len(entry.GetFilterMatches(mol)))
        idx = np.array(idx, dtype=np.uint32)
        counts = np.array(counts, dtype=np.uint8)
        return idx, counts

    def match_bit(self, mol, bit):
        entry = self.catalog_.GetEntry(int(bit))
        matches = [tuple(i_atom for i, i_atom in match.atomPairs) for match in entry.GetFilterMatches(mol)]
        return matches

    def assign(self, name):
        FP_MOLFUNC_D[name] = (self.get_fp, self.match_bit)
        bit_d = {}
        for bit in range(self.catalog_.GetNumEntries()):
            entry = self.catalog_.GetEntry(bit)
            submol = self.submols_[bit]
            if submol:
                diameter = submol.GetNumAtoms() - 1
            else:
                diameter = np.nan
            bit_d[bit] = (entry.GetDescription().partition(" ")[-1], diameter)
        FP_BIT_D[name] = bit_d
        self.name_ = name
        logging.info(f"Assign SMARTS fingerprinter: {self.name_}")


class FragFingerprint(object):
    """
    Fingerprinting molecules by fragment cutter.

    Parameters
    ----------
    frag_func : Callable[Mol, [List[str]]]
        A function cutting molecule into fragments.
    assign_name : str
        Fingerprint name to assign.
    aggregate : bool
        Whether aggregate fingerprint bits of similar type to new bits.
    """
    def __init__(self, frag_func, assign_name=None, aggregate=True):
        self.frag_func_ = frag_func
        self.aggregate_ = aggregate
        if assign_name:
            self.assign(assign_name)

    def get_fp(self, mol):
        count_d = frag_hashmap(self.frag_func_(mol))
        if self.aggregate_:
            agg_count_d = collections.Counter()
            for bit, (frag, count) in count_d.copy().items():
                agg_bit = bit // 100000
                count_d[agg_bit] = (frag, 0)
                agg_count_d[agg_bit] += count
            for agg_bit, agg_count in agg_count_d.items():
                count_d[agg_bit] = (count_d[agg_bit][0], agg_count)
        bit_d = FP_BIT_D[self.name_]
        for bit, (frag, count) in count_d.items():
            if bit not in bit_d:
                atoms = re.findall(
                    r"\[[\w+-@:\d]+\]|[BCONSPFQWIbcnops]",
                    frag.replace("Cl", "Q").replace("Br", "W"),
                )
                bit_d[bit] = (frag, len(atoms) - 1)
        idx = np.fromiter(count_d, np.int64)
        counts = np.array([val[1] for val in count_d.values()], np.uint8)
        return idx, counts

    def match_bit(self, mol, bit):
        bit_d = FP_BIT_D[self.name_]
        frag, diameter = bit_d.get(bit, (None, np.nan))
        if frag:
            submol = Chem.MolFromSmarts(frag)
            matches = list(mol.GetSubstructMatches(submol, maxMatches=255))
        else:
            matches = []
        return matches

    def assign(self, name):
        FP_MOLFUNC_D[name] = (self.get_fp, self.match_bit)
        self.name_ = name
        logging.info(f"Assign fragment fingerprinter: {self.name_}")


def similarity(X, X_ref=None, a=1, b=1, verbose=True):
    """
    Calculate Tversky / cosine similarity between two fingerprint
    arrays.

    Parameters
    ----------
    X : 2d-array or sparse.csr_matrix[int] of shape (n, fp_size)
        Fingerprint matrix. There are several types available:
            sparse.csr_matrix[int]: Tversky similarity
            2d-array[int]: Tversky similarity
            2d-array[float]: Cosine similarity
    X_ref : 2d-array or sparse.csr_matrix[int] of shape (n_ref, fp_size)
        optional (default=None)
        Fingerprint matrix for comparison, should be same type as `X`.
        If None, use `X` instead.
    a, b: float
        Tversky index of int. If a = b = 1, it is Tanimoto coefficient, if
        a = b = 0.5, it is Dice coefficient
    verbose : bool
        Whether print information.

    Returns
    -------
    S : ndarray[float32] of shape (n, sample_ref_size)
        Similarity matrix of fingerprint.

    See also
    --------
    hcluster_linkage
        Perform and plot hierarchical clustering from a similarity
        matrix.
    hcluster_group
        Get cluster group from hierachical clustering matrix and
        similarity threshold.
    """
    begin_time = time.time()
    if type(X) == sparse.csr_matrix:
        pos_d = collections.defaultdict(list)
        X_count = np.diff(X.indptr)
        for i, bit in zip(
            np.repeat(np.arange(X.shape[0]), X_count), X.indices
        ):
            pos_d[bit].append(i)
    if X_ref is None:
        X_ref = X
        if type(X) == sparse.csr_matrix:
            X_ref_count = X_count
            pos_ref_d = pos_d
    elif len(X.shape) != len(X_ref.shape):
        logging.error(f"Dimension error: {X.shape} {X_ref.shape}")
    elif type(X_ref) == sparse.csr_matrix:
        pos_ref_d = collections.defaultdict(list)
        X_ref_count = np.diff(X_ref.indptr)
        for i, bit in zip(
            np.repeat(np.arange(X_ref.shape[0]), X_ref_count), X_ref.indices
        ):
            pos_ref_d[bit].append(i)
    S = np.zeros((X.shape[0], X_ref.shape[0]), dtype=np.float32)
    if type(X) == sparse.csr_matrix:
        for bit, pos in pos_d.items():
            pos_ref = pos_ref_d.get(bit)
            if pos_ref:
                S[np.ix_(pos, pos_ref)] += 1
        for i, i_sim in enumerate(S):
            if i_sim.any():
                S[i] /= a * X_count[i] + b * X_ref_count + (1 - a - b) * i_sim
    elif np.issubdtype(X.dtype, np.floating):
        X_norm = np.linalg.norm(X, axis=1)
        X_ref_norm = np.linalg.norm(X_ref, axis=1)
        for i, (x, x_norm) in enumerate(zip(X, X_norm)):
            if x_norm > 0:
                S[i] = X_ref @ x / x_norm
        S /= np.maximum(X_ref_norm, 1e-8)
    elif type(X) == np.ndarray:
        X_count = np.zeros(len(X), dtype=np.int32)
        X_ref_count = np.zeros(len(X_ref), dtype=np.int32)
        for count in range(int(X.max()).bit_length()):
            X_bool = X >= (1 << count)
            X_ref_bool = X_ref >= (1 << count)
            for bit in X_bool.any(axis=0).nonzero()[0]:
                S[np.ix_(X_bool[:, bit], X_ref_bool[:, bit])] += 1
            X_count += X_bool.sum(axis=1)
            X_ref_count += X_ref_bool.sum(axis=1)
        for i, i_sim in enumerate(S):
            if i_sim.any():
                S[i] /= a * X_count[i] + b * X_ref_count + (1 - a - b) * i_sim
    if verbose:
        logging.info(
            f"Calculate a {S.shape[0]} by {S.shape[1]} similarity matrix on {X.shape[-1]} dimensions, time: {time.time() - begin_time:.1f} s"
        )
    return S


def sim_mcs(mols, mols_ref):
    """
    Get maximum common substructure (MCS) similarity matrix between
    input molecules and reference molecules saved.

    Parameters
    ----------
    mols : List[Mol]
        Input molecules.
    mols_ref : List[Mol]
        Reference molecules.

    Returns
    -------
    S : 2darray[float32] of shape (len(mols), len(mols_ref))
        Similarity matrix of MCS.
    """
    S = np.zeros((len(mols), len(mols_ref)), dtype=np.float32)
    n_atoms_ref = np.array(
        [mol_ref.GetNumAtoms() for mol_ref in mols_ref], dtype=np.float32
    )
    for i, mol in enumerate(mols):
        if not mol:
            continue
        n_atom = mol.GetNumAtoms()
        for j, mol_ref in enumerate(mols_ref):
            if not mol_ref:
                continue
            mcs = rdFMCS.FindMCS(
                [mol, mol_ref],
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                matchValences=True,
            )
            S[i, j] = mcs.numAtoms
        S[i] /= n_atom + n_atoms_ref - S[i]
    return S


def save_pharm_shaep(filepath, mol_p):
    lines = ["@<TRIPOS>ATOM"]
    mol_name = mol_p.GetProp("_Name")
    conf = mol_p.GetConformer()
    for atom, coord in zip(mol_p.GetAtoms(), conf.GetPositions()):
        atomtype = atom.GetSymbol()
        if atomtype == "N":
            atomcharge = 0.5
        elif atomtype == "O":
            atomcharge = -0.3
        else:
            atomtype = "C"
            atomcharge = 0
        cur_line = (
            [str(len(lines)), atomtype]
            + [f"{val:.3f}" for val in coord]
            + [f"{atomtype}.3", str(len(lines)), "UNL1", f"{atomcharge:.3f}"]
        )
        lines.append("\t".join(cur_line))
    lines = [
        "@<TRIPOS>MOLECULE",
        f"Molecule {mol_name}",
        f"{len(lines) - 1}",
        "SMALL",
        "USER_CHARGES",
    ] + lines
    with open(filepath, "w") as file:
        file.write("\n".join(lines))


def save_pharm_alignit(filepath, mol_p):
    alpha_d = {
        "LIPO": "0.7",
        "AROM": "0.7",
        "HACC": "1.0",
        "HDON": "1.0",
        "POSC": "1.0",
        "NEGC": "1.0",
    }
    phar_d = dict(zip(PHARMATOMS, alpha_d))
    conf = mol_p.GetConformer()
    lines = [mol_p.GetProp("_Name")]
    for atom, coord in zip(mol_p.GetAtoms(), conf.GetPositions()):
        phar = phar_d.get(atom.GetSymbol())
        if phar:
            lines.append(
                "\t".join(
                    [phar]
                    + [f"{val:.3f}" for val in coord]
                    + [alpha_d[phar]]
                    + ["0"] * 4
                )
            )
    lines.append("$$$$")
    with open(filepath, "w") as file:
        file.write("\n".join(lines))


def get_dense(X, bits=1024):
    """
    Get dense array of Morgan fingerprint from sparse matrix.

    Parameters
    ----------
    X : sparse.csr_matrix of shape (n, fp_size)
        Sparse fingerprint matrix
    bits : int or List[int]
        Set the length of compressed fingerprint. If a list, extract
        the bits from the sparse matrix.

    Returns
    -------
    X_dense : ndarray[uint8] of shape (n, bits)
        Compressed array of fingerprint.
    """
    if type(X) == np.ndarray:
        X_dense = np.nan_to_num(X[:, bits], posinf=0, neginf=0)
    elif type(X) == sparse.csr_matrix:
        if type(bits) == int:
            X_dense = np.zeros((X.shape[0], bits), dtype=X.dtype)
            for i, (j_begin, j_end) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
                np.add.at(X_dense[i], X.indices[j_begin:j_end] % bits, X.data[j_begin:j_end])
        else:
            X_dense = np.zeros((X.shape[0], len(bits)), dtype=X.dtype)
            argidx = np.argsort(X.indices)
            idx_sample = np.repeat(np.arange(X.shape[0], dtype=np.uint32), np.diff(X.indptr))[argidx]
            idx_data = X.data[argidx]
            samples = np.searchsorted(X.indices[argidx], np.add.outer(bits, [0, 1]))
            for i, val in enumerate(samples):
                idx_bit = slice(val[0], val[1])
                X_dense[idx_sample[idx_bit], i] = idx_data[idx_bit]
    return X_dense


def filter_dense(X_df, freq=0, colinear=1):
    """
    Filter dense fingerprint matrix by bit frequency and colinearity.

    Parameters
    ----------
    X_df : DataFrame
        Input fingerprint matrix.
    freq : float
        Frequency threshold to filter columns.
    colinear : float
        Colinearity threshold to filter columns.

    Returns
    -------
    X_out_df : DataFrame
        Fingerprint matrix with sparse and colinear columns filtered.
    """
    begin_time = time.time()
    freqs = 0.5 - np.abs((X_df > X_df.mean()).mean() - 0.5)
    stds = X_df.std(axis=0)
    keeps_freq = freqs.values >= max(0, freq)
    X_df_std = (
        X_df.loc[:, keeps_freq] - X_df.loc[:, keeps_freq].mean(axis=0)
    ) / stds.loc[keeps_freq]
    keeps = np.ones(X_df_std.shape[1], dtype=bool)
    X = X_df_std.values.T
    for i, x in enumerate(X):
        if keeps[i]:
            r = X[i + 1:] @ x
            keeps[i + 1 + np.nonzero(np.abs(r) >= colinear * len(x))[0]] = 0
    logging.info(
        f"Filter columns with freq {freq} / correlation coefficient {colinear}: {sum(keeps)} left / {len(keeps)} freq-filtered / {len(stds)} total, time: {time.time() - begin_time:.1f} s"
    )
    X_out_df = X_df.iloc[:, np.nonzero(keeps_freq)[0][keeps]]
    return X_out_df


def hcluster_linkage(
    S, method="average", optimal_ordering=False, labels=None, figsize=(25, 10)
):
    """
    Perform and plot hierarchical clustering from a similarity matrix.

    Parameters
    ----------
    S : ndarray[float] of shape (n, n)
        Symmetric similarity matrix.
    method : str, {"average", "single", "complete", "weighted"}
        The linkage algorithm to use.
    optimal_ordering : bool, (default=False)
        If True, the linkage matrix will be reordered so that the
        distance between successive leaves is minimal. This results in
        a more intuitive tree structure when the data are visualized.
    labels : List[str], optional (default=None)
        Labels of sample for plot.
    figsize : Tuple[float, float]
        Plot width, height in inches.

    Returns
    -------
    Z : ndarray[float64] of shape (n - 1, 4)
        Linkage matrix of hierarchical clustering.

    References
    ----------
    * Scipy
      scipy.cluster.hierarchy.linkage
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    See also
    --------
    similarity
        Calculate Tanimoto similarity between two fingerprint matrices.
    hcluster_group
        Get cluster group from hierachical clustering matrix and
        similarity threshold.
    """
    Z = cluster.hierarchy.linkage(
        spatial.distance.squareform(1 - S),
        method=method,
        optimal_ordering=optimal_ordering,
    )
    if figsize:
        plt.figure(figsize=figsize)
        cluster.hierarchy.dendrogram(Z, labels=labels)
        plt.show()
    return Z


def hcluster_group(X, Z, similarity_threshold=0.5):
    """
    Get cluster group from hierachical clustering matrix and similarity
    threshold.

    Parameters
    ----------
    X : ndarray[uint8] of shape (n, fp_size)
        Fingerprint array.
    Z : ndarray[float64] of shape (n - 1, 4)
        Linkage matrix of hierarchical clustering.
    similarity_threshold : float
        Lowest threshold of similarity to form a group.

    Returns
    -------
    group : ndarray of shape (n_group,)
        Group index of fingerprints.
    group_X_sim : ndarray of shape (n_group, n)
        Similarity matrix between group centers and fingerprints.
    group_info_df : DataFrame of shape (n_group, 4)
        Information table of fingerprint group with 4 columns [count,
        idx_center, sim_mean, sim_center]:
        index : int
            group index
        count : int
            number of members belong to group.
        idx_center : int
            median fingerprint of group.
        sim_mean : float
            mean similarity between group center and its members.
        sim_center : float
            maximum similarity between group center and its members.

    See also
    --------
    similarity
        Calculate Tanimoto similarity between two fingerprint matrices.
    hcluster_linkage
        Perform and plot hierarchical clustering from a similarity
        matrix.
    """
    group = cluster.hierarchy.fcluster(
        Z, t=1 - similarity_threshold, criterion="distance"
    )
    group_count = np.bincount(group)
    group_center = np.vstack(
        [np.median(X[group == i], axis=0) for i in range(len(group_count))]
    ).astype(np.uint8)
    group_X_sim = similarity(group_center, X)
    group_info_df = pd.DataFrame(
        {
            "count": group_count,
            "idx_center": group_X_sim.argmax(axis=1),
            "sim_mean": [
                group_X_sim[i, group == i].mean() for i in range(len(group_count))
            ],
            "sim_center": group_X_sim.max(axis=1),
        }
    )
    return group, group_X_sim, group_info_df


def get_fpbit_info(fpbits):
    """
    Get SMILES / SMARTS representations of fingerprint bits.

    Parameters
    ----------
    fpbits : List[str]
        Features '{METHOD}_{bit}' to use for getting fragments.
        MORGAN and TOPO need pre-calculation.

    Returns
    -------
    frags : List[str]
        SMILES / ?SMARTS / (property) representations of fingerprint
        bits.
    diameters : List[float]
        If not 3D fingerprints, indicate bonds diameter of the
        fingerprint fragments as graph, e.g.
            single atom: diameter 0,
            2 connected atoms: diameter 1
            4-atom path: diameter 3.
        If 3D fingerprints, indicates geometry diameter of 1e-10 meter.
    """
    fpbit_d = {}
    method_d = fpbits_to_method(fpbits)
    for method, bits in method_d.items():
        frags = []
        diameters = []
        if method in FP_BIT_D:
            frags, diameters = zip(
                *[FP_BIT_D[method].get(bit, (None, np.nan)) for bit in bits]
            )
        elif method == "PAIR":
            for bit in bits:
                a1, dist, a2 = Pairs.ExplainPairScore(bit)
                frags.append(
                    f"?[{a1[0]};D{a1[1]}:{a1[2]}]{'*' * (dist-1)}[{a2[0]};D{a2[1]}:{a2[2]}]"
                )
                diameters.append(dist + 1)
        elif method == "TORSION":
            for i, bit in enumerate(bits):
                if 0 < i < len(bits) - 1:
                    bit += 2
                else:
                    bit += 1
                frags.append(
                    "?"
                    + "".join(
                        [
                            f"[{a[0]};D{a[1]}:{a[2]}]"
                            for a in Torsions.ExplainPathScore(bit)
                        ]
                    )
                )
                diameters.append(3)
        elif method == "ESTATE":
            calcs = ["COUNT", "ESUM"]
            for bit in bits:
                i_calc, i_smarts = divmod(bit, 79)
                frags.append(
                    f"?{EState.AtomTypes._rawD[i_smarts][1]} (stat:{calcs[i_calc]})"
                )
                diameters.append(0)
        elif method == "MACCS":
            for bit in bits:
                frag, count = MACCSkeys.smartsPatts.get(bit, ("?", 0))
                mol = Chem.MolFromSmarts(frag)
                if mol:
                    frags.append(f"?{frag} (count:{count})")
                    diameters.append(mol.GetNumAtoms() - 1)
                else:
                    frags.append(None)
                    diameters.append(np.nan)
        elif method == "AUTOCORR":
            calcs = ["i*j", "abs((i-mean)*(j-mean))", "(i-mean)*(j-mean)", "(i-j)**2"]
            props = ["Mass", "VdW", "ENeg", "Pol", "IonPol", "IState"]
            for bit in bits:
                i_calc, i_distprop = divmod(bit, 48)
                i_prop, dist = divmod(i_distprop, 8)
                frags.append(
                    f"(prop:{props[i_prop]} stat:{calcs[i_calc]} dist:{dist + 1})"
                )
                diameters.append(dist + 1)
        elif method == "VSA":
            stats_peoe = np.r_[-np.inf, np.arange(-0.3, 0.35, 0.05), np.inf]
            stats_slogp = [
                -np.inf,
                -0.4,
                -0.2,
                0,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                np.inf,
            ]
            stats_smr = [
                -np.inf,
                1.29,
                1.82,
                2.24,
                2.45,
                2.75,
                3.05,
                3.63,
                3.8,
                4,
                np.inf,
            ]
            props = (
                [
                    f"(prop:PEOE bin:[{stats_peoe[i]:.2f},{stats_peoe[i + 1]:.2f}))"
                    for i in range(14)
                ]
                + [
                    f"(prop:SlogP bin:[{stats_slogp[i]:.2f},{stats_slogp[i + 1]:.2f}))"
                    for i in range(12)
                ]
                + [
                    f"(prop:SMR bin:[{stats_smr[i]:.2f},{stats_smr[i + 1]:.2f}))"
                    for i in range(10)
                ]
            )
            frags, diameters = zip(*[(props[bit], np.nan) for bit in bits])
        elif method == "MQN":
            props = [
                "?[#6]", "?[#9]", "?[#17]", "?[#35]", "?[#53]", "?[#16]", "?[#15]", "?[#7;!R]", "?[#7;R]", "?[#8;!R]",
                "?[#8;R]", "?*", "?*!@;-*", "?*!@;=*", "?*!@;#*", "?*@;-*", "?*@;=*", "?*@;#*", "?**", "(ACC site)", "(ACC atom)",
                "(DON site)", "(DON atom)", "(ANI)", "(POS)", "?[D1]", "?[D2;!R]", "?[D3;!R]", "?[D4;!R]", "?[D2;R]",
                "?[D3;R]", "?[D4;R]", "(3-atom ring)", "(4-atom ring)", "(5-atom ring)", "(6-atom ring)", "(7-atom ring)", "(8-atom ring)", "(9-atom ring)", "(>=10-atom ring)",
                "(fuse-ring atom)", "(fuse-ring bond)"
            ]
            for bit in bits:
                frags.append(props[bit])
                diameters.append(np.nan)
        elif method in ["PAIRPROP", "DISTPROP"]:
            n_prop = len(ATOMPROPS)
            props1, props2 = np.triu_indices(n_prop, 1)
            for bit in bits:
                if bit < n_prop:
                    frags.append(f"(prop:{ATOMPROPS[bit]})")
                    diameters.append(0)
                else:
                    frags.append(
                        f"(prop:{ATOMPROPS[props1[bit - n_prop]]}~{ATOMPROPS[props2[bit - n_prop]]})"
                    )
                    diameters.append(1)
        elif method == "SCAF":
            props = ["Ring", "HeteroRing", "NonSp3CarbonRing"]
            dists = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, "inf"]
            for bit in bits:
                i_prop, dist = divmod(bit, 10)
                frags.append(f"({props[i_prop]} dist:[{dists[dist]},{dists[dist+1]}))")
                diameters.append(dists[dist])
        elif method in ["PHARM", "3DPHARM"]:
            pharms = sorted(PHARMS)
            props = [f"{i_pharm} {j_pharm}" for i, i_pharm in enumerate(pharms) for j_pharm in pharms[i:]]
            dists = [0, 2, 3, 5, 8, 13, "inf"]
            for bit in bits:
                i_prop, dist = divmod(bit, 6)
                if bit <= 5:
                    frags.append(
                        f"({PHARMS[bit]} | {props[i_prop]} dist:[{dists[dist]},{dists[dist+1]}))"
                    )
                else:
                    frags.append(f"({props[i_prop]} dist:[{dists[dist]},{dists[dist+1]}))")
                diameters.append(dists[dist])
        elif method == "3DAUTOCORR":
            props = ["Unit", "Mass", "VdW", "ENeg", "Pol", "IonPol", "IState", "Rcov"]
            for bit in bits:
                i_prop, dist = divmod(bit, 10)
                frags.append(f"(prop:{props[i_prop]} stat:EigVec dist:{dist + 1})")
                diameters.append(dist + 1)
        elif method == "3DRDF":
            props = ["Unit", "Mass", "VdW", "ENeg", "Pol", "IonPol", "IState"]
            for bit in bits:
                i_prop, dist = divmod(bit, 30)
                frags.append(f"(prop:{props[i_prop]} stat:i*j dist:{dist / 2 + 1})")
                diameters.append(dist / 2 + 1)
        elif method == "3DMORSE":
            props = ["Unit", "Mass", "VdW", "ENeg", "Pol", "IonPol", "IState"]
            for bit in bits:
                i_prop, rate = divmod(bit, 32)
                frags.append(
                    f"(prop:{props[i_prop]} stat:i*j*sin(dist_{rate})/(dist_{rate})"
                )
                diameters.append(np.nan)
        elif method == "3DGETAWAY":
            props = ["Unit", "Mass", "VdW", "ENeg", "Pol", "IonPol", "IState"]
            calcs_h = ["H", "HAT"]
            calcs_r = ["R", "Rmax"]
            for bit in bits:
                if bit < 4:
                    frags.append(
                        ["(stat:ITH)", "(stat:ISH)", "(stat:HIC)", "(stat:HGM)"][bit]
                    )
                    diameters.append(np.nan)
                elif bit < 144:
                    i_calcprop, dist = divmod(bit - 4, 10)
                    i_prop, i_calc = divmod(i_calcprop, 2)
                    frags.append(
                        f"(prop:{props[i_prop]} stat:{calcs_h[i_calc]} dist:{dist})"
                    )
                    diameters.append(dist)
                elif bit < 147:
                    frags.append(
                        ["(stat:RCON)", "(stat:RARS)", "(stat:REIG)"][bit - 144]
                    )
                    diameters.append(np.nan)
                else:
                    i_calcprop, dist = divmod(bit - 147, 9)
                    i_prop, i_calc = divmod(i_calcprop, 2)
                    frags.append(
                        f"(prop:{props[i_prop]} stat:{calcs_r[i_calc]} dist:{dist + 1})"
                    )
                    diameters.append(dist + 1)
        elif method == "3DDESC":
            props = [
                "Asphericity",
                "Eccentricity",
                "InertialShapeFactor",
                "NPR1",
                "NPR2",
                "PBF",
                "RadiusOfGyration",
                "SpherocityIndex",
            ]
            for bit in bits:
                frags.append(f"(prop:{props[bit]})")
                diameters.append(np.nan)
        else:
            frags, diameters = zip(*[(None, np.nan) for bit in bits])
        fpbit_d.update(
            {
                f"{method}_{bit}": (frag, diameter)
                for bit, frag, diameter in zip(bits, frags, diameters)
            }
        )
    frags, diameters = zip(*[fpbit_d.get(fpbit) for fpbit in fpbits])
    return frags, diameters


def fps_summary(X_fp, prop_df=None):
    """
    Get bit summary information of fingerprint matrix.

    Parameters
    ----------
    X_fp : sparse.csr_matrix[uint8] or 2darray[uint8]
        Fingerprint matrix of molecules.
    prop_df : DataFrame, optional
        Property dataframe to summarize bit effects by statistical
        difference between nonzero-bit and zero-bit. Index are NAME of
        SMILES.

    Returns
    -------
    bitinfo_df : DataFrame of shape (:, 3)
        Information table of fingerprint bits. With index as
        bit id, and 4 columns [COUNT, NONZERO, MAX_ORDER, EVEN_ORDER].
        COUNT : int
            Total counts of fingerprint bit.
        NONZERO : float
            Proportions of existence in molecules of fingerprint
            bit.
        MAX_ORDER : int
            Maximum count of fingerprint bit in a molecule.
        EVEN_ORDER : float
            Proportion of non-zero even count of fingerprint bit.
        if prop_df exists, the following columns are added for each
        property {PROP}:
        BITCOEF_{PROP} : float
            Difference of property value between zero-bit and
            nonzero-bit groups.
        BITSIZE_{PROP} : float
            Equivalent sample size when comparing group difference.
    sim_nonzero : 1d-array[float32]
        Nonzero distribution mapping of each row on all fingerprints.
    """
    info_d = {}
    if type(X_fp) == sparse.csr_matrix:
        bits, idx, nonzeros = np.unique(
            X_fp.indices, return_inverse=True, return_counts=True
        )
        idx_bits = np.argsort(X_fp.indices)
        idx_sample = nonzeros.cumsum()[:-1]
        X_fp_split = np.split(X_fp.data[idx_bits], idx_sample)
        counts = np.array([val.sum() for val in X_fp_split])
        max_orders = np.array([val.max() for val in X_fp_split])
        even_orders = 1 - np.array([(val % 2).sum() for val in X_fp_split]) / nonzeros
        if prop_df is not None:
            props_valid = prop_df.notna().values
            props = (prop_df - prop_df.mean()).fillna(0).values.astype(np.float32)
            props -= props.mean(axis=0)
            coefs = []
            samples = []
            for i_sample in np.split(np.repeat(np.arange(X_fp.shape[0]), np.diff(X_fp.indptr))[idx_bits], idx_sample):
                coefs.append(props[i_sample].sum(axis=0))
                samples.append(props_valid[i_sample].sum(axis=0))
            coefs = np.vstack(coefs)
            samples = np.vstack(samples)
            samples = 1 / (1 / samples + 1 / (props_valid.sum(axis=0) - samples))
            coefs /= (samples + 1e-8)
            for i, col in enumerate(prop_df.columns):
                info_d[f"BITCOEF_{col}"] = coefs[:, i]
                info_d[f"BITSIZE_{col}"] = samples[:, i]
        del X_fp_split
        nonzeros = nonzeros / X_fp.shape[0]
        sim_nonzero = np.array(
            [
                val.sum()
                for val in np.split(X_fp.data * nonzeros[idx], X_fp.indptr[1:-1])
            ]
        ) / (
            np.array([val.sum() for val in np.split(X_fp.data, X_fp.indptr[1:-1])])
            + 1e-16
        )
    elif type(X_fp) == np.ndarray:
        if len(X_fp.shape) == 3:
            X_fp = X_fp.reshape(-1, X_fp.shape[2])
        if np.issubdtype(X_fp.dtype, np.floating):
            X_fp = np.abs(X_fp)
        valids = X_fp.any(axis=0)
        X_fp = X_fp[:, valids]
        bits = np.nonzero(valids)[0]
        counts = X_fp.sum(axis=0)
        nonzeros = (X_fp != 0).sum(axis=0)
        max_orders = X_fp.max(axis=0)
        even_orders = ((X_fp != 0) & (X_fp % 2 < 1)).sum(axis=0) / (nonzeros + 1e-16)
        nonzeros = nonzeros / X_fp.shape[0]
        sim_nonzero = (X_fp @ nonzeros) / (X_fp.sum(axis=1) + 1e-16)
    bitinfo_df = pd.DataFrame({
        "COUNT": counts,
        "NONZERO": nonzeros,
        "MAX_ORDER": max_orders,
        "EVEN_ORDER": even_orders,
    }, index=bits)
    for key, val in info_d.items():
        bitinfo_df[key] = val
    return bitinfo_df, sim_nonzero


def smi_frag_from_circular(mol, i_atom_center, radius, atom_rings=[]):
    """
    Get SMILES representation of Morgan fingerprint fragments from a
    molecule based on a center atom and radius.

    Parameters
    ----------
    mol : Mol
        Input molecule.
    i_atom_center : int
        Index of center atom.
    radius : int
        Radius from center atom of Morgan fingerprint.
    atom_rings : List[Tuple[int, ...]], optional
        Ring information from mol.GetRingInfo().AtomRings(). Every
        tuple is a ring with atom indices.

    Returns
    -------
    smi_frag : str
        SMILES representation of fragments.
    """
    atom_d = {i_atom_center: 0}
    bitPath = Chem.FindAtomEnvironmentOfRadiusN(mol, radius + 1, i_atom_center)
    for b in bitPath:
        bond = mol.GetBondWithIdx(b)
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        dist1 = atom_d.get(atom1)
        dist2 = atom_d.get(atom2)
        if dist1 is None:
            if dist2 == radius:
                break
            else:
                atom_d[atom1] = dist2 + 1
        elif dist2 is None:
            if dist1 == radius:
                break
            else:
                atom_d[atom2] = dist1 + 1

    # set the coordinates of the submol based on the coordinates of the original molecule
    amap = {}
    submol = Chem.PathToSubmol(mol, bitPath, atomMap=amap)
    i_atom_subcenter = amap.get(i_atom_center)
    if i_atom_subcenter is not None:
        boundary_atoms = set()
        for i_atom_mol, i_atom_submol in amap.items():
            if i_atom_mol not in atom_d:
                atom = submol.GetAtomWithIdx(i_atom_submol)
                atom.SetAtomicNum(0)
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)
                boundary_atoms.add(i_atom_submol)
        edit_set = set()
        for ring in atom_rings:
            vertex = tuple(boundary_atoms.intersection(set(amap.get(i) for i in ring)))
            if len(vertex) == 2 and (vertex not in edit_set):
                if not edit_set:
                    esubmol = Chem.EditableMol(submol)
                esubmol.AddBond(*vertex, Chem.rdchem.BondType.SINGLE)
                edit_set.add(vertex)
        if edit_set:
            submol = esubmol.GetMol()
        Chem.RemoveStereochemistry(submol)
        smi_frag = Chem.MolToSmiles(
            submol, rootedAtAtom=i_atom_subcenter, allBondsExplicit=True
        )
        return smi_frag
    else:
        return ""


def mol_pharmfp(mol, D=None, pharm_d=PHARM_FACTORY):
    """
    Generate pharmacophore fingerprint.

    Parameters
    ----------
    mol : Mol
        Input molecule
    D : 2d-array of shape (n_atom, n_atom)
        Distance matrix of molecule.
    pharm_d : Dict[str, List[Tuple[int]]] or MolChemicalFeatureFactory
        If a dict, used as pre-calculated pharmacophore information of
        input molecule. Keys are pharmacophore groups, values are list
        of atom index of each group.
        Or feature factory used as pharmacophore extraction.

    Returns
    -------
    fp : 1d-array[uint8] of length 168
        3D pharmacophore fingerprint (fast version).

    Examples
    --------
    >>> smi = "CN(c1ncccc1CNc1nc(Nc2ccc(CCl)cc2)ncc1C(F)(F)F)S(C)(=O)=O"
    >>> mol = Chem.MolFromSmiles(smi)
    >>> fp = mol_pharmfp(mol)
    >>> fp.shape
    (168,)
    >>> fp[:6]
    array([5, 2, 5, 2, 0, 0], dtype=uint8)
    """
    if not mol:
        return np.zeros(168, dtype=np.uint8)
    fp = [0] * 168
    if D is None:
        try:
            D = AllChem.GetMoleculeBoundsMatrix(mol)
            D = (D + D.T) / 2
        except RuntimeError:
            D = AllChem.GetDistanceMatrix(mol)
    if type(pharm_d) != dict:
        pharm_d = mol_pharm(mol, D, pharm_d)
    D_bin = np.digitize(D, [2, 3, 5, 8, 13])
    pharms = []
    for i, pharm in enumerate(sorted(PHARMS)):
        for atoms in pharm_d[pharm]:
            if len(atoms) == 1:
                pharms.append((i, atoms[0]))
            else:
                pharms.append((i, list(atoms)))
    for i, (i_pharm, i_atoms) in enumerate(pharms):
        d_bin = D_bin[:, i_atoms]
        for j_pharm, j_atoms in pharms[(i + 1):]:
            d = d_bin[j_atoms]
            if type(d) == np.ndarray:
                d = d.min()
            bit = 6 * ((i_pharm * (13 - i_pharm)) // 2 + j_pharm) + int(d)
            fp[bit] += 1
    fp = np.minimum(fp, 255).astype(np.uint8)
    if not fp[:6].any():
        fp[:6] = [len(pharm_d.get(pharm, [])) for pharm in PHARMS[:6]]
    return fp


def mol_pharm(mol, D=None, factory=PHARM_FACTORY):
    """
    Get pharmacophore atom groups for given molecule.

    Parameters
    ----------
    mol : Mol
        Input molecule
    D : 2d-array of shape (n_atom, n_atom)
        Distance matrix of molecule.
    factory : MolChemicalFeatureFactory
        Feature factory used as pharmacophore extraction.

    Returns
    -------
    pharm_d : Dict[str, List[Tuple[int]]]
        Pharmacophore information of input molecule. Keys are
        pharmacophore groups, values are list of atom index of each
        group.

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("c1ccccc1CN")
    >>> mol_pharm(mol)
    {'HYD': [(5,), (0, 1, 2, 3, 4, 5)],
     'AR': [(0, 1, 2, 3, 4, 5)],
     'ACC': [],
     'DON': [(7,)],
     'CAT': [(7,)],
     'ANI': [],
     '@CTR': [(0, 1, 2, 3, 4, 5, 6, 7)]}
    """
    pharm_d = {pharm: [] for pharm in PHARMS}
    if mol:
        for i in range(factory.GetNumMolFeatures(mol)):
            feat = factory.GetMolFeature(mol, i, recompute=(i == 0))
            pharm_d[feat.GetFamily()].append(feat.GetAtomIds())
        if not pharm_d.get("@CTR"):
            if D is None:
                n_atom = mol.GetNumAtoms()
                pharm_d["@CTR"].append(tuple(range(n_atom)))
            else:
                pharm_d["@CTR"].append((int(D.sum(axis=0).argmin()),))
    return pharm_d


def get_pharm_mol(mol, pharm_d=None):
    """
    Get pharmacophore pseudo molecule with 3D conformation.

    Parameters
    ----------
    mol : Mol
        Input molecule with conformations.
    pharm_d : Dict[str, List[Tuple[int]]]
        Pharmacophore information of input molecule. Keys are
        pharmacophore groups, values are list of atom index of each
        group.

    Returns
    -------
    mol_p : Mol
        Pharmacophore pseudo molecule with 3D conformations.
    """
    if pharm_d is None:
        mol_noHs = Chem.RemoveHs(mol, sanitize=False)
        pharm_d = mol_pharm(mol_noHs)
    mol_p = Chem.MolFromSmiles(
        ".".join([".".join(PHARMATOMS[PHARMS.index(pharm)] * len(val)) for pharm, val in pharm_d.items() if val]), sanitize=False
    )
    confs = mol.GetConformers()
    AllChem.EmbedMultipleConfs(
        mol_p,
        len(confs),
        ignoreSmoothingFailures=True,
        enforceChirality=False,
        useExpTorsionAnglePrefs=False,
        useBasicKnowledge=False,
    )
    confs_p = mol_p.GetConformers()
    for i, (conf, conf_p) in enumerate(zip(confs, confs_p)):
        X_coord = conf.GetPositions()
        i_atom = 0
        for pharm, val in pharm_d.items():
            if not val:
                continue
            for atoms in val:
                conf_p.SetAtomPosition(i_atom, Point3D(*X_coord[list(atoms)].mean(axis=0)))
                i_atom += 1
    return mol_p


def mol_hit_fpbits(mol, fpbits):
    """
    Highlight fingerprint bits

    Parameters
    ----------
    mol : Mol
        Input molecule.
    fpbits : List[str]
        Fingerprint bits with each element has format "{METHOD}_{bit}".

    Returns
    -------
    hit_counts : 1d-array[uint8] of length len(fpbits)
        Counts of occurrence of each fingerprint bit on molecule.
    hit_atoms : 2d-array[uint8] of shape (n_atoms, len(fpbits))
        Counts of occurence of each fingerprint bit on each atom.
    hit_bonds : 2d-array[uint8] of shape (n_bonds, len(fpbits))
        Counts of occurence of each fingerprint bit on each bond.
    """
    fpbit_d = fpbits_to_method(fpbits, keep_index=True)
    hit_counts = np.zeros(len(fpbits), dtype=np.uint8)
    hit_atoms = np.zeros((mol.GetNumAtoms(), len(fpbits)), dtype=np.uint8)
    hit_bonds = np.zeros((mol.GetNumBonds(), len(fpbits)), dtype=np.uint8)
    if "MORGAN" in fpbit_d:
        bitinfo_d = {}
        AllChem.GetMorganFingerprint(mol, 2, bitInfo=bitinfo_d)
        for bit, j in fpbit_d["MORGAN"]:
            vals = bitinfo_d.get(bit)
            if not vals:
                continue
            for val in vals:
                hit_atoms[val[0], j] += 1
                hit_bonds[
                    list(Chem.FindAtomEnvironmentOfRadiusN(mol, val[1] + 1, val[0])), j
                ] += 1
            hit_counts[j] += len(vals)
    if "TOPO" in fpbit_d:
        bitinfo_d = {}
        Chem.rdmolops.UnfoldedRDKFingerprintCountBased(mol, bitInfo=bitinfo_d)
        for bit, j in fpbit_d["TOPO"]:
            vals = bitinfo_d.get(bit)
            if not vals:
                continue
            for val in vals:
                hit_bonds[list(val), j] += 1
            hit_counts[j] += len(vals)
    if "ESTATE" in fpbit_d:
        atomvals = EState.AtomTypes.TypeAtoms(mol)
        for bit, j in fpbit_d["ESTATE"]:
            bitval = EState.AtomTypes._rawD[bit % len(EState.AtomTypes._rawD)][0]
            for i, vals in enumerate(atomvals):
                if bitval in vals:
                    hit_atoms[i, j] += 1
                    hit_counts[j] += 1
    if "PAIR" in fpbit_d:
        D = AllChem.GetDistanceMatrix(mol)
        atomcodes = np.array([Utils.GetAtomCode(atom) for atom in mol.GetAtoms()])
        codeMask = (1 << rdMolDescriptors.AtomPairsParameters.codeSize) - 1
        pathMask = (1 << Pairs.numPathBits) - 1
        for bit, j in fpbit_d["PAIR"]:
            bit_res = bit
            dist = bit_res & pathMask
            bit_res = bit_res >> Pairs.numPathBits
            code1 = bit_res & codeMask
            bit_res = bit_res >> rdMolDescriptors.AtomPairsParameters.codeSize
            code2 = bit_res & codeMask
            atom_begin = np.nonzero(atomcodes == code1)[0]
            atom_end = np.nonzero(atomcodes == code2)[0]
            idx_begin, idx_end = np.where(D[np.ix_(atom_begin, atom_end)] == dist)
            if code1 == code2:
                no_dup = idx_begin < idx_end
                idx_begin, idx_end = idx_begin[no_dup], idx_end[no_dup]
            hit_atoms[atom_begin[idx_begin], j] += 1
            hit_atoms[atom_end[idx_end], j] += 1
            hit_counts[j] += len(idx_begin)
    if "TORSION" in fpbit_d:
        D = AllChem.GetDistanceMatrix(mol)
        atomcodes = np.array([Utils.GetAtomCode(atom) for atom in mol.GetAtoms()])
        codeMask = (1 << rdMolDescriptors.AtomPairsParameters.codeSize) - 1
        size = 4
        for bit, j in fpbit_d["TORSION"]:
            bit_res = bit
            codes = []
            for i in range(size):
                if i == 0 or i == (size - 1):
                    sub = 1
                else:
                    sub = 2
                codes.append(sub + bit_res & codeMask)
                bit_res = bit_res >> rdMolDescriptors.AtomPairsParameters.codeSize
            atom_begin = np.nonzero(atomcodes == codes[0])[0]
            atom_end = np.nonzero(atomcodes == codes[-1])[0]
            idx_begin, idx_end = np.where(D[np.ix_(atom_begin, atom_end)] == (size - 1))
            pair_set = set(zip(atom_begin[idx_begin], atom_end[idx_end]))
            for path in Chem.FindAllPathsOfLengthN(mol, size, useBonds=False):
                if (path[0], path[-1]) in pair_set:
                    for i, code in zip(path[1:-1], codes[1:-1]):
                        if atomcodes[i] != code:
                            break
                    else:
                        hit_atoms[list(path), j] += 1
                        hit_counts[j] += 1
                        continue
                if (path[-1], path[0]) in pair_set:
                    for i, code in zip(path[1:-1], reversed(codes[1:-1])):
                        if atomcodes[i] != code:
                            break
                    else:
                        hit_atoms[list(path), j] += 1
                        hit_counts[j] += 1
    for method, (fp_func, fpmatch_func) in FP_MOLFUNC_D.items():
        if fpmatch_func and method in fpbit_d:
            for bit, j in fpbit_d[method]:
                vals = fpmatch_func(mol, bit)
                for val in vals:
                    hit_atoms[list(val), j] += 1
                hit_counts[j] += len(vals)
    return hit_counts, hit_atoms, hit_bonds


def mol_atomfeatures(mol):
    """
    Get Atom features

    Parameters
    ----------
    mol : Mol
        Input molecule.

    Returns
    -------
    K : 2d-array[uint8] of shape (HEAVY_ATOM, 5)
        Atom codes. Number of (atomic, principle quantum, outer electrons, Hs, degree)
    V : 2d-array[float32] of shape (HEAVY_ATOM, 60)
        Atom features.
        0 : all-1 vector
        1 ~ 52: Atom features
            1 ~ 14: symbol (B, C, N, O, S, F, Si, P, Cl, Br, I, H, *, other)
            15 ~ 21: degree (0, 1, 2, 3, 4, 5, 6)
            22 ~ 26: hybridization (SP, SP2, SP3, SP3D, SP3D2)
            27 ~ 33: implicit valence (0, 1, 2, 3, 4, 5, 6)
            34 ~ 36: formal charge (-1, 0, 1)
            37 ~ 42: ring size (3, 4, 5, 6, 7, 8)
            43: aromatic
            44 ~ 48: total number of Hs (0, 1, 2, 3, 4)
            49 : number of atoms
            50 ~ 52: chiral tag (clockwise, anti-clockwise, other)
        53 ~ 54: Gasteiger charge of (heavy atom, implicit hydrogens)
        55 ~ 56 : Crippen features (logP, MR)
        57 ~ 58 : Surface area (Labute solvent-accessible, topological polar)
        59 : Intrinsic electrotopological state
    """
    AllChem.ComputeGasteigerCharges(mol)
    K = np.zeros((mol.GetNumAtoms(), 5), dtype=np.uint8)
    V = np.zeros((len(K), 60), dtype=np.float32)
    V[0] = 1
    ptable = Chem.GetPeriodicTable()
    for i, atom in enumerate(mol.GetAtoms()):
        anum = atom.GetAtomicNum()
        K[i] = [
            anum,
            EState.GetPrincipleQuantumNumber(anum),
            ptable.GetNOuterElecs(anum),
            atom.GetTotalNumHs(True),
            atom.GetDegree(),
        ]
        V[i, 1:53] = rdMolDescriptors.GetAtomFeatures(mol, i, True)
        V[i, 53] = atom.GetProp("_GasteigerCharge")
        V[i, 54] = atom.GetProp("_GasteigerHCharge")
    V[:, 55:57] = np.array(rdMolDescriptors._CalcCrippenContribs(mol), dtype=np.float32)
    V[:, 57] = list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])
    V[:, 58] = rdMolDescriptors._CalcTPSAContribs(mol, includeSandP=True)
    V[:, 59] = (4 / K[:, 1] ** 2 * (K[:, 2] - K[:, 3]) + 1) / K[:, 4]
    V[~np.isfinite(V)] = 0
    return K, V


def tree_map(X, d=128, random_state=0):
    """
    Dimensional reduction using minimum spanning trees.

    Parameters
    ----------
    X : sparse.csr_matrix or 2d-array or List[1d-array]
        Fingerprint matrix of molecules.
    d : int
        The number of permutations used for hashing.
    random_state : int
        The seed used for the random number generator.

    Returns
    -------
    x : 1d-array[float32] of length len(X)
        Embedding X coordinates of molecules.
    y : 1d-array[float32] of length len(X)
        Embedding Y coordinates of molecules.
    parents : 1d-array[int] of length len(X)
        Parent molecules using breadth-first-search from first molecule.
    gprop.degrees : 1d-array[int] of length len(X)
        Degrees of molecules in network.
    """
    begin_time = time.time()
    encoder = tmap.Minhash(d, seed=random_state)
    lf = tmap.LSHForest(d)
    if type(X) == sparse.csr_matrix:
        X = encoder.batch_from_sparse_binary_array([bits for bits, counts in split_sparse_fp(X)])
    elif type(X) == np.ndarray:
        X = encoder.batch_from_weight_array(X)
    else:
        X = encoder.batch_from_sparse_binary_array(X)
    lf.batch_add(X)
    lf.index()
    parents = np.full(len(X), -1, dtype=int)
    x, y, e1, e2, gprop = tmap.layout_from_lsh_forest(lf)
    G = nx.from_edgelist(zip(e1, e2))
    for i, j in nx.bfs_edges(G, 0):
        parents[j] = i
    logging.info(f"Finish TMAP embedding of size {len(X)} * {d}, time: {time.time() - begin_time:.1f} s")
    return x, y, parents, gprop.degrees


if __name__ == "__main__":
    import doctest

    doctest.testmod()
