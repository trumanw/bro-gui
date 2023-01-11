# -*- coding: utf-8 -*-
import bisect
import collections
import csv
import hashlib
import io
import itertools
import logging
import math
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from PIL import Image
from IPython.display import SVG
# from pyarrow.csv import read_csv
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Crippen,
    Draw,
    FilterCatalog,
    QED,
    rdCoordGen,
    rdMolDescriptors,
    rdqueries,
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.FilterCatalog import FilterCatalogParams

from . import vectools
# import vectools

warnings.filterwarnings("ignore")

RE_ATOMSQ = r"\[\S+?\]"
RE_ATOMHV = r"\[\S+?\]|[ABCONSPFQWIabcnops]"
RE_ATOM = r"\[\S+?\]|[ABCONSPFQWIabcnops*]"
RE_BOND = r"[-=#:/\\~]"
RE_SITE = r"\[\d*\*:?\d*\]|\*"
RE_RINGNUM = r"\d|%\d\d"


def smi_atom_ahead(smi, site="*", kekulize=False, repair=5):
    """
    Get SMILES which put specific atom symbol ahead. Each atom only put
    ahead once.

    Parameters
    ----------
    smi : str
        Input SMILES.
    site : str or List[str]
        The atom SMARTS to put ahead in SMILES.
    kekulize : bool
        Whether kekulize output SMILES.
    repair : int
        The maximum times trying to repair invalid SMILES.

    Returns
    -------
    smiles : List[str]
        Converted SMILES with specific atom symbol put ahead.

    See also
    --------
    smi_site_behind
        Rotate substituent site to the end of SMILES.

    Examples
    --------
    >>> smi_atom_ahead("c1nc(N[*:3])nc(N[*:2])c1[*:1]", "[*:3]")
    ['[*:3]Nc1ncc([*:1])c(N[*:2])n1']
    >>> smi_atom_ahead("c1nc(N[*:3])nc(N[*:2])c1[*:1]", "n")
    ['n1cc([*:1])c(N[*:2])nc1N[*:3]', 'n1c(N[*:3])ncc([*:1])c1N[*:2]']
    """
    mol = smi_to_mol(smi, kekulize=kekulize, repair=repair)
    if mol:
        if type(site) == str:
            site = [site]
        try:
            smiles = [Chem.MolToSmiles(mol, rootedAtAtom=i, kekuleSmiles=kekulize) for i, i_atom in enumerate(mol.GetAtoms()) if i_atom.GetSmarts() in site]
        except RuntimeError:
            logging.exception(f"Moving atom ahead error for {smi}")
            return []
    else:
        smiles = []
    return smiles


def smi_site_behind(smi, site, keep_first=False, sep=""):
    """
    Rotate substituent site to the end of SMILES.

    Parameters
    ----------
    smi : str
        Input SMILES.
    site : str
        The substituent site to put behind, like "*" or "[*:1]".
    keep_first : bool
        Whether keep the site at the beginning unchanged.
    sep : str
        A seperator to indicate the branch with site moved behind.

    Returns
    -------
    smi : str
        Converted SMILES with specific site put behind.

    Examples
    --------
    >>> smi_site_behind("c1nc(N[*:3])nc(N[*:2])c1[*:1]", "[*:3]")
    'c1nc(nc(N[*:2])c1[*:1])N[*:3]'
    >>> smi_site_behind("*c1cccc(CC2=NNC(=O)c3ccccc23)c1", "*")
    'c1(cccc(CC2=NNC(=O)c3ccccc23)c1)*'
    >>> smi_site_behind("[*:1]S(=O)(=O)c1ccc2c(c1)OCCO2", "[*:1]")
    'S(=O)(=O)(c1ccc2c(c1)OCCO2)[*:1]'
    """
    smi, end_level = smi.rstrip(")"), len(smi) - len(smi.rstrip(")"))
    if smi.endswith(site):
        pass
    elif (not keep_first) and smi.startswith(site):
        # If smi started with site, find the site-connected atom.
        match = re.search(r"[A-Za-z][\d(]*", smi)
        if match.group().endswith("("):
            # If site-connected atom has multiple branches
            count_para = 1
            for pos_end in range(match.end(), len(smi)):
                char = smi[pos_end]
                if char == ")":
                    count_para -= 1
                    if count_para == 0 and smi[pos_end + 1] != "(":
                        # If the branches of site-connected atom ended
                        break
                elif char == "(":
                    count_para += 1
            smi = f"{smi[match.start():(pos_end + 1)]}({smi[(pos_end + 1):]}){smi[len(site):match.start()]}{site}"
        else:
            smi = f"{match.group()}({smi[match.end():]}){smi[len(site):match.start()]}{site}"
    elif site + ")" in smi:
        ## pos_end: find site branch
        pos_end = len(smi) - smi[::-1].find(")" + site[::-1])
        ## iteration for branch (parathesis) level of site (with end level controlled)
        n_para = smi[pos_end:].count(")") - smi[pos_end:].count("(")
        for k_para in range(1 + n_para, 0, -1):
            i_para = 1
            ## pos_begin: find core
            for pos_begin in range(pos_end - 2, -1, -1):
                char = smi[pos_begin]
                if char == "(":
                    i_para -= 1
                    if i_para == 0:
                        break
                elif char == ")":
                    i_para += 1
            ## pos_end2: find fixed branch
            pos_end2 = pos_end
            while pos_end2 < len(smi) and smi[pos_end2] == "(":
                i_para = 0
                for pos in range(pos_end2, len(smi)):
                    char = smi[pos]
                    if char == ")":
                        i_para -= 1
                        if i_para == 0:
                            break
                    elif char == "(":
                        i_para += 1
                pos_end2 = pos + 1
            i_para = 1
            ## pos_end3: find tail branch to switch
            for pos_end3 in range(pos_end2, len(smi)):
                char = smi[pos_end3]
                if char == ")":
                    i_para -= 1
                    if i_para == 0:
                        break
                elif char == "(":
                    i_para += 1
            else:
                pos_end3 = len(smi)
            if smi[pos_end2:pos_end3].count(")") != smi[pos_end2:pos_end3].count("("):
                break
            ## find unclosed ring in core
            smi_begin = smi[:pos_begin]
            smi_sitebranch = smi[(pos_begin + 1):(pos_end - 1)]
            smi_tailbranch = smi[pos_end2:pos_end3]
            compile_str = re.compile(
                r"(?<!\[)(?<!\[\d)(?<!\[\d\d)("
                + RE_RINGNUM
                + r")(?!\+{0,2}\])(?!\-{1,2}\])"
            )
            ring_core_d = collections.Counter(re.findall(compile_str, smi_begin))
            ring_core_begins = set(
                ring for ring, val in ring_core_d.items() if val % 2 == 1
            )
            ring_tail_d = collections.Counter(re.findall(compile_str, smi_tailbranch))
            rings_convert = [
                ring
                for ring, val in ring_tail_d.items()
                if val % 2 == 0 and ring in ring_core_begins
            ]
            if rings_convert:
                #                print("rings: ", rings_convert)
                rings_sub = list(
                    set("987654321")
                    .difference(ring_tail_d)
                    .difference(ring_core_begins)
                )
                for ring in rings_convert:
                    ring_num = 10
                    if rings_sub:
                        smi_tailbranch = re.sub(
                            r"(?<!\[){}(?![+-]*\])".format(ring),
                            rings_sub.pop(),
                            smi_tailbranch,
                        )
                    else:
                        smi_tailbranch = re.sub(
                            r"(?<!\[){}(?![+-]*\])".format(ring),
                            f"%{ring_num}",
                            smi_tailbranch,
                        )
                        ring_num += 1
            ## pos_branch: find and adjust joint atom of core
            pos_branch = pos_begin - 1
            pos_chiral = smi_begin.find("@")
            if pos_chiral > 0:
                while smi_begin[pos_branch] == ")":
                    i_para = 1
                    for pos in range(pos_branch - 1, pos_chiral, -1):
                        char = smi[pos]
                        if char == "(":
                            i_para -= 1
                            if i_para == 0:
                                pos_branch = pos - 1
                                break
                        elif char == ")":
                            i_para += 1
                    else:
                        pos_branch = 0
                else:
                    if smi_begin[pos_branch] in "0123456789%":
                        pos_branch = (
                            len(smi_begin[:pos_branch].rstrip("0123456789%")) - 1
                        )
                ## Adjust chirality of joint atom
                if smi_begin[pos_branch] == "]":
                    pos_searchchiral = max(pos_branch - 4, 0)
                    pos_chiral = smi_begin[pos_searchchiral:].find("@")
                    if pos_chiral >= 0:
                        pos_chiral += pos_searchchiral
                        #                    print("find chircal: ", smi_begin[:pos_chiral], smi_begin[pos_chiral:])
                        if smi_begin[pos_chiral + 1] == "@":
                            smi_begin = (
                                f"{smi_begin[:pos_chiral]}{smi_begin[pos_chiral + 1:]}"
                            )
                        else:
                            smi_begin = (
                                f"{smi_begin[:pos_chiral]}@{smi_begin[pos_chiral:]}"
                            )
            smi = f"{smi_begin}({smi_tailbranch}){smi[pos_end:pos_end2]}{sep}{smi_sitebranch}{smi[pos_end3:]}"
            #            print(f"level: {k_para}\n{smi}\njoint: {smi_begin[:pos_branch]}\ncore : {smi_begin}\nsite branch: {smi_sitebranch}\nfixed branch: {smi[pos_end:pos_end2]}\ntail branch: {smi_tailbranch}")
            if smi.endswith(site):
                break
            else:
                pos_end = len(smi) - smi[::-1].index(")" + site[::-1])
    smi = smi + ")" * end_level
    return smi


class RingRenumber(object):
    """
    Renumber ring using SMILES regex operation.

    Arguments
    ---------
    ring_d_ : Dict[str, int]
        Saved ring map, keys are old ring, values are new ring.
    new_rings_ : List[int]
        Available list for taking a new ring from.
    last_pos_ : int
        Last closed ring position, lest using repeated ring number for
        a new ring at the same atom.
    last_ring_ : int
        Last closed ring index in `new_rings_`.

    Examples
    --------
    >>> import re
    >>> renumberer = RingRenumber()
    >>> compile_str = re.compile(r"\[[\w+-@*:\d]+\]|[()1-9]|\%\d\d")
    >>> compile_str.sub(renumberer.match, "C1CCC12CC(C3CC3)CC24CC4C")
    'C1CCC12CC(C1CC1)CC21CC1C'
    """

    def __init__(self):
        self.reset()

    def reset(self, min_num=1, max_num=100):
        self.ring_d_ = {}
        self.new_rings_ = list(range(min_num, max_num))
        self.last_pos_ = 0
        self.last_ring_ = 0
        self.used_rings_ = set()

    def match(self, match):
        ring = match.group()
        if ring[0] not in "%123456789":
            return ring
        ring_new = self.ring_d_.pop(ring, None)
        if not ring_new:
            if match.start() == self.last_pos_ and self.last_ring_ == 0:
                ring_new = self.new_rings_.pop(1)
            else:
                ring_new = self.new_rings_.pop(0)
            self.ring_d_[ring] = ring_new
        else:
            self.last_pos_ = match.end()
            bisect.insort(self.new_rings_, ring_new)
            self.last_ring_ = self.new_rings_.index(ring_new)
        self.used_rings_.add(ring_new)
        if ring_new > 9:
            return f"%{ring_new}"
        else:
            return str(ring_new)


def smi_decouple_ring(smi, joint="*.*"):
    """
    Decouple unfused ring systems of SMILES with joint notation.

    Parameters
    ----------
    smi : str
        Input SMILES.
    joint : None or str
        The separator used to join decoupled ring systems. If None,
        return a list instead.

    Returns
    -------
    seqs : str or List[str]
        Decoupled ring systems.

    Examples
    --------
    >>> smi_decouple_ring("*c1cccc(CC2=NNC(=O)c3ccccc23)c1", joint="*.*")
    '*c1cccc(c1)*.*CC1=NNC(=O)c2ccccc12'
    >>> smi_decouple_ring("Cn1ccnc1[C@]1(CNC(=O)c2cc3cc(Cl)ccc3o2)NC(=O)NC1=O", joint="")
    'Cn1ccnc1[C@@]1(NC(=O)NC1=O)CNC(=O)c1cc2cc(Cl)ccc2o1'
    >>> smi_decouple_ring("OC[C@]12C3N(Cc4ccccc4)C4[C@](CO)(C5N(Cc6ccccc6)C1[C@]3(CO)[C@@H](c1ccccc1)[C@]54CO)[C@@H]2c1ccccc1", joint="")
    'OC[C@]12C3N(C4[C@@](CO)([C@@H]2c2ccccc2)C2N(C1[C@]3(CO)[C@@H](c1ccccc1)[C@]24CO)Cc1ccccc1)Cc1ccccc1'
    """
    begin_seqs = []
    end_seqs = []
    compile_str = re.compile(RE_ATOMSQ + r"|[()]|" + RE_RINGNUM)
    for i_try in range(100):
        ring_level = -1
        end_level = len(smi) - len(smi.rstrip(")"))
        level = 0
        ring_d = {}
        rings_end = []
        rings_branch_end = []
        levels = [0]
        for match in compile_str.finditer(smi):
            #            print(match, match.endpos)
            #            print(f"ring_d: {ring_d}, rings_end: {rings_end}, rings_branch_end: {rings_branch_end}, levels: {levels}")
            #            print(f"ring_level: {ring_level}, level: {level}, end_level: {level}")
            if match.group() == "(":
                ## enter a higher parathesis level, and record begin position of level.
                level += 1
                levels.append(match.end())
            elif match.group() == ")":
                pos_begin = levels.pop()
                if level < ring_level:
                    ## Reduce ring level to the parathesis branch
                    ring_level = level
                elif (
                    rings_branch_end
                    and level == ring_level
                    and level not in ring_d.values()
                ):
                    ## Open and close a ring in the branch
                    if match.end() < match.endpos:
                        smi_out = smi_site_behind(
                            smi[:match.start()] + ">" + smi[match.start():], ">"
                        )
                        len_ring = match.start() - pos_begin + smi_out[::-1].index(">")
                        smi_out = smi_out.replace(">", "")
                    else:
                        smi_out = smi
                        len_ring = match.endpos - pos_begin
                        if smi_out.endswith(")"):
                            len_ring += 1
                    end_seqs = [smi_out[-len_ring:]] + end_seqs
                    smi = smi_out[:-len_ring]
                    end_level = len(smi) - len(smi.rstrip(")"))
                    #                    print(f"Decouple end ring: {smi}\nBegin seq: {begin_seqs}\nEnd seq: {end_seqs}")
                    break
                elif rings_end and (not ring_d) and level == 1:
                    ## Open a ring in core and close in the branch
                    begin_seqs.append(smi[: match.end()])
                    smi = smi[match.end():]
                    #                    print(f"Decouple begin ring: {smi}\nBegin seq: {begin_seqs}\nEnd seq: {end_seqs}")
                    break
                level -= 1
            elif match.group().startswith("["):
                pass
            else:
                ring = match.group()
                if ring in ring_d:
                    ## If close a ring, find cumulative parathesis level of closed rings
                    if ring_level < 0:
                        ring_level = min(ring_d.pop(ring), level)
                    else:
                        ring_level = min(ring_d.pop(ring), level, ring_level)
                    rings_end.append(ring)
                    if level > end_level:
                        rings_branch_end.append(ring)
                    #                    print(f"ring_d: {ring_d}, rings_end: {rings_end}, levels: {levels}")
                    #                    print(f"ring_level: {ring_level}, level: {level}")
                    ## Cut an indepedent ring from core
                    if (
                        rings_end
                        and (not ring_d)
                        and level == 0
                        and (
                            match.end() == len(smi)
                            or smi[match.end()] not in "(1234567890%"
                        )
                    ):
                        begin_seqs.append(smi[:match.end()])
                        smi = smi[match.end():]
                        #                        print(f"Decouple begin ring: {smi}\nBegin seq: {begin_seqs}\nEnd seq: {end_seqs}")
                        break
                else:
                    ## If open a ring, record parathesis level of ring
                    ring_d[ring] = level

        else:
            begin_seqs.append(smi)
            smi = ""
            #            print(smi, begin_seqs, end_seqs)
            break
    seqs = joint.join([seq for seq in begin_seqs + end_seqs if seq])
    renumberer = RingRenumber()
    seqs = compile_str.sub(renumberer.match, seqs)
    return seqs


def smi_hybrid(smi, smi_ref):
    """
    """
    smiles_out = []
    frags = smi_decouple_ring(smi).split(".")
    frags_ref = smi_decouple_ring(smi_ref).split(".")
    for i in range(1, len(frags)):
        for i_ref in range(1, len(frags_ref)):
            smi_out = ".".join(frags[:i] + frags_ref[i_ref:]).replace("*.*", "")
            if smi_out.count("(") == smi_out.count(")"):
                smiles_out.append(smi_out)
                smiles_out.append(
                    ".".join(frags_ref[:i_ref] + frags[i:]).replace("*.*", "")
                )
    return smiles_out


def smi_close_ring(smi, min_size=5, max_size=6):
    """
    Close all possible non-aromatic rings satisfying size range from
    the atom ahead for input SMILES.

    Parameter
    ---------
    smi : str
        Input SMILES with the first atom non-aromatic and with at least
        1 valence free.
    min_size : int
        Minimum ring size allowed from the first atom.
    max_size : int
        Maximum ring size allowed from the first atom.

    Returns
    -------
    smiles_out : List[str]
        Enumerated SMILES with new ring closed from the first atom.

    Examples
    --------
    >>> smi_close_ring("CCC(CO)CCC")
    ['C%99CC(CO%99)CCC', 'C%99CC(CO)CC%99C', 'C%99CC(CO)CCC%99']
    """
    smiles_out = []
    compile_str = re.compile(RE_ATOM + r"|[()#]|" + RE_RINGNUM)
    dist = 0
    levels = []
    ring_d = {}
    for match in compile_str.finditer(smi):
        pattern = match.group()
        if pattern == "(":
            levels.append(dist)
        elif pattern == ")":
            if not levels:
                break
            dist = levels.pop()
        elif pattern == "#":
            if dist < max_size:
                pos = match.end()
                smi = smi[: (pos - 1)] + "=" + smi[pos:]
        elif pattern[-1].isdigit():
            if pattern in ring_d:
                dist = ring_d.pop(pattern) + 1
            else:
                ring_d[pattern] = dist
        else:
            dist += 1
            if (
                min_size <= dist <= max_size
                and pattern.isalpha()
                and pattern.isupper()
                and (pattern.strip("FQWI"))
            ):
                pos = match.end()
                if ")" in smi[:pos]:
                    smi_out = "%99".join(
                        [
                            smi[0],
                            smi[1:pos].lstrip(RE_BOND[1:-1]),
                            smi[pos:].lstrip(RE_BOND[1:-1]),
                        ]
                    )
                else:
                    smi_out = "%99".join(
                        [
                            smi[0].lower(),
                            re.sub(RE_BOND, "", smi[1:pos].lower()),
                            smi[pos:].lstrip(RE_BOND[1:-1]),
                        ]
                    )
                smiles_out.append(smi_out)
    return smiles_out


def count_branch_site(smi):
    """
    Count the number of sites `*` on the terminals of branch in SMILES.
    
    Parameters
    ----------
    smi : str
        Input SMILES to count terminal sites.
        
    Returns
    -------
    n_branch_site : int
        The number of terminal sites in SMILES.
    """
    n_branch_site = smi.count("*)") + smi.endswith("*")
    if smi.startswith("*") and len(smi) > 1 and not smi.startswith("*1"):
        n_branch_site += 1
    return n_branch_site

    
def smi_remove_site(smi):
    """
    Remove all substituent sites (*) from SMILES with order kept.

    Parameters
    ----------
    smi : str
        Input SMILES.

    Returns
    -------
    str
        SMILES with substituents removed.

    Examples
    --------
    >>> smi_remove_site("c1nc(N[*:3])nc(N[*:2])c1[*:1]")
    'c1nc(N)nc(N)c1'
    >>> smi_remove_site("*c1cccc(CC2=NNC(=O)c3ccccc23)c1")
    'c1cccc(CC2=NNC(=O)c3ccccc23)c1'
    """
    return ".".join(
        [
            sub.strip(RE_BOND[1:-1])
            for sub in re.sub(RE_SITE, "", smi).replace("()", "").split(".")
        ]
    )


def smi_replace_site(core, subs, site, permute=False):
    """
    Replace a substituent site of a core with substituents.

    Parameters
    ----------
    core : str
        Input SMILES of core structure, which contains the pattern of
        site. If both `core` and `site` contains ".", then `subs` are
        used as linkers within `core` with multiple site "*".
    subs : List[str]
        Input SMILES of substituents. Each substituent should contains
        "*", otherwise concatenate at first atom.
    site : str
        SMARTS of a substituent site of `core` like "*" or "[*:1]".
    permute : bool
        Whether permute all replacement if `subs` contains multiple "*".

    Returns
    -------
    smiles : List[str]
        Replaced SMILES.

    Examples
    --------
    >>> smi_replace_site("c1nc(N[*:3])nc(N[*:2])c1[*:1]", ["*CO", "*c1ccccc1"], "[*:3]")
    ['c1nc(NCO)nc(N[*:2])c1[*:1]', 'c1nc(Nc2ccccc2*)nc(N[*:2])c1[*:1]']
    >>> smi_replace_site("c1nc(N[*:3])nc(N[*:2])c1[*:1]", ["*C(F)(F)F.*CO", "*NC1CCCNC1.*c1ccccc1"], "[*:1].[*:3]")
    ['c1nc(NCO)nc(N[*:2])c1C(F)(F)F', 'c1nc(Nc2ccccc2)nc(N[*:2])c1NC2CCCNC2']
    >>> smi_replace_site("[*:1]-C1Oc2ccc(NC(=O)C=C)cc2NC1=O.Oc1cc(Cl)ccc1-[*:2]", ["*CO*", "*c1ccccc1*"], "[*:1].[*:2]")
    ['Oc1cc(Cl)ccc1-OCC1Oc2ccc(NC(=O)C=C)cc2NC1=O', 'Oc1cc(Cl)ccc1-c1ccccc1C1Oc2ccc(NC(=O)C=C)cc2NC1=O']
    """
    subs_clean = []
    if permute:
        for sub in subs:
            if sub and (sub.count("*") > 1 or sub.find("*") > 0):
                subs_clean.extend(smi_atom_ahead(sub, site="*"))
            else:
                subs_clean.append(sub)
        subs_clean = list(set(subs_clean))
    else:
        for sub in subs:
            if sub and sub.find("*") > 0:
                subs_clean.append(next(iter(smi_atom_ahead(sub, site="*")), None))
            else:
                subs_clean.append(sub)
    if core.endswith(site):
        # move site to end (non-ring)
        core = core[: -len(site)]
        smiles = [
            core + sub.lstrip("*" + RE_BOND[1:-1]) if sub else None
            for sub in subs_clean
        ]
        return smiles
    smiles = []
    subsites = site.split(".")
    if "." in core:
        # If core has multiple parts, paste all subs with single part of core each step
        subcores = core.split(".")
        for sub in subs_clean:
            if not sub:
                smiles.append(None)
            elif "*" in sub[1:]:
                smiles.append(sub)
            else:
                smiles.append(None)
                logging.warning(f"At least two subsites needed to link {core}")
                # smiles.append(f"*{sub.lstrip('*')}*")
        for i, subcore in enumerate(subcores):
            for subsite in subsites:
                if subsite in subcore:
                    smiles = smi_replace_site(subcore, smiles, subsite, permute=permute)
    else:
        for subsite in subsites:
            loc = core.find(subsite)
            if loc == 0:
                core = smi_site_behind(core, subsite)
        renumberer = RingRenumber()
        compile_str = re.compile(RE_ATOMSQ + r"|[()]|" + RE_RINGNUM)
        compile_str.sub(renumberer.match, core)
        ring_num = max(renumberer.used_rings_, default=0) + 1
        for sub in subs_clean:
            if not sub:
                smiles.append(None)
            else:
                renumberer.reset(min_num=ring_num, max_num=90)
                smi = core
                sub = compile_str.sub(renumberer.match, sub)
                for subsite, side in zip(subsites, sub.split(".")):
                    side = side.lstrip("*" + RE_BOND[1:-1])
                    smi = smi.replace(subsite, side, 1)
                smiles.append(smi)
    return smiles


class SmilesCoreSub(object):
    """
    Rotate atoms (substituent sites) in SMILES to endpoint of sequence
    for concatenation. Mainly used for paste of core and substituents.

    Attributes
    ----------
    smi_ : str
        Rotated core with specific substituent site put at the end.
    site_ : str
        The substituent site at the end of core.
    smi_nosite_ : str
        Rotated core with all substituent sites removed.

    Examples
    --------
    >>> model_core = SmilesCoreSub()
    >>> model_core.fit("c1nc(N[*:3])nc(N[*:2])c1[*:1]", "[*:2]")
    ('c1nc(N)nc(c1)N', 'c1nc(N[*:3])nc(c1[*:1])N[*:2]')
    >>> model_core.transform(["c1nc(N)nc(c1)NCCCO", "c1nc(N)nc(c1)Nc1ccccc1"])
    ['c1nc(N[*:3])nc(c1[*:1])NCCCO', 'c1nc(N[*:3])nc(c1[*:1])Nc1ccccc1']
    """

    def __init__(self):
        pass

    def fit(self, smi, site="*"):
        """
        Rotate a SMILES representation with site moved to the end.

        Parameters
        ----------
        smi : str
            Input SMILES.
        site : str
            The substituent site to put behind, like "*" or "[*:1]".

        Returns
        -------
        smi_nosite_ : str
            Rotated core with all substituent sites removed.
        smi_ : str
            Rotated core.
        """
        if not smi:
            return "", ""
        self.site_ = site
        self.smi_ = smi_site_behind(smi, site=site)
        self.smi_nosite_ = smi_remove_site(self.smi_)
        if self.smi_nosite_ is None:
            logging.error(f"Error when rotate site:\n{smi}\n{self.smi_}")
        return self.smi_nosite_, self.smi_

    def transform(self, smiles):
        """
        Recover other sites of the core of concatenated SMILES.

        Parameters
        ----------
        smiles : List[str]
            Input concatenated SMILES of a core and a substituent.

        Returns
        -------
        smiles_out : List[str]
            SMILES with core structure replaced.
        """
        core = self.smi_[: -len(self.site_)]
        len_core = len(self.smi_nosite_)
        smiles_out = [
            core + smi[len_core:] for smi in smiles if smi.startswith(self.smi_nosite_)
        ]
        return smiles_out


def smi_randomize(smi, n_gen=10, fix_first=0, kekulize=False, random_state=None):
    """
    Enumerate SMILES with order randomization.

    Parameters
    ----------
    smi : str
        Input SMILES.
    n_gen : int
        The number of SMILES to generate.
    fix_first : int
        The number of first atoms to keep order as fixed.
    kekulize : bool
        Whether kekulize output SMILES.
    random_state : int
        Random seed of randomizing SMILES.

    Returns
    -------
    List[str]
        Enumerated SMILES.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> sorted(smi_randomize("*c1cccc(CC2=NNC(=O)c3ccccc23)c1", 5, random_state=0))
    ['C(c1cc(*)ccc1)c1n[nH]c(=O)c2c1cccc2', 'c1(Cc2c3ccccc3c(=O)[nH]n2)cc(*)ccc1', 'c1c(*)cccc1Cc1n[nH]c(=O)c2c1cccc2', 'c1ccc2c(=O)[nH]nc(Cc3cc(*)ccc3)c2c1', 'c1cccc2c(=O)[nH]nc(Cc3cccc(*)c3)c12']
    >>> sorted(smi_randomize("*c1cccc(CC2=NNC(=O)c3ccccc23)c1", 5, fix_first=1, random_state=0))
    ['*c1cc(Cc2c3c(cccc3)c(=O)[nH]n2)ccc1', '*c1cc(Cc2n[nH]c(=O)c3c2cccc3)ccc1', '*c1cc(Cc2n[nH]c(=O)c3ccccc32)ccc1', '*c1cccc(Cc2c3ccccc3c(=O)[nH]n2)c1', '*c1cccc(Cc2n[nH]c(=O)c3ccccc23)c1']
    """
    smiles_out = set()
    mol = smi_to_mol(smi, kekulize=kekulize)
    if mol:
        idx_fix = list(range(fix_first))
        np.random.seed(random_state)
        idxs = np.random.rand(2 * n_gen, mol.GetNumAtoms() - fix_first).argsort(axis=1)
        idxs += fix_first
        for idx in idxs.tolist():
            mol_out = Chem.RenumberAtoms(mol, newOrder=idx_fix + idx)
            smi_out = Chem.MolToSmiles(mol_out, canonical=False, kekuleSmiles=kekulize)
            smiles_out.add(smi_out)
            if len(smiles_out) >= n_gen:
                break
    return list(smiles_out)


def smi_permute(smi, n_gen=10):
    """
    Enumerate SMILES with restricted permutation of first atom.

    Parameters
    ----------
    smi : str
        Input SMILES.
    n_gen : int
        The number of SMILES to generate. If negative, permute all
        atoms ahead.

    Returns
    -------
    smiles : List[str]
        Enumerated SMILES.

    Examples
    --------
    >>> smi_permute("*c1cccc(CC2=NNC(=O)c3ccccc23)c1", 5)
    ['*c1cccc(Cc2n[nH]c(=O)c3ccccc23)c1',\
 'c1ccc(*)cc1Cc1n[nH]c(=O)c2ccccc12',\
 '[nH]1nc(Cc2cccc(*)c2)c2ccccc2c1=O',\
 'c1cccc2c(Cc3cccc(*)c3)n[nH]c(=O)c12',\
 'c1c(*)cccc1Cc1n[nH]c(=O)c2ccccc12']
    """
    smiles_out = []
    mol = smi_to_mol(smi)
    if mol:
        n_atom = mol.GetNumAtoms()
        if 0 <= n_gen < n_atom:
            iters = np.linspace(0, n_atom - 1, n_gen, dtype=int)
        else:
            iters = range(n_atom)
        try:
            smiles_out = [Chem.MolToSmiles(mol, rootedAtAtom=int(i)) for i in iters]
        except RuntimeError:
            logging.warning(f"SMILES enumeration failed on {smi}", exc_info=True)
    return smiles_out


def smi_to_subs(smi):
    """
    """
    smiles = set()
    mol = smi_to_mol(smi)
    if mol:
        try:
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetNumImplicitHs() > 0:
                    smiles.add("*" + Chem.MolToSmiles(mol, rootedAtAtom=i))
        except RuntimeError:
            logging.warning(f"SMILES substitution construction failed on {smi}", exc_info=True)
    return list(smiles)


def mol_permute_mask(mol, clear_bond=True):
    """
    Mask each atom of molecules by replacing atom with dummy site.

    Parameters
    ----------
    mol : Mol
        Input molecule.
    clear_bond : bool
        Whether clear bond type information of masked atom.

    Returns
    -------
    mols_out : List[str]
        Masked SMILES with index as atom index replaced.

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("COc1c[nH]cc1")
    >>> [Chem.MolToSmiles(mol_mask, canonical=False) for mol_mask in mol_permute_mask(mol)]
    ['*~Oc1c[nH]cc1', 'C~*~c1c[nH]cc1', 'CO~*1~c[nH]cc~1', 'COc1~*~[nH]cc1', 'COc1c~*~cc1', 'COc1c[nH]~*~c1', 'COc1~*~c[nH]c1']
    """
    mols_out = []
    for i in range(mol.GetNumAtoms()):
        mol_out = Chem.Mol(mol)
        atom = mol_out.GetAtomWithIdx(i)
        atom.SetAtomicNum(0)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
        atom.SetNumRadicalElectrons(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetHybridization(Chem.HybridizationType.UNSPECIFIED)
        atom.SetIsotope(1)
        if clear_bond:
            for bond in atom.GetBonds():
                bond.SetBondType(Chem.BondType.ZERO)
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                bond.SetIsAromatic(0)
        mols_out.append(mol_out)
    return mols_out


class SmilesEnumerator(object):
    """
    Enumerate SMILES representation of a molecule.

    Parameters
    ----------
    n_gen : int
        The number of SMILES to generate from each molecule.
    decouple_joint : str, optional
        If not None, use decouple-ring algorithm for SMILES conversion,
        and use the string to join splitted fragments. Usually use:
            None: Do not use decouple-ring algorithm for SMILES.
            "": Get ring-decoupled SMILES.
            ".": Get ring-splitted fragments.
            "*.*": Add sites to ring-splitted fragments.

    References
    ----------
    * Esben Jannik Bjerrum
      SMILES Enumeration as Data Augmentation for Neural Network
      Modeling of Molecules.
      https://arxiv.org/abs/1703.07076

    * Josep Arús-Pous, Simon Johansson, Oleksii Prykhodko, Esben Jannik
      Bjerrum, Christian Tyrchan, Jean-Louis Reymond, Hongming Chen Ola
      Engkvist
      Randomized SMILES Strings Improve the Quality of Molecular
      Generative Models.
      https://chemrxiv.org/articles/Randomized_SMILES_Strings_Improve_the_Quality_of_Molecular_Generative_Models/8639942

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> model_enum = SmilesEnumerator(n_gen=5)
    >>> model_enum.smi_enumerate("COc1c[nH]cc1")
    ['COc1cc[nH]c1', 'O(C)c1cc[nH]c1', 'c1[nH]ccc1OC', '[nH]1ccc(OC)c1', 'c1c[nH]cc1OC']
    >>> model_enum.transform(["c1cc(C)ccc1O", "Cc1c(N)cccc1CC"])
    ['c1cc(C)ccc1O',\
 'c1cc(O)ccc1C',\
 'Cc1ccc(O)cc1',\
 'c1cc(C)ccc1O',\
 'Oc1ccc(C)cc1',\
 'Cc1c(N)cccc1CC',\
 'c1(N)cccc(CC)c1C',\
 'c1ccc(CC)c(C)c1N',\
 'c1ccc(N)c(C)c1CC',\
 'CCc1cccc(N)c1C']
    """

    def __init__(self, n_gen=10, decouple_joint=""):
        self.n_gen_ = n_gen
        self.decouple_joint_ = decouple_joint

    def transform(self, smiles):
        """
        Enumerate SMILES.

        Parameters
        ----------
        smiles : List[str]
            Input SMILES.

        Returns
        -------
        smiles_enum : List[str]
            Enumerated SMILES.
        """
        smiles_enum = [
            smi_enum for smi in smiles for smi_enum in self.smi_enumerate(smi)
        ]
        return smiles_enum

    def smi_enumerate(self, smi):
        """
        Enumerate SMILES.

        Parameters
        ----------
        smi : str
            Input SMILES.

        Returns
        -------
        smiles : List[str]
            Enumerated SMILES.
        """
        n_site = smi.count("*")
        if self.n_gen_ < 1:
            smiles = [smi]
        elif n_site == 0:
            smiles = smi_permute(smi, self.n_gen_)
        elif n_site == 1:
            smiles = smi_randomize(smi, self.n_gen_, 1)
        else:
            smiles = smi_atom_ahead(smi)
        if self.decouple_joint_ is not None:
            smiles = [
                smi_decouple_ring(smi_enum, joint=self.decouple_joint_)
                for smi_enum in smiles
            ]
        return smiles


def smi_repair(smi: str) -> str:
    """
    Repair SMILES by removing unclosed ring and branch parenthesis.

    Parameters
    ----------
    smi : str
        Input SMILES.

    Returns
    -------
    str
        Repaired SMILES.

    Examples
    --------
    >>> smi_repair("CC12C(OC(Cl)CC1")
    'CC1COC(Cl)CC1'
    """
    ring_d = {}
    levels = []
    smi = smi.lstrip("(0123456789%" + RE_BOND[1:-1])
    for match in re.finditer(RE_ATOMSQ + r"|[()]|" + RE_RINGNUM, smi):
        pattern = match.group()
        if pattern == "(":
            levels.append(match.start())
        elif pattern == ")":
            if levels:
                levels.pop()
            else:
                pos = match.start()
                smi = smi[:pos] + "?" + smi[(pos + 1):]
        elif pattern[-1].isdigit():
            if pattern in ring_d:
                ring_d.pop(pattern)
            else:
                ring_d[pattern] = (match.start(), match.end())
    for pos in levels:
        smi = smi[:pos] + "?" + smi[(pos + 1):]
    for ring, (pos1, pos2) in ring_d.items():
        smi = smi[:pos1] + "?" * (pos2 - pos1) + smi[pos2:]
    return smi.replace("?", "").replace("()", "")


def smi_to_mol(smi, sanitize=True, kekulize=False, removeHs=True, removeIsotope=True, repair=0):
    """
    Convert SMILES to rdkit molecule.

    Parameters
    ----------
    smi : str or Mol
        Input SMILES.
    sanitize : bool or Chem.SanitizeFlags
        Whether sanitize molecule, or use rdkit sanitize options. If
        False, the molecule may be invalid.
    kekulize : bool
        Whether kekulize molecule.
    removeHs : bool
        Whether remove explicit hydrogens.
    removeIsotope : bool
        Whether remove isotope
    repair : int
        The maximum times trying to repair invalid SMILES.

    Returns
    -------
    mol : Mol
        Molecule from SMILES.

    Examples
    --------
    >>> mol = smi_to_mol("C1ccncCC.C1", sanitize=False)
    >>> Chem.MolToSmiles(mol, canonical=False)
    'C(ccncCC)C'
    >>> mol = smi_to_mol("C=C(=O)cccncC(Cl)Oc1cncc1", repair=5)
    >>> Chem.MolToSmiles(mol, canonical=False)
    'CC(=O)CCCNCC(Cl)OC1CNCC1'
    >>> mol = smi_to_mol("[H]c1sc([C@@]2([H])N([H])c3c([H])c([H])c(Br)c([H])c3[C@@]3([H])OC([H])([H])C([H])([H])[C@]32[H])c([H])c1[H]")
    >>> Chem.MolToSmiles(mol, canonical=False)
    'c1sc([C@H]2Nc3ccc(Br)cc3[C@H]3OCC[C@@H]23)cc1'
    """
    if not smi:
        return None
    elif type(smi) == str:
        smi = smi.partition(" ")[0]
        if smi.startswith("?"):
            mol = Chem.MolFromSmarts(smi.lstrip("?"))
        else:
            if removeHs:
                smi = smi.replace("[CH]", "C").replace("[C]", "C")
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if (not mol) and (repair > 0):
                mol = Chem.MolFromSmiles(smi_repair(smi), sanitize=False)
    elif type(smi) == Chem.Mol:
        mol = smi
    else:
        return None
    if not mol:
        return None
    elif removeIsotope:
        atoms = mol.GetAtomsMatchingQuery(rdqueries.IsotopeGreaterQueryAtom(0))
        if len(atoms) > 0:
            for atom in atoms:
                atom.SetIsotope(0)
    for i in range(1 + repair):
        try:
            if isinstance(sanitize, Chem.SanitizeFlags):
                Chem.SanitizeMol(mol, sanitize)
            elif sanitize:
                Chem.SanitizeMol(mol)
                Chem.SanitizeMol(mol)
            if kekulize:
                Chem.Kekulize(mol)
            if removeHs and mol:
                mol = Chem.RemoveHs(mol)
            return mol
        except ValueError as error:
            if repair > 0:
                msg = str(error)
                if "kekulize" in msg:
                    pass
                elif "non-ring atom" in msg:
                    for atom in mol.GetAtoms():
                        if atom.GetIsAromatic() and not atom.IsInRing():
                            atom.SetIsAromatic(False)
                            for b in atom.GetBonds():
                                if b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                                    b.SetBondType(Chem.rdchem.BondType.SINGLE)
                elif "valence" in msg:
                    atom = mol.GetAtomWithIdx(
                        int(msg.partition("# ")[-1].partition(" ")[0])
                    )
                    if atom.GetDegree() > 4:
                        return None
                    else:
                        for b in atom.GetBonds():
                            if b.GetBondType() != Chem.rdchem.BondType.SINGLE:
                                b.SetBondType(Chem.rdchem.BondType.SINGLE)
                                break
                        else:
                            if atom.GetAtomicNum() != 6:
                                atom.SetAtomicNum(6)
                                atom.SetNumRadicalElectrons(0)
                    # logging.debug(f"Repaired SMILES from {smi} to {Chem.MolToSmiles(mol, canonical=False)}")
                else:
                    return None
            else:
                break
        if not mol:
            break
    # logging.debug(f"Molecule convertion failed: {smi}", exc_info=True)
    return None


def canonize_smi(smi, sanitize=True, kekulize=False, repair=5, atom_ahead=-1):
    """
    Canonize SMILES.

    Parameters
    ----------
    smi : str
        Input SMILES.
    sanitize : bool or Chem.SanitizeFlags
        Whether sanitize molecule, or use rdkit sanitize options.
    kekulize : bool
        Whether kekulize output SMILES.
    repair : int
        The maximum times trying to repair invalid SMILES.
    atom_ahead : int
        The atom index to put ahead of SMILES.

    Returns
    -------
    str
        Canonized SMILES.

    References
    ----------
    * Daylight Chemical Information Systems, Inc.
      SMILES - A Simplified Chemical Language
      https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html

    Examples
    --------
    >>> canonize_smi("C=C(=O)cccncC(Cl)Oc1cncc1")
    'CC(=O)CCCNCC(Cl)OC1CCNC1'
    """
    mol = smi_to_mol(smi, sanitize=sanitize, kekulize=kekulize, repair=repair)
    if mol:
        return Chem.MolToSmiles(mol, kekuleSmiles=kekulize, rootedAtAtom=atom_ahead)


def mol_scaffold(mol):
    """
    Get Murcko scaffold of a molecule.

    A Murcko scaffold retains every ring system within the molecule.
    All non-ring systems are removed from the molecule unless the
    non-ring system is required to connect two ring systems together,
    any single-bond branches off the retained non-ring system(s) are
    removed also.

    Parameters
    ----------
    mol : Mol
        Input molecule.

    Returns
    -------
    smi_scaffold : str
        SMILES representation of Murcko scaffold.

    References
    ----------
    * Guy W. Bemis, Mark A. Murcko
      The Properties of Known Drugs. 1. Molecular Frameworks
      https://pubs.acs.org/doi/10.1021/jm9602928

    Examples
    --------
    >>> mol1 = Chem.MolFromSmiles("COc1cc(Cl)cc(C(=O)Nc2ccc(Cl)cn2)c1NC(=O)c1scc(CN(C)C2=NCCO2)c1Cl")
    >>> mol_scaffold(mol1)
    'O=C(Nc1ccccc1C(=O)Nc1ccccn1)c1cc(CNC2=NCCO2)cs1'
    >>> mol2 = Chem.MolFromSmiles("CCC(CO)CC")
    >>> mol_scaffold(mol2)
    ''
    """
    smi_scaffold = None
    if mol:
        try:
            mol_scaffold = Chem.MurckoDecompose(mol)
            smi_scaffold = Chem.MolToSmiles(mol_scaffold, isomericSmiles=False)
        except Exception:
            logging.warning(f"Scaffold failed of {Chem.MolToSmiles(mol)}")
    return smi_scaffold


def molform_elements(mol_formula):
    """
    Get count of element atoms from molecular formula.

    Parameters
    ----------
    mol_formula : str
        Molecular formula.

    Returns
    -------
    atom_freq_d : Dict[str, int]
        Counts of atoms by element.

    Examples
    --------
    >>> molform_elements("C2H6O")
    {'C': 2, 'H': 6, 'O': 1}
    """
    pattern = re.compile("[A-Z\+\-\*][a-z]*")
    atom_freq_d = {}
    for symbol, freq in zip(
        pattern.findall(mol_formula), pattern.split(mol_formula)[1:]
    ):
        if freq:
            atom_freq_d[symbol] = int(freq)
        else:
            atom_freq_d[symbol] = 1
    return atom_freq_d


def clean_from_smiles_info(
    smi_df,
    valid_elements={
        "H",
        "B",
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "+",
        "-",
        "*",
    },
):
    """
    Filter SMILES from its information table.

    Parameters
    ----------
    smi_df : DataFrame
        Information table of smiles with 2 columns [smi, mol_formula].
    valid_elements : Set[str]
        Valid elements for filtering.

    Returns
    -------
    smi_sr : Series
        Filtered SMILES with index in information table.
    """
    valid_elements = set(valid_elements)
    valid_element_sr = smi_df["FORMULA"].apply(
        lambda x: set(molform_elements(x)).issubset(valid_elements)
    )
    single_mol_sr = ~smi_df["SMILES"].str.contains("\.")
    smi_sr = smi_df.loc[valid_element_sr & single_mol_sr, "SMILES"].str.strip("\n ")
    logging.info(f"Cleaned / all SMILES: {len(smi_sr)} / {len(smi_df)}")
    return smi_sr


def mol_with_atom_index(mol):
    """
    Add atom index for showing a molecule.

    Parameters
    ----------
    mol : Mol
        The molecule.

    Returns
    -------
    mol : Mol
        The molecule with atom index added.
    """
    for i, i_atom in enumerate(mol.GetAtoms()):
        i_atom.SetProp("molAtomMapNumber", str(i))
    return mol


def smiles_plot(
    smiles,
    n_plot=15,
    n_column=3,
    size=(480, 320),
    atom_index=False,
    showHs=False,
    sanitize=Chem.SANITIZE_ADJUSTHS,
    legends="",
    highlight_atoms=None,
    highlight_bonds=None,
    weights=None,
    svg=True,
    macro_coord=False,
    max_n_atom=100,
):
    """
    Plot multiple SMILES on an grid image.

    Parameters
    ----------
    smiles : List[str]
        Input SMILES.
    n_plot : int
        The number of first SMILES to plot.
    n_column : int
        The number of columns when plotting.
    size : Tuple[int, int]
        Size of subplot for each molecule.
    atom_index : bool
        Whether show atom index in plot.
    showHs : bool
        Whether show hydrogens in plot.
    sanitize : bool or Chem.SanitizeFlags
        Whether sanitize molecule, or use rdkit sanitize options.
    legends : List[str], optional
        Annotation of SMILES plots. If not None, should be the same
        length with `smiles`.
    highlight_atoms : List[List[int]]
        The atom indices for plot to highlight. If highlight the first
        atom in SMILES, set `[0]`.
    highlight_bonds : List[List[int]] or str or Set[str]
        The bond indices for plot to highlight.
        If "aromatic", highlight all aromatic bonds in molecule.
        If a set of "MORGAN_*", highlight bonds within selected Morgan
        fingerprints.
    weights : List[List[float]] or Callable[Mol, List[float]]
        The weights for plot to highlight atom from red (lower) to
        green (higher).
        If a function, convert molecule into atom-wise masked value and
        overall value as the last element.
    svg : bool
        Whether use SVG or PNG format for plot.
    macro_coord : bool
        Whether use coord generation better for macrocycle (may disturb
        grid show).
    max_n_atom : int
        When use `weights` function to calculate weights, the maximum
        number of atoms of molecules allowed to have to calculate atom
        effects.

    Returns
    -------
    PIL Image
        Grid image of SMILES.
    """
    n_plot = min(len(smiles), n_plot)
    n_row = math.ceil(n_plot / n_column)
    if svg:
        d2d = Draw.MolDraw2DSVG(size[0] * n_column, size[1] * n_row, size[0], size[1])
    else:
        d2d = Draw.MolDraw2DCairo(size[0] * n_column, size[1] * n_row, size[0], size[1])
    mols = [None] * n_plot
    hl_atoms = [None] * n_plot
    hl_atomcolors = [{} for i in range(n_plot)]
    hl_atomrad = [{} for i in range(n_plot)]
    hl_bonds = [None] * n_plot
    hl_bondcolors = [{} for i in range(n_plot)]
    if not legends:
        legends = [""] * n_plot
    elif type(legends) == str:
        legends = [legends] * n_plot
    else:
        legends = legends[:n_plot]
    for i, smi in enumerate(smiles[:n_plot]):
        if type(smi) == str:
            mol = smi_to_mol(smi, sanitize=sanitize)
        elif type(smi) == Chem.Mol:
            mol = Chem.Mol(smi)
            if hasattr(smi, "__sssAtoms"):
                hl_atoms[i] = smi.__sssAtoms
        else:
            mol = None
        if not mol:
            continue
        if atom_index:
            mol = mol_with_atom_index(mol)
        if highlight_atoms:
            if type(highlight_atoms[0]) == int:
                hl_atoms[i] = highlight_atoms
            elif type(highlight_atoms[i]) == dict:
                atom_hl_d = collections.defaultdict(list)
                bond_hl_d = collections.defaultdict(list)
                A = np.tril(Chem.GetAdjacencyMatrix(mol))
                for rgb, vals in highlight_atoms[i].items():
                    atoms = set()
                    bonds = set()
                    for val in vals:
                        atoms.update(val)
                        if len(val) > 1:
                            for atom1, atom2 in zip(*A[np.ix_(val, val)].nonzero()):
                                bond = mol.GetBondBetweenAtoms(val[atom1], val[atom2])
                                bonds.add(bond.GetIdx())
                    atoms = {atom for val in vals for atom in val}
                    for atom in atoms:
                        atom_hl_d[atom].append(rgb)
                    for bond in bonds:
                        bond_hl_d[bond].append(rgb)
                hl_atomcolors[i] = hl_atoms[i] = dict(atom_hl_d)
                hl_bondcolors[i] = dict(bond_hl_d)
            else:
                hl_atoms[i] = highlight_atoms[i]
        if highlight_bonds == "aromatic":
            hl_bonds[i] = [
                i
                for i, bond in enumerate(mol.GetBonds())
                if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
            ]
            hl_bondcolors[i] = {bond: (1, 0, 1) for bond in hl_bonds[i]}
        elif highlight_bonds:
            if type(highlight_bonds) == set:
                count_d = find_morgan_bonds(mol, highlight_bonds)
                if count_d:
                    hl_bonds[i] = list(count_d)
                    hl_bondcolors[i] = {bond: (0.9, 0.9, 1 - 0.3 * min(count, 3)) for bond, count in count_d.items()}
            elif type(highlight_bonds[0]) == int:
                hl_bonds[i] = highlight_bonds
            else:
                hl_bonds[i] = highlight_bonds[i]
        i_row, i_column = divmod(i, n_column)
        n_atom = mol.GetNumAtoms()
        if weights:
            weight = None
            if hasattr(weights, "__call__"):
                if n_atom <= max_n_atom:
                    if sanitize != 1:
                        Chem.SanitizeMol(mol)
                    *weight, y_ref = weights(mol)
                    weight = [y_ref - y for y in weight]
                    legends[i] += f"{y_ref:.4g} "
            else:
                weight = weights[i]
            if weight:
                weight = np.array(weight)
                w_max, w_min = weight.max(), weight.min()
                w_ref = max(w_min, 0) + min(w_max, 0)
                legends[i] += f"({w_min:.3g} ~ {w_max:.3g})"
                if w_ref != 0:
                    legends[i] += f" [{w_ref:.3g}]"
                w_absmax = max(1e-8, abs(w_max - w_ref), abs(w_min - w_ref))
                weight_adj = (weight[:, np.newaxis] - w_ref) / w_absmax
                weight_sign = (weight[:, np.newaxis] >= 0).astype(int)
                weight_absadj = np.abs(weight_adj)
                if hl_atoms[i] is None:
                    atomcolors = 1 - weight_absadj * ([0, 2, 2] - weight_sign * [-2, 2, 0])
                    atomcolors = np.clip(atomcolors * np.minimum(1.5 - weight_absadj, 1), 0, 1)
                    atomcolors[np.isnan(atomcolors)] = 1
                    hl_atoms[i] = list(range(n_atom))
                    hl_atomcolors[i] = {atom: tuple(val) for atom, val in zip(hl_atoms[i], atomcolors.tolist())}
                else:
                    hl_atomcolors[i] = {atom: [(*rgb, 0.2 + 0.8 * weight_sign[atom, 0]) for rgb in val] for atom, val in hl_atoms[i].items()}
                    hl_atomrad[i] = {atom: float(0.25 + 0.3 * weight_absadj[atom, 0]) for atom in hl_atoms[i]}
        if hl_atoms[i] and not hl_atomrad[i]:
            hl_atomrad[i] = {atom: 0.4 for atom in hl_atoms[i]}
        try:
            mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True, forceCoords=True)
        except ValueError:
            continue
        if macro_coord:
            rdCoordGen.AddCoords(mol)
        mols[i] = mol
    try:
        if type(hl_atoms[0]) == dict:
            d2d.DrawMoleculeWithHighlights(mols[0], legends[0], hl_atomcolors[0], hl_bondcolors[0], hl_atomrad[0], {})
        else:
            d2d.DrawMolecules(
                mols,
                highlightAtoms=hl_atoms,
                highlightBonds=hl_bonds,
                highlightAtomColors=hl_atomcolors,
                highlightBondColors=hl_bondcolors,
                highlightAtomRadii=hl_atomrad,
                legends=legends,
            )
    except Exception:
        logging.exception("Unexpected plotting error")
        print(len(mols), hl_atoms, hl_bonds, hl_atomcolors, hl_bondcolors, hl_atomrad, len(legends))
    d2d.FinishDrawing()
    if svg:
        return SVG(d2d.GetDrawingText())
    else:
        return Image.open(Draw.BytesIO(d2d.GetDrawingText()))


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

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("COc1c[nH]cc1")
    >>> smi_frag_from_circular(mol, 1, 2)
    'O(-C)-c(:c:*):c:*'
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
                esubmol.AddBond(*vertex, Chem.rdchem.BondType.ZERO)
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


def find_morgan_bonds(mol, bits):
    count_d = collections.Counter()
    bitinfo_d = {}
    AllChem.GetMorganFingerprint(mol, 2, bitInfo=bitinfo_d)
    for bit, val in bitinfo_d.items():
        if f"MORGAN_{bit}" in bits:
            for atom, rad in val:
                for bond in Chem.FindAtomEnvironmentOfRadiusN(mol, rad + 1, atom):
                    count_d[bond] += 1
    return count_d


def sparse_concat(X, X_ref):
    """
    Concatenate two sparse matrix by same columns.

    Parameters
    ----------
    X, X_ref : sparse.csr_matrix with same columns
        Input sparse matrix

    Returns
    -------
    X_concat : sparse.csr_matrix
        Concatenated matrix.
    """
    X_concat = sparse.csr_matrix(
        (
            np.concatenate([X.data, X_ref.data]),
            np.concatenate([X.indices, X_ref.indices]),
            np.concatenate([X.indptr, X.indptr[-1] + X_ref.indptr[1:]]),
        ),
        shape=(X.shape[0] + X_ref.shape[0], X.shape[1]),
    )
    return X_concat


def sparse_weightsum(X, bits, weights, null_value=0, bitx=None, idx=None, bins=None):
    """
    Calculate weighted sum along specific columns for sparse matrix.

    Parameters
    ----------
    X : sparse.csr_matrix
        Sparse matrix
    bits : 1d-array[int]
        Column index containing weight information.
    weights : 1d-array[float]
        Weights for corresponding column index.
    null_value : float
        Default weight for unseen bits.
    bitx : 1d-array[int], optional
        Unique sorted bits in sparse matrix. Used for reduce repeated
        computation.
    idx : 1d-array[int], optional
        Inverse index of bitx. Used for reduce repeated computation.
    bins : 1d-array[float], optional
        If not None, output a 2d-array with bin-count statistics.

    Returns
    -------
    out : 1d-array[float]
        Weighted sum of X on bits with weights.
    """
    if bitx is None or idx is None:
        bitx, idx = np.unique(X.indices, return_inverse=True)
    idx_bit = np.searchsorted(bitx, bits)
    idx_bit[idx_bit >= len(bitx)] = len(bitx) - 1
    idx_revbit = np.argwhere(bitx[idx_bit] == bits).ravel()
    bit_weight = np.full(len(bitx), null_value, dtype=np.float32)
    bit_weight[idx_bit[idx_revbit]] = weights[idx_revbit]
    if bins is None:
        out = np.array(
            [(X.data[i_begin:i_end] * bit_weight[idx[i_begin:i_end]]).sum() for i_begin, i_end in zip(X.indptr[:-1], X.indptr[1:])],
            dtype=np.float32,
        )
    else:
        bit_weight = np.digitize(bit_weight, bins)
        out = np.zeros((X.shape[0], len(bins) + 1), dtype=int)
        for j in range(out.shape[1]):
            out[:, j] = [(X.data[i_begin:i_end] * (bit_weight[idx[i_begin:i_end]] == j)).sum() for i_begin, i_end in zip(X.indptr[:-1], X.indptr[1:])]
    return out


def get_dense(X, bits=1024):
    """
    Get dense array of Morgan fingerprint from sparse matrix.

    Parameters
    ----------
    X : sparse.csr_matrix of shape (n, 2 ** 32)
        Sparse fingerprint matrix
    bits : int or List[int]
        Set the length of compressed fingerprint. If a list, extract
        the bits from the sparse matrix.

    Returns
    -------
    X_dense : ndarray[uint8] of shape (n, bits)
        Compressed array of fingerprint.
    """
    if not sparse.isspmatrix_csr(X):
        X_dense = X[:, bits]
    elif type(bits) == int:
        X_dense = np.zeros((X.shape[0], bits), dtype=np.uint8)
        for i, (j_begin, j_end) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
            np.add.at(X_dense[i], X.indices[j_begin:j_end] % bits, X.data[j_begin:j_end])
    else:
        X_dense = np.zeros((X.shape[0], len(bits)), dtype=np.uint8)
        argidx = np.argsort(X.indices).astype(np.uint32)
        idx_sample = np.repeat(
            np.arange(X.shape[0], dtype=np.uint32), np.diff(X.indptr)
        )[argidx]
        idx_data = X.data[argidx]
        samples = np.searchsorted(X.indices[argidx], np.add.outer(bits, [0, 1]))
        for i, val in enumerate(samples):
            X_dense[idx_sample[slice(*val)], i] = idx_data[slice(*val)]
    return X_dense


def smi_to_charseq(smi):
    """
    Convert SMILES to a sequence of characters with 2-length element
    name abbreviated.

    Parameters
    ----------
    smi : str
        Input SMILES.

    Returns
    -------
    seq : str
        Abbreviated character sequence from SMILES.

    See also
    --------
    charseq_to_smi
        Recover SMILES from abbreviated character sequence.
    """
    return smi.replace("Cl", "Q").replace("Br", "W")


def charseq_to_smi(seq):
    """
    Recover SMILES from abbreviated character sequence.

    Parameters
    ----------
    seq : str
        Abbreviated character sequence.

    Returns
    -------
    smi : str
        SMILES from character sequence.

    See also
    --------
    smi_to_charseq
        Convert SMILES to a sequence of characters with 2-length
        element name abbreviated.
    """
    return re.sub(f"\({RE_BOND}*\)", "", seq.replace("Q", "Cl").replace("W", "Br"))


def mol_calc_fsp3ring(mol):
    ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
    n_carbon = 0
    n_sp3ring = 0
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 6:
            n_carbon += 1
            if (i in ring_atoms) and (atom.GetTotalDegree() == 4):
                n_sp3ring += 1
    if n_carbon == 0:
        return 0
    else:
        return n_sp3ring / n_carbon


def mol_calc_chiralc(mol):
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True)
    return rdMolDescriptors.CalcNumAtomStereoCenters(mol)


def set_hit_smarts(key, smarts, names=None, update=True):
    if update and (key in FILTER_D):
        filtercat = FILTER_D[key]
    else:
        filtercat = FilterCatalog.FilterCatalog()
        FILTER_D[key] = filtercat
    if names is None:
        names = smarts
    for name, sma in zip(names, smarts):
        submol = Chem.MolFromSmarts(sma)
        if submol:
            matcher = FilterCatalog.SmartsMatcher(name, submol)
            filtercat.AddEntry(FilterCatalog.FilterCatalogEntry("matcher", matcher))


FPWEIGHT_D = {}
"""
    SASCORE : Tuple[1darray[int], 1darray[float]]
        To calculate synthetic accessibility score from
        Morgan fingerprint.
    SCSCORE : List[ndarray[float]]
        To calculate synthetic complexity score from Morgan
        fingerprint.
    RASCORE : LGBMClassifier
        To calculate retrosynthetic accessibility score from Morgan,
        pharm and scaf fingerprint and relative properties.
    NPSCORE : Tuple[1darray[int], 1darray[float]]
        To calculate Natural Product-likeness Score from Morgan
        fingerprint.

"""

FILTER_D = {
    key: FilterCatalog.FilterCatalog(val)
    for key, val in FilterCatalogParams.FilterCatalogs.names.items()
}

MOLSTRFUNC_D = dict(
    FORMULA=rdMolDescriptors.CalcMolFormula, MURCKO_SCAFFOLD=mol_scaffold,
)
MOLNUMFUNC_D = dict(
    MW=rdMolDescriptors._CalcMolWt,
    LOGP=Crippen.MolLogP,
    HBD=rdMolDescriptors.CalcNumLipinskiHBD,
    HBA=rdMolDescriptors.CalcNumLipinskiHBA,
    RTB=rdMolDescriptors.CalcNumRotatableBonds,
    TPSA=lambda mol: rdMolDescriptors.CalcTPSA(mol, includeSandP=True),
    MR=Crippen.MolMR,
    FSP3=rdMolDescriptors.CalcFractionCSP3,
    FSP3RING=mol_calc_fsp3ring,
    CHIRALC=mol_calc_chiralc,
    HEAVY_ATOM=lambda mol: mol.GetNumAtoms(),
    BRIDGEHEAD=rdMolDescriptors.CalcNumBridgeheadAtoms,
    SPIRO=rdMolDescriptors.CalcNumSpiroAtoms,
    AR=AllChem.CalcNumAromaticRings,
    NRING=rdMolDescriptors.CalcNumRings,
    MAXRING=lambda mol: max(
        [len(ring) for ring in mol.GetRingInfo().AtomRings()], default=0
    ),
    QINDEX=lambda mol: 3
    + sum((atom.GetDegree() ** 2) / 2 - 2 for atom in mol.GetAtoms()),
    MORGANBIT=lambda mol: len(
        AllChem.GetMorganFingerprint(mol, 2).GetNonzeroElements()
    ),
    HIT_SMARTS_BRENK=lambda mol: len(FILTER_D["BRENK"].GetMatches(mol)),
)

COL_DOC_D = dict(
    FORMULA="Molecular formula",
    MURCKO_SCAFFOLD="Scaffold with non-ring side-chain removed",
    GENERIC_SCAFFOLD="Scaffold with atom-bond information removed from Murcko",
    SIMPLE_SCAFFOLD="Scaffold with non-junction atom simplified from generic",
    AS_ALL="Homologous series from RENOVA.Asteroid",
    AS_GENERIC_SCAFFOLD="Generic scaffold of homologous series from RENOVA.Asteroid",
    AS_SIMPLE_SCAFFOLD="Simplified scaffold of homologous series from RENOVA.Asteroid",
    SMILES_FROM="Source SMILES in generation step",
    MW="Molecule weight",
    LOGP="Log partition coefficient (solubility)",
    HBD="Num of Lipinski H-bond donors",
    HBA="Num of Lipinski H-bond acceptors",
    RTB="Num of rotatable bonds",
    TPSA="Topological polar surface area",
    MR="Molar refractivity (polarizability)",
    FSP3="Fraction of sp3 carbon atoms",
    FSP3RING="Fraction of sp3 carbon atoms in rings",
    CHIRALC="Num of chiral centers",
    HEAVY_ATOM="Num of heavy atoms",
    BRIDGEHEAD="Num of bridgehead atoms",
    SPIRO="Num of spiro atoms",
    AR="Num of aromatic rings",
    NRING="Num of rings",
    MAXRING="Maximum atom num of a ring",
    QINDEX="Normalized quadratic index [Balaban,1979]",
    MORGANBIT="Num of unique bits of Morgan fingerprint",
    QED="Quantitative Estimate of Drug-likeness [Bickerton,2012]",
    MCE18="Medicinal Chemistry Evolution-2018 (3D complexity) [Ivanenkov,2019]",
    SASCORE="Synthetic Accessibility Score [Ertl,2009]",
    SCSCORE="Synthetic Complexity Score [Coley,2018]",
    RASCORE="Retrosynthetic accessibility score (AiZynthFinder) [Thakkar, 2021]",
    RASTEP2="Probability of AiZynthFinder finding retrosynthetic route in 2 steps",
    NPSCORE="Natural Product-likeness Score [Ertl,2008]",
    MAX_ORDER="Maximum absolute value of fingerprint bit in a molecule",
    EVEN_ORDER="Proportion of integer parts as even number to all non-zero values",
    DIAMETER="Diameter distance of fingerprint",
    NBRANCH="Num of branch terminals with 1-degree connection",
)
COL_SIMSTAT_D = dict(
    MEAN="Mean similarity",
    MAX="Maximum similarity",
    MAX5="Similarity of 5th-nearest neighbors",
    REF="Similarity with reference molecule",
    REFDIFF="Similarity difference between reference molecule and other closest molecules",
)


def get_column_doc(col, libs=None):
    if not col:
        return ""
    if col in COL_DOC_D:
        return COL_DOC_D[col]
    out = ""
    pattern = col.split("_")
    if pattern[0] == "COUNTMOL":
        out = "Number of molecule contained in structure"
    elif pattern[0] == "COUNT":
        out = "Number of total occurrence"
    elif pattern[0] == "NONZERO":
        out = "Frequency of nonzero occurrence in molecules"
    if len(pattern) <= 1:
        return out
    elif out:
        out += f" {pattern[1]}"
    elif pattern[0] == "LIBSIM":
        out = f"Mean similarity on fingerprint {pattern[1]}"
    elif pattern[0] == "LIBMAP":
        out = f"Library mapping on fingerprint {pattern[1]}:"
        if pattern[2] in ["X", "Y"]:
            out += f"coordinate {pattern[2]}"
        elif pattern[2] == "PARENT":
            out += "parent id of BFS tree"
        elif pattern[2] == "DEGREE":
            out += "degree of nodes"
    elif pattern[0] == "HIT" and pattern[1] == "SMARTS":
        filtercat = FILTER_D.get("_".join(pattern[2:]))
        if filtercat:
            return f"Counts of SMARTS hits of {pattern[2]}({filtercat.GetNumEntries()} structures)"
        else:
            return f"Counts of SMARTS hits of {pattern[2]}"
    elif pattern[0] == "SIM":
        if len(pattern) == 3 and pattern[2] in COL_SIMSTAT_D:
            return f"{COL_SIMSTAT_D[pattern[2]]} in library at {pattern[1]}"
    elif pattern[0] == "PHARMSIM":
        if len(pattern) == 3 and pattern[2] in COL_SIMSTAT_D:
            return f"{COL_SIMSTAT_D[pattern[2]]} in pharmacophore molecules at {pattern[1]}"
    elif pattern[0] == "ACTIVE":
        return "Activity predicted value of " + pattern[1]
    elif pattern[0] == "QSAR":
        return "QSAR predicted value of " + pattern[1]
    elif pattern[0] == "FPCOORD":
        return "Fingerprint chemical space coordinate of " + pattern[1]
    elif pattern[0] == "COORD":
        return "Chemical space coordinate of " + pattern[1]
    if pattern[-1].startswith("lib"):
        if libs:
            out += f" in library {libs[int(pattern[-1][3:])-1]}"
        else:
            out += f" in library {pattern[-1][3:]}"
    return out


def update_core(core):
    core_init = core
    sites = smi_find_sites(core, pattern=RE_SITE)
    n_sites = len(sites)
    if n_sites <= 1:
        return core
    base_site = sites[0]
    if core.count(base_site) > 1:
        logging.warning(f"more than one site '{base_site}' found in {core}")
        logging.warning("We will ignore their order...")
        base_idx = 0
    else:
        base_idx = 1
        core = core.replace(base_site, "$$")
    core = re.sub(RE_SITE, "_", core).split("_")
    core_smi = core[0]
    for idx in range(1, len(core)):
        core_smi += f"[*:{idx+base_idx}]{core[idx]}"
    core_smi = core_smi.replace("$$", "[*:1]")
    logging.info(f"Update {core_init} to {core_smi}")
    return core_smi


def smi_find_sites(smi, pattern=RE_SITE):
    """
    Find all matched pattern from SMILES.

    Parameters
    ----------
    smi : str
        Input SMILES
    pattern : str
        Regex of pattern, default substituent sites.

    Returns
    -------
    List[str]
        Sorted patterns found in SMILES.
    """
    return sorted(re.findall(pattern, smi))


def smi_reassign_sites(smi, sites):
    """
    Reassign substituent sites from `*` to with named sites

    Parameters
    ----------
    smi : str
        Input SMILES
    sites : List[str]
        SMARTS of substituent sites to replace, such as "[*:1]",
        "[*:2]", "[*:3]". Can also be sidechains with first atom as
        connector, but be aware of ring number collision with original
        SMILES.

    Returns
    -------
    smiles_out : List[str]
        SMILES with substituent sites `*` reassigned.
    """
    smi = smi.replace("*", "[*]")
    smiles_out = []
    for sites_perm in itertools.permutations(sites):
        smi_out = smi
        for site in sites_perm:
            smi_out = smi_out.replace("[*]", site, 1)
        smiles_out.append(smi_out)
    return smiles_out


def smi_count_atoms(smi, pattern=RE_ATOMHV):
    """
    Count the number of pattern from SMILES.

    Parameter
    ----------
    smi : str
        Input SMILES
    pattern : str
        Regex of pattern to count, default heavy atoms.

    Returns
    -------
    int
        Number of patterns found in SMILES.
    """
    return len(re.findall(pattern, smi_to_charseq(smi)))


def smi_count_atomrings(smi):
    """
    Count the number of atom, rings and branches from SMILES.

    Parameters
    ----------
    smi : str
        Input SMILES

    Returns
    -------
    n_atom : int
        Number of atoms.
    n_ring : int
        Number of rings.
    n_branch : int
        Number of branches with 1-degree connection.
    """
    patterns = re.findall(RE_ATOM + "|\d\)?|%\d\d\)?", smi_to_charseq(smi) + ")")
    rings = [i for i, val in enumerate(patterns) if val.lstrip("%")[0].isdigit()]
    n_branch = smi.count(")") + 2
    if rings:
        n_branch = (
            n_branch
            - (rings[0] == 1)
            - len([i for i in rings if patterns[i].endswith(")")])
        )
    n_atom = len(patterns) - len(rings)
    n_ring = len(rings) >> 1
    return n_atom, n_ring, n_branch


def mol_filter_match(mol, filtercat):
    """
    Get filter match atoms for molecule

    Parameters
    ----------
    mol : Mol
        Input molecule.
    filtercat : FilterCatalog
        Filter defined by multiple SMARTS with names.

    Returns
    -------
    match_d : Dict[str, List[Tuple[int, ...]]]
        Matched atoms for patterns detected by filter.
        Keys are names of patterns, values are matched atoms of patterns.

    Examples
    --------
    >>> mol = Chem.MolFromSmiles('CN(c1ncccc1CNc1nc(Nc2ccc(CCl)cc2)ncc1C(F)(F)F)S(C)(=O)=O')
    >>> mol_filter_match(mol, FILTER_D["BRENK"])
    {'alkyl_halide': [(18, 19)]}
    """
    match_d = collections.defaultdict(list)
    for match in filtercat.GetFilterMatches(mol):
        match_d[match.filterMatch.GetName()].append(tuple(i_atom for i, i_atom in match.atomPairs))
    return dict(match_d)


def smiles_filter_plot(smiles, filtercat, n_plot=60, **kwargs):
    mols = []
    highlight_atoms = []
    legends = []
    for i, smi in enumerate(smiles):
        mol = smi_to_mol(smi)
        match_d = mol_filter_match(mol, filtercat)
        if match_d:
            mols.append(mol)
            highlight_atoms.append([atom for key, val in match_d.items() for atoms in val for atom in atoms])
            legends.append(", ".join([f"id: {i}"] + [f"{key}: {len(val)}" for key, val in match_d.items()]))
            if len(mols) >= n_plot:
                break
    plt = smiles_plot(mols, highlight_atoms=highlight_atoms, legends=legends, n_plot=n_plot, **kwargs)
    return plt


class SmilesRecorder(object):
    """
    Canonize, deduplicate, calculate molecule-wise properties with
    Morgan fingerprint analysis for SMILES.

    Parameters
    ----------
    sanitize : bool or Chem.SanitizeFlags
            Whether sanitize molecule, or use rdkit sanitize options.
    save_fp : bool
        Whether calculate and save Morgan fingerprint information.
    filepath : str
        Location to save into disk.
    keep_cols : str
        Which columns to keep in memory. If filepath is None, default
        all in `molstrfunc_d` and `molnumfunc_d`.
    molstrfunc_d : Dict[str, Callable[[Mol], str]]
        Use function operations on rdkit molecule to get string result.
        Keys are names, and values are rdkit molecule functions which
        return a string.
    molnumfunc_d : Dict[str, Callable[[Mol], float]]
        Use function operations on rdkit molecule to get numeric result.
        Keys are names, and values are rdkit molecule functions which
        return a float or integer.
    core : str, optional
        Whether concatenate input SMILES as substituents with core. It
        should include substituent site like "*", "[*:1]".
    repair : int
        The maximum times trying to repair invalid SMILES.

    Attributes
    ----------
    smi_d_ : Dict[str, int]
        Canonical SMILES with record indices. Keys are SMILES, and
        values are indices.
    strs_d_ : Dict[str, Dict[str, int]]
        Categories of string properties of molecules. Keys are property
        names, and values are property categories.
    nums_d_ : Dict[str, 1darray[float32]]
        Numeric properties or index of string properties of molecules.
        Keys are property names, and values are property records.
    model_fp_ : SmilesFingerprint
        Calculate and save Morgan fingerprint of recorded molecules.
    begin_size_ : int
        Number of molecules saved before recent fit / load_library step.
    smiles_raw_ : List[str]
        Input SMILES in fit step.
    idx_canon_ : 1darray[int32]
        Index of input SMILES on canonical SMILES records, where -1
        indicates invalid SMILES.
    errlogs_ : List[str]
        Error log when canonize input SMILES and do calculation.

    Examples
    --------
    >>> smiles = ["C1CNCC1", "c1ccccc1", "N1CCCC1", "C1CC", "CCCCC"]
    >>> smi_recorder = SmilesRecorder()
    >>> smi_df = smi_recorder.fit(smiles)
    >>> print(smi_df.to_string())
         SMILES FORMULA MURCKO_SCAFFOLD         MW    LOGP  HBD  HBA  RTB   TPSA         MR  FSP3  FSP3RING  CHIRALC  HEAVY_ATOM  BRIDGEHEAD  SPIRO  AR  NRING  MAXRING  QINDEX  MORGANBIT
    0   C1CCNC1   C4H9N         C1CCNC1  71.123001  0.3698    1    1    0  12.03  22.103701     1         1        0           5           0      0   0      1        5       3          8
    1  c1ccccc1    C6H6        c1ccccc1  78.113998  1.6866    0    0    0   0.00  26.441999     0         0        0           6           0      0   1      1        6       3          3
    3       CCC    C3H8                  44.097000  1.4163    0    0    0   0.00  15.965000     1         0        0           3           0      0   0      0        0       0          4
    4     CCCCC   C5H12                  72.151001  2.1965    0    0    2   0.00  25.198999     1         0        0           5           0      0   0      0        0       0          7
    >>> X_fp = smi_recorder.model_fp_.get_fps()
    >>> X_fp.shape
    (4, 4294967296)
    >>> bitinfo_df = smi_recorder.model_fp_.fps_summary(X_fp)
    >>> print(bitinfo_df.head().to_string())
                       COUNT  NONZERO  MAX_ORDER  EVEN_ORDER   SMILES_FRAG  DIAMETER          PARENT_ID
    FPBIT_ID
    MORGAN_98513984        6     0.25          6         1.0  c1:c:*~*:c:1         2  MORGAN_3218693969
    MORGAN_416356657       2     0.25          2         1.0  C1-C-*~*-N-1         2  MORGAN_2968968094
    MORGAN_725338437       2     0.25          2         1.0  C1-C-C-N-C-1         4  MORGAN_2142032900
    MORGAN_1173125914      2     0.25          2         1.0     C(-C)-C-*         2  MORGAN_2245384272
    MORGAN_1289643292      1     0.25          1         0.0  N1-C-*~*-C-1         2  MORGAN_2132511834
    """

    def __init__(
        self,
        sanitize=True,
        save_fp=True,
        filepath=None,
        keep_cols=True,
        molstrfunc_d=MOLSTRFUNC_D,
        molnumfunc_d=MOLNUMFUNC_D,
        core=None,
        verbose=True,
        repair=5,
    ):
        self.sanitize_ = sanitize
        if save_fp:
            self.model_fp_ = SmilesFingerprint(fp_rad=2, bitinfo_df=(save_fp == 1))
        else:
            self.model_fp_ = None
        self.molstrfunc_d_ = molstrfunc_d
        self.molnumfunc_d_ = molnumfunc_d
        self.molfunc_d_ = collections.ChainMap(molstrfunc_d, molnumfunc_d)
        self.repair_ = repair
        self.filepath_ = filepath
        self.set_core(core)
        self.reset(keep_cols=keep_cols)
        self.verbose_ = verbose

    def __len__(self):
        return len(self.smi_d_)

    def set_core(self, core):
        """
        Set core for converting input substituents.

        Parameters
        ----------
        core : str
            Input SMILES of core structure containing site.
        """
        if (type(core) == str) and ("*" in core):
            self.core_ = core
            sites = smi_find_sites(core)
            if "." in core:
                self.site_ = ".".join(sites)
            else:
                self.site_ = sites[0]
            logging.info(
                f"{len(sites)} sites found in {core}, site {self.site_} chosen"
            )
        else:
            self.core_ = ""
            self.site_ = ""

    def reset(self, n_init=1000000, reset_mol=True, reset_fitlog=False, keep_cols=True):
        """
        Clear memory of records.

        Parameters
        ----------
        n_init : int
            Initiated sample size to save data.
        reset_mol : bool
            Whether clear recorded molecules (SMILES) and related data.
        reset_fitlog : bool
            Whether clear recorded log of fitting.
        keep_cols : List or bool
            Which column to keep in memory. If True, save all. If
            False, save none.
        """
        if reset_mol:
            self.smi_d_ = {}
            self.strs_d_ = collections.defaultdict(dict)
            self.nums_d_ = collections.defaultdict(
                lambda: np.full(n_init, np.nan, dtype=np.float32)
            )
            self.vecs_d_ = collections.defaultdict(list)
            self.file_cols_ = list(self.molstrfunc_d_) + list(self.molnumfunc_d_)
            if self.model_fp_:
                self.model_fp_.reset()
            if hasattr(keep_cols, '__iter__'):
                for col in keep_cols:
                    if col in self.file_cols_:
                        self.keep_cols_.append(col)
                    else:
                        self.nums_d_[col]
            elif keep_cols:
                self.keep_cols_ = self.file_cols_.copy()
            else:
                self.keep_cols_ = []
            if self.filepath_:
                with open(self.filepath_, "w"):
                    pass
        if reset_fitlog:
            self.smiles_raw_ = []
            self.idx_canon_ = np.full(n_init, -1, dtype=np.int32)
            self.errlogs_ = []

    def add_strval(self, key, value):
        """
        Check category index of string value with new index added.

        Parameters
        ----------
        key : str
            Name of a string property.
        value : str or List[str]
            Value(s) of a string property.

        Returns
        -------
        int or 1darray[int]
            Index of values of the string property.
        """
        if value is None:
            return None
        elif value.__hash__:
            i_val = self.strs_d_[key].get(value)
            if i_val is None:
                i_val = len(self.strs_d_[key])
                self.strs_d_[key][value] = i_val
            return i_val
        else:
            idx_val = np.full(len(value) + 1, -1, dtype=np.int32)
            idx_cat, val_cat = pd.factorize(value)
            for i, val in enumerate(val_cat):
                i_val = self.strs_d_[key].get(val)
                if i_val is None:
                    i_val = len(self.strs_d_[key])
                    self.strs_d_[key][val] = i_val
                idx_val[i] = i_val
            return idx_val[idx_cat]

    def add_mol(self, mol):
        """
        Try to canonize and add a molecule to record. If existed, then
        do not update record.

        Parameters
        ----------
        mol : Mol
            The molecule.

        Returns
        -------
        smi : str
            Canonical SMILES of input molecule.
        info_d : Dict[str, Any]
            Information values calculated from input molecule.
        """
        try:
            sys.stderr = io.StringIO()
            smi = Chem.MolToSmiles(mol)
        except RuntimeError:
            return None, None
        if "." in smi:
            smi = sorted([(len(s), s) for s in smi.split(".")])[-1][-1]
            mol = smi_to_mol(smi, sanitize=self.sanitize_, repair=self.repair_)
        i_smi = self.smi_d_.get(smi)
        if i_smi is not None:
            info_d = None
        else:
            i_smi = len(self)
            self.smi_d_[smi] = i_smi
            info_d = {}
            if self.model_fp_:
                self.model_fp_.add_molfp(mol)
            for key, func in self.molfunc_d_.items():
                if mol.HasProp(key):
                    info_d[key] = mol.GetDoubleProp(key)
                    continue
                try:
                    info_d[key] = func(mol)
                except Exception:
                    logging.warning(
                        f"Function {key}: {func} failed on {smi}", exc_info=True
                    )
            for key in self.keep_cols_:
                if i_smi >= len(self.nums_d_[key]):
                    self.nums_d_[key] = np.r_[
                        self.nums_d_[key], np.full_like(self.nums_d_[key], np.nan)
                    ]
                val = info_d.get(key)
                if val is not None:
                    if key in self.molstrfunc_d_:
                        val = self.add_strval(key, val)
                    self.nums_d_[key][i_smi] = val
        return smi, info_d

    def fit(self, smiles, column_d={}, calc_fpscore=False, X_ref_d={}):
        """
        Get information table of given SMILES about canonical SMILES
        and properties. Also count the number of valid / unique SMILES.

        Parameters
        ----------
        smiles : List[str]
            Input SMILES
        column_d : Dict[str, List[Any]], optional (default={})
            Other columns used for appending after generating `smi_df`.
            Keys are column names, and values should have same length
            with input `smiles`.
        calc_fpscore : bool
            Whether calculate composite / fingerprint scores.
        X_ref_d : Dict[str, 2d-array[uint8] or sparse.csr_matrix[uint8]]
            Reference fingerprint for similarity calculation. Keys are
            names of reference, values are fingerprint matrices for
            similarity comparison.
        query : str
            A query expression used for conditional filter of output.

        Returns
        -------
        smi_df : DataFrame
            Information table of canonical SMILES.
        """
        begin_time = time.time()
        self.reset(len(smiles), reset_mol=False, reset_fitlog=True)
        self.smiles_raw_ = smiles
        self.begin_size_ = len(self)
        if not self.core_:
            pass
        elif self.core_ == self.site_:
            for i, smi in enumerate(smiles):
                if smi and (self.core_ not in smi):
                    smiles[i] = self.core_ + smi
            logging.info(f"Concatenate {len(smiles)} SMILES with site {self.core_}")
        else:
            smiles = smi_replace_site(self.core_, smiles, self.site_)
            logging.info(
                f"Concatenate {len(smiles)} SMILES with core {self.core_} and site {self.site_}"
            )
        if self.filepath_:
            if self.begin_size_ == 0:
                f = open(self.filepath_, "w")
                f.close()
            with open(self.filepath_, "a") as csvfile:
                if self.begin_size_ == 0:
                    self.file_cols_.extend(list(column_d))
                    writer = csv.DictWriter(
                        csvfile, fieldnames=["SMILES"] + self.file_cols_
                    )
                    writer.writeheader()
                else:
                    writer = csv.DictWriter(
                        csvfile, fieldnames=["SMILES"] + self.file_cols_
                    )
                    column_d = {col: val for col, val in column_d.items() if col in self.file_cols_}
                for i, smi_raw in enumerate(smiles):
                    mol = smi_to_mol(
                        smi_raw, sanitize=self.sanitize_, repair=self.repair_
                    )
                    if mol:
                        smi, info_d = self.add_mol(mol)
                        if info_d is not None:
                            info_d["SMILES"] = smi
                            for key, val in column_d.items():
                                info_d[key] = val[i]
                            writer.writerow(info_d)
                    else:
                        smi = None
                    if smi:
                        self.idx_canon_[i] = self.smi_d_[smi]
        else:
            for i, smi_raw in enumerate(smiles):
                #                sys.stderr = io.StringIO()
                mol = smi_to_mol(smi_raw, sanitize=self.sanitize_, repair=self.repair_)
                if mol:
                    smi, info_d = self.add_mol(mol)
                else:
                    smi = None
                #                self.errlogs_.append(sys.stderr.getvalue())
                if smi:
                    self.idx_canon_[i] = self.smi_d_[smi]
        idx_canon, idx_raw, idx_begin, quantile_canon = self.get_index()
        idx_canon = idx_canon[idx_begin[0]:]
        idx_raw = idx_raw[idx_begin[0]:]
        argidx_raw = idx_raw.argsort()
        self.update_record(idx_canon, column_d, idx_raw)
        if calc_fpscore or X_ref_d:
            smi_df = self.get_library().iloc[idx_canon[argidx_raw]]
            X_fp = self.model_fp_.get_fps(idx_canon)
            smi_df = fpscore_predict(smi_df, X_fp, X_ref_d=X_ref_d)
        else:
            smi_df = self.get_library(filepath=False).iloc[idx_canon[argidx_raw]]
        smi_df.index = idx_raw[argidx_raw]
        if self.verbose_:
            logging.info(
                f"Accumulated / updated / unique / valid / input molecules: {len(self)} / {len(self) - self.begin_size_} / {len(smi_df)} / {np.count_nonzero(self.idx_canon_ >= 0)} / {len(smiles)}, time: {time.time() - begin_time:.1f} s"
            )
            logging.info(
                "Quantile validity / uniqueness ratio: {:.4f} {:.4f} {:.4f} {:.4f} / {:.4f} {:.4f} {:.4f} {:.4f}".format(
                    *quantile_canon[1:].ravel()
                )
            )
        return smi_df

    def update_record(self, idx_canon, column_d, idx_raw=slice(None)):
        """
        Assign new / updated record values to recorder.

        Parameters
        ----------
        idx_canon : List[int]
            Index of canonical SMILES.
        column_d : Dict[str, List[Any]], optional (default={})
            Other columns used for appending into records. Keys are
            column names, and values should have same length with input
            `smiles`.
        idx_raw : List[int], optional
            Assign partial rows of `column_d` to records. Should have
            same length with idx_canon
        """
        for key, record in column_d.items():
            gap_size = len(self) - len(self.nums_d_[key])
            if gap_size > 0:
                self.nums_d_[key] = np.r_[
                    self.nums_d_[key], np.full(gap_size, np.nan, dtype=np.float32)
                ]
            try:
                record = np.asarray(record, dtype=np.float32)
            except ValueError:
                self.strs_d_[key]
            if key in self.strs_d_:
                idx_record = self.add_strval(key, record)
                self.nums_d_[key][idx_canon] = idx_record[idx_raw]
            else:
                self.nums_d_[key][idx_canon] = record[idx_raw]

    def get_index(self, begin_size=None, splits=4):
        """
        Get record index assigned of recent fit step, with also summary
        statistics of validity / uniqueness based on splits of input
        into multiple chunks.

        Parameters
        ----------
        begin_size : int, optional
            From which record index to begin. If None, use the position
            where last fit step begins from (i.e. updated records only).
        splits : int or List[int]
            Number of splits, or index of input SMILES to split for
            summary statistics.

        Returns
        -------
        idx_canon : 1darray[int]
            Unique record index of recent fit step.
        idx_raw : 1darray[int]
            First occurence index of input.
        quantile_canon : 2d-array[float] of shape (3, splits)
            Information of each quantile split of input. 3 rows
            indicates count, validity ratio and uniqueness ratio.
        """
        if begin_size is None:
            begin_size = self.begin_size_
        idx_canon, idx_raw = np.unique(self.idx_canon_, return_index=True)
        ## if contains invalid value -1, then remove.
        idx_begin = np.searchsorted(idx_canon, np.arange(len(self) + 1))
        uniques_canon = np.zeros(len(self.idx_canon_), dtype=bool)
        uniques_canon[idx_raw[idx_begin[begin_size]:]] = True
        quantile_canon = np.column_stack(
            [
                np.r_[len(idx), np.mean(idx, axis=0)]
                for idx in np.array_split(
                    np.c_[self.idx_canon_ >= 0, uniques_canon], splits
                )
            ]
        )
        return idx_canon, idx_raw, idx_begin, quantile_canon

    def get_library(self, filepath=True, calc_fpscore=False, X_ref_d={}):
        """
        Get information table of recorded molecules and properties.

        Parameters
        ----------
        filepath : bool or str
            If str, read from csv file.
            If True and filepath exist, read from csv file.
        calc_fpscore : bool
            Whether calculate composite / fingerprint scores.
        X_ref_d : Dict[str, 2d-array[uint8] or sparse.csr_matrix[uint8]]
            Reference fingerprint for similarity calculation. Keys are
            names of reference, values are fingerprint matrices for
            similarity comparison.

        Returns
        -------
        smi_df : DataFrame
            Information table of recorded molecules.
        """
        if not filepath or (filepath is True and not self.filepath_):
            file_cols = set()
        else:
            if filepath is True:
                filepath = self.filepath_
            # smi_file_df = read_csv(filepath).to_pandas()
            file_cols = set(smi_file_df.columns)
        record_d = {"SMILES": list(self.smi_d_)}
        for key, val in self.nums_d_.items():
            if key in file_cols:
                continue
            gap_size = len(self) - len(val)
            if gap_size > 0:
                val = np.r_[val, np.full(gap_size, np.nan, dtype=np.float32)]
            if key in self.strs_d_:
                record_d[key] = pd.Categorical.from_codes(
                    np.where(np.isnan(val[: len(self)]), -1, val[: len(self)]).astype(
                        np.int32
                    ),
                    list(self.strs_d_[key]),
                )
            else:
                record_d[key] = pd.to_numeric(val[: len(self)], downcast="integer")
        smi_df = pd.DataFrame(record_d)
        if file_cols:
            smi_df.set_index("SMILES", inplace=True)
            smi_df = smi_file_df.join(smi_df, on="SMILES")
        if calc_fpscore:
            try:
                X_fp = self.model_fp_.get_fps()
                smi_df = fpscore_predict(smi_df, X_fp, X_ref_d=X_ref_d)
            except Exception:
                logging.exception("Unexpected fingerprint score prediction error")
        return smi_df

    def get_fitlog(self):
        """
        Get detailed log table of fit step.

        Returns
        -------
        smi_fitlog_df : DataFrame of shape (:, 3)
            smi_raw : str
                Input SMILES.
            smi : str
                Canonized SMILES.
            errorlog : str
                Error log when canonize input SMILES and do calculation.
        """
        smi_fitlog_df = pd.DataFrame(
            {
                "SMILES_RAW": self.smiles_raw_,
                "SMILES": np.array(list(self.smi_d_) + [None])[self.idx_canon_],
                #                "ERRLOG": self.errlogs_,
            }
        )
        logging.info("Get fit log finished")
        return smi_fitlog_df

    def load_library(self, smi_df):
        """
        Load information table from a fitted library.

        Parameters
        ----------
        smi_df : DataFrame
            Information table of recorded molecules.

        Returns
        -------
        idx_canon : List[int] of length len(smi_df)
            Mapped index of records for each input.
        """
        self.begin_size_ = len(self)
        idx_add = []
        idx_canon = []
        for i, smi in enumerate(smi_df["SMILES"].values):
            i_canon = self.smi_d_.get(smi)
            if i_canon is None:
                i_canon = len(self)
                self.smi_d_[smi] = i_canon
                idx_add.append(i)
            idx_canon.append(i_canon)
        for key, values in smi_df.iloc[idx_add].iteritems():
            if key == "SMILES":
                continue
            elif key == "NAME":
                values = values.astype(str)
            gap_size = self.begin_size_ + len(idx_add) - len(self.nums_d_[key])
            if gap_size > 0:
                self.nums_d_[key] = np.r_[
                    self.nums_d_[key], np.full(gap_size, np.nan, dtype=np.float32)
                ]
            if (values.dtype == "O") or (key in self.strs_d_):
                self.nums_d_[key][
                    self.begin_size_:(self.begin_size_ + len(idx_add))
                ] = self.add_strval(key, values.values)
            else:
                self.nums_d_[key][
                    self.begin_size_:(self.begin_size_ + len(idx_add))
                ] = values.values
        return idx_canon


class SmilesFingerprint(object):
    """
    Calculate and summarize Morgan fingerprint from SMILES.

    Parameters
    ----------
    radius : int, positive
        The radius of Morgan Fingerprint from a center atom.
    bitinfo_df : bool or DataFrame
        Whether record bit information of Morgan fingerprint. If DataFrame,
        use as initial bit information.
    verbose : bool
        Controls the verbosity when transforming.

    Attributes
    ----------
    bit_d_ : Dict[int, str]
        Morgan fingerprint bit with SMILES representation of fragments.
        Keys are Morgan bits, and values are fragments.
    bit_parent_d_ : Dict[str, Tuple(int, str)]
        Get parent fingerprint which has same atom but 1 less radius.
        Keys are bits, and values are tuple of (diameter, parent bits).
    fps_ : List[1d-array[uint32]]
        Morgan fingerprint bit ID of molecules.
    fps_counts_ : List[1d-array[uint8]]
        Morgan fingerprint bit counts of molecules.
    fps_pharm_ : List[1d-array[uint8] of length 168]
        Pharmacophore fingerprint of molecules.

    Examples
    --------
    >>> from chemtools import SmilesFingerprint
    >>> smiles = ["c1ccccc1", "CCCCC", "C1CNCC1"]
    >>> model_fp = SmilesFingerprint()
    >>> X_fp = model_fp.transform(smiles)
    >>> X_fp.shape
    (3, 4294967296)
    """

    def __init__(self, fp_rad=2, bitinfo_df=True, verbose=True):
        self.fp_rad_ = fp_rad
        self.verbose_ = verbose
        self.reset(bitinfo_df)

    def reset(self, bitinfo_df=True, X_fp=None, update=False):
        """
        Reload Morgan fingerprint information.

        Parameters
        ----------
        bitinfo_df : bool or Dict[str, str]
            Whether record bit information of Morgan fingerprint. If a
            dict, use as initial bit information.
        X_fp : sparse.csr_matrix[uint8] of shape (:, 2 ** 32), optional
            Morgan fingerprint matrix of recorded molecules.
        update : bool
            Whether keep existed records.

        See also
        --------
        self.get_fps
            Summarize information table of Morgan fingerprint bits.
        """
        if X_fp is None:
            fps = []
            fps_count = []
            fps_pharm = []
            fps_scaf = []
        else:
            fps = np.split(X_fp.indices.astype(np.uint32), X_fp.indptr[1:-1])
            fps_count = np.split(X_fp.data, X_fp.indptr[1:-1])
            if hasattr(X_fp, "fp_pharm"):
                fps_pharm = list(X_fp.fp_pharm)
            else:
                fps_pharm = [np.zeros(168, dtype=np.uint8)] * len(fps)
            if hasattr(X_fp, "fp_scaf"):
                fps_scaf = list(X_fp.fp_scaf)
            else:
                fps_scaf = [np.zeros(30, dtype=np.uint16)] * len(fps)
        if update:
            self.fps_ += fps
            self.fps_count_ += fps_count
            self.fps_pharm_ += fps_pharm
            self.fps_scaf_ += fps_scaf
        else:
            self.fps_ = fps
            self.fps_count_ = fps_count
            self.fps_pharm_ = fps_pharm
            self.fps_scaf_ = fps_scaf
            self.bit_d_ = {}
            self.bit_parent_d_ = {}
        if type(bitinfo_df) == pd.DataFrame:
            if "PARENT_ID" not in bitinfo_df.columns:
                bitinfo_df["PARENT_ID"] = None
            for fpbit, frag, diameter, parent_id in bitinfo_df[
                ["SMILES_FRAG", "DIAMETER", "PARENT_ID"]
            ].to_records():
                if fpbit.startswith("MORGAN_"):
                    bit = int(fpbit.partition("_")[2])
                    self.bit_d_[bit] = (frag, diameter)
                    self.bit_parent_d_[bit] = parent_id
        elif not bitinfo_df:
            self.bit_d_ = None

    def add_molfp(self, mol):
        """
        Get Morgan fingerprint from molecule, and update records of
        fingerprint bit.

        Parameters
        ----------
        mol : Mol
            The molecule.

        Returns
        -------
        bitinfo_d : Dict[int, List[Tuple[int, int]]]
            Fingerprint of molecule. Keys are fingerprint bit, and
            values are locations of bit in the molecule as a tuple
            (atom_index, radius).
        """
        count_d = AllChem.GetMorganFingerprint(mol, 2).GetNonzeroElements()
        fp = np.fromiter(count_d, np.uint32)
        count = np.fromiter((min(val, 255) for val in count_d.values()), np.uint8)
        bitinfo_d = {}
        if self.bit_d_ is not None:
            bits = set(
                bit
                for bit in count_d
                if bit not in self.bit_d_ or self.bit_d_[bit][0] is None
            )
            if bits:
                AllChem.GetMorganFingerprint(mol, 2, bitInfo=bitinfo_d)
        self.fps_.append(fp)
        self.fps_count_.append(count)
        pharm_d = vectools.mol_pharm(mol)
        self.fps_pharm_.append(vectools.mol_pharmfp(mol, pharm_d=pharm_d))
        atom_rings = mol.GetRingInfo().AtomRings()
        n_atom = mol.GetNumAtoms()
        mol.SetDoubleProp("AR", len(pharm_d["AR"]))
        mol.SetDoubleProp("HEAVY_ATOM", n_atom)
        mol.SetDoubleProp("NRING", len(atom_rings))
        mol.SetDoubleProp("MAXRING", max([len(ring) for ring in atom_rings], default=0))
        mol.SetDoubleProp("MORGANBIT", len(fp))
        carbons = []
        sp3rings = []
        qindex = 0
        ring_atoms = {i for ring in atom_rings for i in ring}
        atom_bits = np.zeros((n_atom, 4), dtype=np.uint32)
        for i, atom in enumerate(mol.GetAtoms()):
            try:
                atom_bit = rdMolDescriptors.GetAtomPairAtomCode(atom)
                atom_bits[i, 0] = atom_bit
                acode0, acode1 = divmod(atom_bit, 32)
                if acode0 == 1:
                    carbons.append(i)
                    if (i in ring_atoms) and acode1 < 8:
                        sp3rings.append(i)
                qindex += ((acode1 % 8) ** 2) / 2 - 2
            except RuntimeError:
                v = atom.GetDegree()
                if atom.GetSymbol() == "C":
                    carbons.append(i)
                    if (i in ring_atoms) and (v + atom.GetNumImplicitHs()) == 4:
                        sp3rings.append(i)
                qindex += (v ** 2) / 2 - 2
        if carbons:
            mol.SetDoubleProp("FSP3RING", len(sp3rings) / len(carbons))
        else:
            mol.SetDoubleProp("FSP3RING", 0)
        mol.SetDoubleProp("QINDEX", 3 + qindex)
        D = np.digitize(AllChem.GetDistanceMatrix(mol)[:, list(ring_atoms)], bins=[1, 2, 3, 5, 8, 13, 21, 34, 55])
        fp_scaf = np.minimum(np.r_[
            np.bincount(D[list(ring_atoms)].ravel(), minlength=10),
            np.bincount(D[list(ring_atoms.difference(carbons))].ravel(), minlength=10),
            np.bincount(D[list(ring_atoms.difference(sp3rings))].ravel(), minlength=10),
        ], 65535).astype(np.uint16)
        self.fps_scaf_.append(fp_scaf)
        if bitinfo_d:
            for bit, val in bitinfo_d.items():
                atom, radius = bitinfo_d[bit][0]
                if bit in bits:
                    try:
                        frag = smi_frag_from_circular(
                            mol, atom, radius, atom_rings=atom_rings
                        )
                    except RuntimeError:
                        frag = None
                    self.bit_d_[bit] = (frag, 2 * radius)
                for i, radius in val:
                    atom_bits[i, radius + 1] = bit
            for bit, val in bitinfo_d.items():
                if bit not in self.bit_parent_d_:
                    atom, radius = val[0]
                    if radius > 0:
                        self.bit_parent_d_[bit] = f"MORGAN_{atom_bits[atom, radius]}"
                    else:
                        self.bit_parent_d_[bit] = f"ATOMCODE_{atom_bits[atom, radius]}"
        return bitinfo_d

    def get_fps(self, index=slice(None)):
        """
        Get fingerprint from recorded molecules.

        Parameters
        ----------
        index : List[int], optional (default=None)
            A subset index of molecules to get information from.

        Returns
        -------
        X_fp : sparse.csr_matrix[uint8] of shape (:, 2 ** 32)
            Morgan fingerprint matrix of recorded molecules.
        """
        if type(index) == slice:
            fps = self.fps_[index]
            fps_count = self.fps_count_[index]
            fps_pharm = self.fps_pharm_[index]
            fps_scaf = self.fps_scaf_[index]
        else:
            fps = [self.fps_[i] for i in index]
            fps_count = [self.fps_count_[i] for i in index]
            fps_pharm = [self.fps_pharm_[i] for i in index]
            fps_scaf = [self.fps_scaf_[i] for i in index]
        fps_idx = np.cumsum([0] + [len(i) for i in fps])
        X_fp = sparse.csr_matrix(
            (np.concatenate(fps_count), np.concatenate(fps), fps_idx),
            shape=(len(fps), 1 << 32),
        )
        if fps_pharm:
            X_fp.fp_pharm = np.vstack(fps_pharm)
        else:
            X_fp.fp_pharm = np.zeros((0, 168), dtype=np.uint8)
        if fps_scaf:
            X_fp.fp_scaf = np.vstack(fps_scaf)
        else:
            X_fp.fp_scaf = np.zeros((0, 30), dtype=np.uint16)
        return X_fp

    def fps_summary(self, X_fp):
        """
        Get bit summary information of Morgan fingerprint matrix.

        Parameters
        ----------
        X_fp : sparse.csr_matrix[uint8] of shape (:, 2 ** 32)
            Morgan fingerprint matrix of recorded molecules.

        Returns
        -------
        bitinfo_df : DataFrame of shape (:, 3), optional
            Information table of Morgan fingerprint bits. With index
            FPBIT_ID, and 6 columns [SMILES_FRAG, COUNT, NONZERO,
            MAX_ORDER, EVEN_ORDER, DIAMETER, PARENT_ID].
            FPBIT_ID : str
                Fingerprint ID of form "{METHOD}_{BIT}".
            COUNT : int
                Total counts of fingerprint bit.
            NONZERO : float
                Proportions of existence in molecules of fingerprint
                bit.
            MAX_ORDER : int
                Maximum count of fingerprint bit in a molecule.
            EVEN_ORDER : float
                Proportion of non-zero even count of fingerprint bit.
            SMILES_FRAG : str
                SMILES representation of fragments of fingerprint bit.
            DIAMETER : str
                Bonds diameter of the fingerprint fragments, e.g.
                single atom: diameter 0, two connected atoms: diameter
                1, 4-atom path: diameter 3.
            PARENT_ID : str
                Fingerprint "{METHOD}_{BIT}" ID of radius - 1.

        See also
        --------
        self.reset
            Reload Morgan fingerprint bit information.
        """

        bits, nonzeros = np.unique(X_fp.indices, return_counts=True)
        idx_bits = np.argsort(X_fp.indices)
        X_fp_split = np.split(X_fp.data[idx_bits], nonzeros.cumsum()[:-1])
        counts = np.array([val.sum() for val in X_fp_split])
        max_orders = np.array([val.max() for val in X_fp_split])
        even_orders = 1 - np.array([(val % 2).sum() for val in X_fp_split]) / nonzeros
        del X_fp_split
        nonzeros = nonzeros / X_fp.shape[0]
        bitinfo_df = pd.DataFrame(
            {
                "COUNT": counts,
                "NONZERO": nonzeros,
                "MAX_ORDER": max_orders,
                "EVEN_ORDER": even_orders,
            },
            index=[f"MORGAN_{bit}" for bit in bits],
        )
        bitinfo_df["SMILES_FRAG"], bitinfo_df["DIAMETER"] = tuple(
            zip(*[self.bit_d_.get(bit, (None, np.nan)) for bit in bits])
        )
        bitinfo_df["PARENT_ID"] = [self.bit_parent_d_.get(bit) for bit in bits]
        bitinfo_df.index.name = "FPBIT_ID"
        return bitinfo_df

    def transform(self, smiles):
        """
        Get Morgan fingerprints from SMILES.

        Parameters
        ----------
        smiles : List[str]
            Input SMILES.

        Returns
        -------
        X : sparse.csr_matrix[uint8] of shape (n, 2 ** 32)
            Coverted fingerprint array.

        See also
        --------
        rdkit.AllChem.GetMorganFingerprint
            An rdkit function to calculate Morgan Fingerprint of
            uint32 bits.
        get_dense
            Compress a sparse array of fingerprint to a dense array
            with finite bits.
        """
        begin_time = time.time()
        self.reset(type(self.bit_d_) == dict)
        count = 0
        for smi in smiles:
            mol = smi_to_mol(smi)
            if not mol:
                self.fps_.append(np.array([], dtype=np.uint32))
                self.fps_count_.append(np.array([], dtype=np.uint8))
                self.fps_pharm_.append(np.zeros(168, dtype=np.uint8))
                self.fps_scaf_.append(np.zeros(30, dtype=np.uint16))
            else:
                self.add_molfp(mol)
                count += 1
        if self.verbose_:
            logging.info(
                f"Calculate {count} fingerprints from {len(smiles)} smiles, time: {time.time() - begin_time:.1f} s"
            )
        return self.get_fps()


class keydefaultdict(collections.defaultdict):
    """
    Use a function of key as default in defaultdict.
    """

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            out = self[key] = self.default_factory(key)
            return out


def get_generic(smi, n_try=10):
    """
    Get generic scaffold from Murcko scaffold.

    Parameters
    ----------
    smi : str
        Murcko scaffold of SMILES.

    Returns
    -------
    smi_out : str
        Generic scaffold with all atoms to aliphatic carbon and all
        bonds to single.
    """
    pattern_sidechain = re.compile(r"\(C+\)|^C+(?![\d(])|(?<!\))C+$")
    smi_out = re.sub(RE_ATOM, "C", smi_to_charseq(smi))
    smi_out = re.sub(RE_BOND, "", smi_out)
    for i in range(n_try):
        smi_old = smi_out
        mol = Chem.MolFromSmiles(re.sub(pattern_sidechain, "", smi_old), sanitize=False)
        if not mol:
            return ""
        smi_out = Chem.MolToSmiles(mol)
        if smi_out == smi_old:
            break
    else:
        logging.info(
            f"Sidechain remove failed, newest / previous / input scaffold: {smi_out} / {smi_old} / {smi}"
        )
    if not smi_out:
        smi_out = "C"
    return smi_out


def hill_number(counts, q=1):
    if q == 0:
        return len(counts)
    elif q == 1:
        counts = counts / np.sum(counts)
        return np.exp(-np.sum(counts * np.log(counts)))
    elif np.isposinf(q):
        return np.sum(counts) / np.max(counts)
    else:
        counts = counts / np.sum(counts)
        return np.sum(counts ** q) ** (1 / (1 - q))


class ScaffoldSimplifier(object):
    """
    Simplify Murcko scaffold with SMILES string operations.

    Parameters
    ----------
    scafinfo_d : Dict[str, str]
        Information for simplifying scaffold. Keys are Murcko generic
        scaffold, and values are topological simplified scaffold.
    n_try : int
        The maximum number of tries to simplify a generic scaffold.

    Attributes
    ----------
    pattern_dupcarbon_ : re.Pattern
        Regex pattern of continuous carbon in SMILES.
    pattern_ringnum_ : re.Pattern
        Regex pattern of ring number in generic scaffold.
    generic_d_ : keydefaultdict[str, str]
        Map Murcko scaffold to Murcko generic scaffold.
    simplify_d_ : keydefaultdict[str, str]
        Map Murcko generic scaffold to topological simplified scaffold.

    References
    ----------
    * Jakub Velkoborsky, David Hoksza
      Scaffold analysis of PubChem database as background for
      hierarchical scaffold-based visualization
      https://link.springer.com/article/10.1186/s13321-016-0186-7
    """

    def __init__(self, scafinfo_d={}, n_try=20):
        self.pattern_dupcarbon_ = re.compile(r"C{2,}")
        self.pattern_ringnum_ = re.compile(RE_RINGNUM)
        self.reset(scafinfo_d)
        self.n_try_ = n_try

    def drop_dupcarbon(self, smi):
        """
        Remove duplicate carbons (partially) in generic scaffold by
        regex operation.

        Parameters
        ----------
        smi : str
            Input generic scaffold

        Returns
        -------
        smi_out : str
            Simplified scaffold.
        """
        ring_d = collections.Counter()
        end = 0
        strs = []
        rings_new = set()
        rings = set()
        rings_keep = set()
        rings_diff = set()
        for match in re.finditer(self.pattern_ringnum_, smi):
            begin_new = match.start()
            end_new = match.end()
            ringnum = smi[begin_new:end_new]
            inter = smi[end:begin_new]
            if inter:
                if strs and not rings_diff - rings_new and not rings_keep - rings_new:
                    strs[-1] = strs[-1].replace("CC", "C")
                rings_diff = rings_new - rings
                rings_keep = rings_keep - (rings - rings_new)
                rings = rings_new.copy()
                if inter.startswith("("):
                    rings_keep = rings_keep.union(rings_diff)
                strs.append(re.sub(self.pattern_dupcarbon_, "CC", inter))
            if ring_d[ringnum] % 2:
                rings_new.remove(ringnum)
            else:
                rings_new.add(ringnum)
            strs[-1] += ringnum
            ring_d[ringnum] += 1
            end = end_new
        #            print(strs, rings_diff, rings, rings_new, rings_keep)
        strs.append(smi[end:])
        smi_out = "".join(strs)
        return smi_out

    def reset(self, scafinfo_d={}, update=False):
        """
        Clear, reload or update saved scaffold information.

        Parameters
        ----------
        scafinfo_d : Dict[str, str]
            Information for simplifying scaffold. Keys are Murcko
            generic scaffold, and values are simplified scaffold.
        update : bool
            Whether keep existed records.
        """
        if update:
            self.simplify_d_.update(scafinfo_d)
        else:
            self.simplify_d_ = keydefaultdict(
                lambda x: canonize_smi(x, sanitize=False), scafinfo_d
            )
            self.simplify_d_[
                "C1CC1C(C1C2C(C3CC3)C12)C1C2C(C3CC3)C12"
            ] = "C1CC1C(C1C2C(C3CC3)C12)C1C2C(C3CC3)C12"

    def simplify_generic(self, smi):
        """
        Get topological simplified scaffold from generic scaffold.

        Parameters
        ----------
        smi : str
            Generic scaffold of SMILES.

        Returns
        -------
        smi_out : str
            Topological simplified scaffold with all unnecessary
            carbons for keeping topological structure removed.
        """
        smi_out = self.drop_dupcarbon(smi_decouple_ring(smi, joint=""))
        for i in range(self.n_try_):
            smi_old = smi_out
            try:
                smi_out = self.drop_dupcarbon(self.simplify_d_[smi_old])
                if smi_out == smi_old:
                    break
            except TypeError:
                logging.error(f"Scaffold simplified failed: {smi_out} from {smi}")
                break
        else:
            logging.info(
                f"Generic simplification failed, newest / previous / input generic scaffold: {smi_out} / {smi_old} / {smi}"
            )
        return smi_out

    def transform(self, smiles, name="MURCKO"):
        """
        Convert scaffolds to generic (all atoms to carbon and bonds to
        single) and simplified scaffolds.

        Parameters
        ----------
        smiles : List[str]
            Scaffolds of SMILES.

        Returns
        -------
        idx_generic : pd.Categorical
            Index of input scaffolds on generic scaffolds table.
        scafinfo_df : DataFrame
            Information table of generic scaffolds, include 4 columns:
            GENERIC_SCAFFOLD (index) : str
                Generic scaffolds.
            SIMPLE_SCAFFOLD (index) : str
                The topological simplified (oprea) scaffold of the generic.
            COUNTMOL_{NAME} : int
                The number of input scaffolds belongs to the generic.
            COUNT_{NAME} : int
                The number of unique scaffolds belongs to the generic.
        """
        begin_time = time.time()
        count_d = {}
        values_scaf, idx_scaf, count_d["SCAFFOLD"] = np.unique(
            smiles, return_inverse=True, return_counts=True
        )
        logging.info(f"Get {len(values_scaf)} unique scaffolds")
        scaffolds_generic = [get_generic(smi) for smi in values_scaf]
        values_generic, idx_generic, count_d["GENERIC"] = np.unique(
            scaffolds_generic, return_inverse=True, return_counts=True
        )
        logging.info(
            f"Get {len(values_generic)} generic scaffolds, time: {time.time() - begin_time:.1f} s"
        )
        scaffolds_simple = [self.simplify_generic(smi) for smi in values_generic]
        values_simple, count_d["SIMPLE"] = np.unique(
            scaffolds_simple, return_counts=True
        )
        idx_generic = pd.Categorical.from_codes(idx_generic[idx_scaf], values_generic)
        scafinfo_df = pd.DataFrame(
            {
                "GENERIC_SCAFFOLD": values_generic,
                "SIMPLE_SCAFFOLD": scaffolds_simple,
                f"COUNTMOL_{name}": np.bincount(idx_generic.codes),
                f"COUNT_{name}": count_d["GENERIC"],
            }
        )
        logging.info(
            f"Finish scaffold simplification of {len(smiles)} scaffolds, time: {time.time() - begin_time:.1f} s. Effective number of {name} species (Hill numbers of order 0/1/2):\n"
            + "\n".join(
                [
                    f"{key}: {len(val)} / {hill_number(val, 1):.1f} / {hill_number(val, 2):.1f}"
                    for key, val in count_d.items()
                ]
            )
        )
        return idx_generic, scafinfo_df


class Predictor(object):
    """
    Wrap a predictor function for multi-output goal-directed learning.

    Parameters
    ----------
    pred_funcs : List[Callable[[List[str]], DataFrame[float]]]
        The functions for prediction on SMILES.
    filepath : str
        Filepath of output.
    comp_cols : List[str]
        Post-computed columns after other columns calculated.
    test_smiles : List[str]
        SMILES used for test `pred_funcs`.

    Attributes
    ----------
    smi_recorder_ : SmilesRecorder
        Saved prediction result. Keys are predicted SMILES, and values
        are saved output.
    model_scaf_ : ScaffoldSimplifier
        Get scaffold information for count-based rewards.
    pred_cols_ : List[str]
        Columns generated by `pred_funcs`.
    molfunc_cols : List[str]
        Numeric properties for molecules in `smi_recorder_`
    comp_cols_ : List[str]
        Post-computed columns after other columns calculated.
    pred_size_ : int
        Output size of all predict / molecule / count-based functions.
    """

    def __init__(
        self, pred_funcs=[], filepath=None, comp_cols=[], test_smiles=["c1ccccc1C"]
    ):
        self.smi_recorder_ = SmilesRecorder(filepath=filepath, verbose=False)
        self.model_scaf_ = ScaffoldSimplifier()
        self.molfunc_cols_ = list(self.smi_recorder_.molnumfunc_d_)
        self.comp_cols_ = list(comp_cols)
        if pred_funcs:
            Y = pd.concat([pred_func(test_smiles) for pred_func in pred_funcs], axis=1)
            Y.index = test_smiles
        else:
            Y = pd.DataFrame(index=test_smiles, dtype=float)
        self.pred_cols_ = list(Y.columns)
        self.columns_ = self.molfunc_cols_ + ["QED", "MCE18"] + list(FPWEIGHT_D) + self.comp_cols_ + list(Y.columns)
        self.pred_size_ = len(self.columns_)
        for col in Y.columns:
            self.smi_recorder_.nums_d_[col]
        self.pred_funcs_ = pred_funcs
        logging.info(f"Set predict function of size {self.pred_size_}\n{Y.to_string()}")

    def predict(self, smiles, column_d={}):
        """
        Get predicted values for SMILES.

        Parameters
        ----------
        smiles : list
            Input SMILES.
        column_d : Dict[str, List[Any]], optional (default={})
            Other columns used for appending into records. Keys are
            column names, and values should have same length with input
            `smiles`.

        Returns
        -------
        Y_df : DataFrame
            Predicted values of input SMILES (with NA values as 0).
            Index as canonical SMILES (may be duplicated).
        valids : 1d-array[int]
            Index of SMILES for valid property.
        """
        smi_df = self.smi_recorder_.fit(smiles, column_d=column_d)
        smiles_pred = smi_df.loc[
            smi_df[self.pred_cols_].isna().any(axis=1), "SMILES"
        ].tolist()
        if smiles_pred:
            Y_pred = pd.concat(
                [pred_func(smiles_pred) for pred_func in self.pred_funcs_], axis=1
            )
            idx_canon = [self.smi_recorder_.smi_d_[smi] for smi in smiles_pred]
            self.smi_recorder_.update_record(
                idx_canon, {col: y.values for col, y in Y_pred.iteritems()}
            )
        Y_raw_df = self.smi_recorder_.get_library(filepath=False, calc_fpscore=True)
        if any(col.startswith("COUNTSCAF_") for col in self.comp_cols_):
            idx_generic, scafinfo_df = self.model_scaf_.transform(
                Y_raw_df["MURCKO_SCAFFOLD"]
            )
            Y_raw_df["COUNTSCAF_MURCKO"] = Y_raw_df["MURCKO_SCAFFOLD"].cat.codes
            Y_raw_df["COUNTSCAF_GENERIC"] = idx_generic.codes
            Y_raw_df["COUNTSCAF_SIMPLE"] = (
                scafinfo_df["SIMPLE_SCAFFOLD"].loc[idx_generic.codes].values
            )
            logging.info("Collect scaffold information (COUNTSCAF_*)")

        Y_df = Y_raw_df.iloc[self.smi_recorder_.idx_canon_]
        for col in self.comp_cols_:
            if col.startswith("COUNTSCAF_"):
                Y_df[col] = np.log10(Y_raw_df[col].value_counts().loc[Y_df[col]].values)
        Y_df = Y_df[self.columns_]
        Y_df.loc[self.smi_recorder_.idx_canon_ == -1] = np.nan
        valids = np.nonzero(np.isfinite(Y_df.values).all(axis=1))[0]
        Y_df.fillna(0, inplace=True)
        Y_df.index = np.array(list(self.smi_recorder_.smi_d_) + [None])[
            self.smi_recorder_.idx_canon_
        ]
        Y_df.index.name = "SMILES"
        logging.info(
            f"Get {len(valids)} valids / {len(Y_df)} SMILES calculated in predictor"
        )
        return Y_df, valids


def smi_strhash(smi):
    """
    Get string hash for SMILES (usually canonical). Used for fast ID
    generation for molecules and scaffolds.

    Parameters
    ----------
    smi : str
        Input SMILES.

    Returns
    -------
    strhash : str
        36-based hash of SMILES with 12 digits. First 6 digits for
        structural features of SMILES and last 6 digits as uniform hash
        to avoid collisions.
            1 : Length of SMILES (divided by 8).
            2 : Count of aliphatic carbons (divided by 4).
            3 : Count of aromatic carbons (divided by 4).
            4 : Count of double/triple bonds.
            5 : Count of branched chains.
            6 : Count of rings.
            7 ~ 12 : base36 hash.

    Examples
    --------
    >>> smi_strhash("c1cccc(CC2=NNC(=O)c3ccccc23)c1")
    '303223pzgwhm'
    """
    char36 = "0123456789abcdefghijklmnopqrstuvwxyz"
    char_d = collections.Counter(smi.replace("Cl", "Q"))
    rawints = [
        len(smi) >> 3,
        char_d["C"] >> 2,
        char_d["c"] >> 2,
        char_d["="] + char_d["#"],
        char_d["("],
        sum([char_d[key] for key in "123456789"]) >> 1,
    ] + list(hashlib.blake2s(bytes(smi, "utf-8"), digest_size=6).digest())
    strhash = "".join([char36[i % 36] for i in rawints])
    return strhash


def qed_ads(x, adsParameter):
    """
    ADS function for QED calculation
    """
    p = adsParameter
    exp1 = 1 + np.exp(-1 * (x - p.C + p.D / 2) / p.E)
    exp2 = 1 + np.exp(-1 * (x - p.C - p.D / 2) / p.F)
    dx = p.A + p.B / exp1 * (1 - 1 / exp2)
    return dx / p.DMAX


def calc_scscore(X, fpweight):
    if sparse.issparse(X):
        X = get_dense(X, 1024).astype(np.float32)
    valids = X.any(axis=1)
    for i in range(0, len(fpweight), 2):
        X = X @ fpweight[i]
        X += fpweight[i + 1]
        if i < len(fpweight) - 2:
            np.maximum(X, 0, out=X)  # ReLU
    out = np.where(valids, 1 + 4 / (1 + np.exp(-X.ravel())), np.nan)
    return out


def fpscore_predict(
    smi_df,
    X_fp=None,
    methods={"QED", "MCE18", "SASCORE", "SCSCORE", "RASCORE", "NPSCORE"},
    X_ref_d={},
    model_d={},
    batch_size=100000,
):
    """
    Calculate scores using Morgan/pharm/scaf fingerprint weights and
    properties.
    The calculated scores may be partially based on FPWEIGHT_D. All
    available scores are:
        QED, MCE18, SASCORE, SCSCORE, RASCORE (RASTEP2), NPSCORE.
    It also supports QSAR models trained with MORGAN/PHARM/SCAF
    fingerprints.

    Parameters
    ----------
    smi_df : DataFrame
        Information table of recorded molecules.
    X_fp : sparse.csr_matrix of shape (n, 2 ** 32)
        Morgan fingerprint matrix of recorded molecules.
    methods : Set[str]
        Choices of scores to calculate.
    X_ref_d : Dict[str, sparse.csr_matrix of shape (n_ref, 2 ** 32) or 1d-array of shape (n_ref, 168)]
        Reference fingerprint matrices. If provided, it will calculate
        {key}_MORGAN_*, {key}_PHARM_*, {key}_SCAF_* when available.
    model_d : Dict[str, ]
        QSAR models for prediction. Keys are column names for output,
        should in values with sklearn-API and method `predict`.

    Returns
    -------
    smi_df : DataFrame
        Information table with new predicted scores appended.


    Reference
    ---------
    * Peter Ertl, Ansgar Schuffenhauer
      Estimation of synthetic accessibility score of drug-like
      molecules based on molecular complexity and fragment
      contributions.
      https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8

    * Connor W. Coley, Luke Rogers, William H. Green, Klavs F. Jensen
      SCScore: Synthetic Complexity Learned from a Reaction Corpus
      https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00622

    * Amol Thakkar, Veronika Chadimová, Esben Jannik Bjerrum, Ola Engkvist, Jean-Louis Reymond
      Retrosynthetic accessibility score (RAscore) – rapid machine learned synthesizability classification from AI driven retrosynthetic planning
      https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05401a

    * Peter Ertl, Silvio Roggo, and Ansgar Schuffenhauer
      Natural Product-likeness Score and Its Application for Prioritization of Compound Libraries
      http://pubs.acs.org/doi/abs/10.1021/ci700286x
    """
    begin_time = time.time()
    if "QED" in methods:
        cols = ["MW", "LOGP", "HBA", "HBD", "TPSA", "RTB", "AR", "HIT_SMARTS_BRENK"]
        smi_df["QED"] = np.exp(
            np.sum(
                [
                    w * np.log(qed_ads(smi_df[col].values, val))
                    for col, val, w in zip(
                        cols, QED.adsParameters.values(), QED.WEIGHT_MEAN
                    )
                ],
                axis=0,
            )
            / sum(QED.WEIGHT_MEAN)
        ).astype(np.float32)
    if "MCE18" in methods:
        smi_df["MCE18"] = smi_df.eval(
            "QINDEX * (2 * FSP3RING / (1 + FSP3) + (AR > 0) + (AR < NRING) + (CHIRALC > 0) + (SPIRO > 0))"
        ).astype(np.float32)
    if X_fp is None:
        return smi_df
    X_tmp_d = {}
    X_tmp_d["bitx"], X_tmp_d["idx"] = np.unique(X_fp.indices, return_inverse=True)
    fpweight = FPWEIGHT_D.get("SASCORE")
    if "SASCORE" in methods and (fpweight is not None):
        sascore1 = sparse_weightsum(
            X_fp, fpweight[0], fpweight[1], null_value=-4, bitx=X_tmp_d["bitx"], idx=X_tmp_d["idx"]
        )
        sascore1 = sascore1 / X_fp.sum(axis=1).A1
        sascore = (
            sascore1
            + smi_df.eval(
                "- (HEAVY_ATOM ** 1.005 - HEAVY_ATOM) - log10(CHIRALC + 1) - log10(BRIDGEHEAD + 1) - log10(SPIRO + 1) - log10((MAXRING > 8) + 1)"
            )
            + smi_df.eval("log(HEAVY_ATOM / MORGANBIT) * 0.5").clip(0)
        ).astype(np.float32)
        sascore = 11 - (sascore + 5) / 6.5 * 9
        sascore[sascore > 8] = 8 + np.log(sascore[sascore > 8] - 7)
        smi_df["SASCORE"] = sascore.clip(1, 10).where(smi_df["MORGANBIT"] > 0)

    fpweight = FPWEIGHT_D.get("SCSCORE")
    if "SCSCORE" in methods and (fpweight is not None):
        X_tmp_d["dense"] = get_dense(X_fp, 1024)
        if X_fp.shape[0] <= batch_size:
            scscore = calc_scscore(X_tmp_d["dense"].astype(np.float32), fpweight)
        else:
            scscore = np.concatenate([calc_scscore(X_tmp_d["dense"][i:(i + batch_size)].astype(np.float32), fpweight) for i in range(0, X_fp.shape[0], batch_size)])
        smi_df["SCSCORE"] = scscore
    fpweight = FPWEIGHT_D.get("RASCORE")
    if ("RASCORE" in methods or "RASTEP2" in methods) and (fpweight is not None):
        sabit, saweight = FPWEIGHT_D["SASCORE"]
        sascorebins = sparse_weightsum(X_fp, sabit, saweight, bitx=X_tmp_d["bitx"], idx=X_tmp_d["idx"], null_value=-4, bins=np.arange(-3.5, 2.5, 0.5))
        sascorebins = np.minimum(sascorebins.cumsum(axis=1), 255).astype(np.uint8)
        if "dense" not in X_tmp_d:
            X_tmp_d["dense"] = get_dense(X_fp, 1024)
        X_tmp_d["fp_ls"] = [
            pd.DataFrame(X_tmp_d["dense"], columns=[f"MORGAN_{i}" for i in range(1024)]),
            pd.DataFrame(X_fp.fp_pharm, columns=[f"PHARM_{i}" for i in range(168)]),
            pd.DataFrame(X_fp.fp_scaf, columns=[f"SCAF_{i}" for i in range(30)]),
            pd.DataFrame(sascorebins, columns=[f"SABITWT_{i}" for i in range(13)])
        ]
        if X_fp.shape[0] <= batch_size:
            rascore = fpweight.predict_proba(smi_df[fpweight.feature_name_[:19]].join(X_tmp_d["fp_ls"]))
        else:
            rascore = np.vstack([fpweight.predict_proba(smi_df[i:(i + batch_size)][fpweight.feature_name_[:19]].join([X[i:(i + batch_size)] for X in X_tmp_d["fp_ls"]])) for i in range(0, X_fp.shape[0], batch_size)])
        smi_df["RASCORE"] = 1 - rascore[:, 0]
        smi_df["RASTEP2"] = rascore[:, 2]
    fpweight = FPWEIGHT_D.get("NPSCORE")
    if "NPSCORE" in methods and (fpweight is not None):
        npscore = (
            sparse_weightsum(X_fp, fpweight[0], fpweight[1], bitx=X_tmp_d["bitx"], idx=X_tmp_d["idx"])
            / smi_df["HEAVY_ATOM"]
        )
        npscore[npscore > 4] = 4 + np.log10(npscore[npscore > 4] - 3)
        npscore[npscore < -4] = -4 - np.log10(-npscore[npscore < -4] - 3)
        smi_df["NPSCORE"] = npscore
    del X_tmp_d
    model_d, method_d = vectools.collect_feature_names(model_d, methods=["MORGAN", "PHARM", "SCAF"])
    if model_d:
        X = []
        morgan_bits = method_d.get("MORGAN")
        if morgan_bits:
            X.append(pd.DataFrame(get_dense(X_fp, morgan_bits), columns=[f"MORGAN_{i}" for i in morgan_bits]))
        if "PHARM" in method_d:
            X.append(pd.DataFrame(X_fp.fp_pharm, columns=[f"PHARM_{i}" for i in range(168)]))
        if "SCAF" in method_d:
            X.append(pd.DataFrame(X_fp.fp_scaf, columns=[f"SCAF_{i}" for i in range(30)]))
        X = pd.concat(X, axis=1)
        for key, Y in vectools.model_predict(model_d, X, batch_size=batch_size).items():
            if len(Y.shape) == 1:
                smi_df[key] = Y
            else:
                smi_df[[f"{key}{i + 1}" for i in range(Y.shape[1])]] = Y
    for key, X_ref_fp in X_ref_d.items():
        if X_ref_fp.shape[1] == 168:
            X_grp = [("PHARM", X_fp.fp_pharm, X_ref_fp)]
        else:
            X_grp = [("MORGAN", X_fp, X_ref_fp), ("PHARM", X_fp.fp_pharm, X_ref_fp.fp_pharm), ("SCAF", X_fp.fp_scaf, X_ref_fp.fp_scaf)]
        for method, X, X_ref in X_grp:
            S = vectools.similarity(X, X_ref)
            for stat, func in vectools.STAT_FUNC_D.items():
                smi_df[f"{key}_{method}_{stat}"] = func(S)
    logging.info(
        f"Finish fingerprint score calculation of {len(smi_df)} molecules, time: {time.time() - begin_time:.1f} s"
    )
    return smi_df


def save_fps(filepath, X):
    """
    Save Morgan fingerprint matrix into *.npz file. It also supports
    pharm and scaf fingerprint calculated in SmilesFingerprint class.

    Parameters
    ----------
    filepath : str
        Filepath of output.
    X : sparse.csr_matrix[uint8] of shape (n, 2 ** 32)
        Morgan fingerprint matrix of recorded molecules.
    """
    fp_d = {
        "indices": X.indices,
        "indptr": X.indptr,
        "format": X.format.encode('ascii'),
        "shape": X.shape,
        "data": X.data
    }
    if hasattr(X, "fp_pharm"):
        fp_d["fp_pharm"] = X.fp_pharm
    if hasattr(X, "fp_scaf"):
        fp_d["fp_scaf"] = X.fp_scaf
    np.savez_compressed(filepath, **fp_d)


def load_fps(filepath):
    """
    Load *.npz file as fingerprint matrix.

    Parameters
    ----------
    filepath : str
        Filepath of *.npz fingerprint matrices.

    Returns
    -------
    X : sparse.csr_matrix[uint8] of shape (n, 2 ** 32)
        Morgan fingerprint matrix of recorded molecules.
    """
    fp_d = dict(np.load(filepath))
    sp_format = getattr(sparse, fp_d.pop("format").item().decode('ascii') + '_matrix')
    X_fp = sp_format((fp_d.pop('data'), fp_d.pop('indices'), fp_d.pop('indptr')), shape=fp_d.pop('shape'))
    for key, val in fp_d.items():
        setattr(X_fp, key, val)
    return X_fp


# def load_library(filepath, smi_recorder=None):
#     """
#     Load a library from files.

#     Parameters
#     ----------
#     filepath : str
#         Filepath of input library.
#     smi_recorder : SmilesRecorder
#         If not None, append library to an existed library.

#     Returns
#     -------
#     smi_recorder : SmilesRecorder
#         Loaded library from file.
#     """
#     if not smi_recorder:
#         smi_recorder = SmilesRecorder()
#     if filepath.endswith("_result.csv"):
#         smi_df = read_csv(filepath).to_pandas()
#         idx_canon = smi_recorder.load_library(smi_df)
#         bitinfo_df = read_csv(
#             filepath.replace("_result.csv", "_bitinfo.csv")
#         ).to_pandas()
#         bitinfo_df.set_index(bitinfo_df.columns[0], inplace=True)
#         X_fp = load_fps(filepath.replace("_result.csv", "_fps.npz"))
#         if len(smi_recorder) - smi_recorder.begin_size_ < len(smi_df):
#             ## If overlap between input and existed records
#             idx_canon, idx_raw = np.unique(idx_canon, return_index=True)
#             i_begin = np.searchsorted(idx_canon, smi_recorder.begin_size_)
#             X_fp = X_fp[idx_raw[i_begin:]]
#         smi_recorder.model_fp_.reset(bitinfo_df, X_fp, update=True)
#     logging.info(f"Load library of {len(smi_df)} molecules")
#     return smi_recorder


class SmilesEvaluator(object):
    """
    Evaluation of a library (SmilesRecorder) on fingerprint scores and
    scaffolds.

    Parameters
    ----------
    filepath : str
        Filepath of output.
    sas_fpweights : Tuple[1darray[int], 1darray[float]]
        Morgan fingerprint bits and weights of synthetic accessibility
        score (sascore).
    scs_fpweights : List[ndarray[float]]
        Morgan fingerprint weights of synthetic complexity score
        (scscore).

    Attributes
    ----------
    filepath_ : str
        Work directory to save output files.
    fpscore_d_ : Dict[int, float]
        To calculate synthetic accessibility score (sascore) from
        Morgan fingerprint.
    model_scaf_ : ScaffoldSimplifier
        To get scaffold information from input SMILES.
    """
    def __init__(self, filepath):
        self.filepath_ = filepath
        self.model_scaf_ = ScaffoldSimplifier()
        self.eval_names_ = []

#     def load_library(self, filepath, smi_recorder=None):
#         """
#         Load a library from files.

#         Parameters
#         ----------
#         filepath : str
#             Filepath of input library.
#         smi_recorder : SmilesRecorder
#             If not None, append library to an existed library.

#         Returns
#         -------
#         smi_recorder : SmilesRecorder
#             Loaded library from file.
#         """
#         smi_recorder = load_library(filepath, smi_recorder)
#         if filepath.endswith("_result.csv"):
#             scafinfo_df = read_csv(
#                 filepath.replace("_result.csv", "_scafinfo.csv")
#             ).to_pandas()
#             scafinfo_df.set_index("GENERIC_SCAFFOLD", inplace=True)
#             self.model_scaf_.reset(
#                 scafinfo_df["SIMPLE_SCAFFOLD"].to_dict(), update=True
#             )
#         return smi_recorder

    def evaluate(
        self, smi_recorder, name, smi_join_df=None, subset=0, query=None, X_ref_d={}, n_plot=60, calc_tmap=True
    ):
        """
        Evaluate and save a molecule library to files.

        Parameters
        ----------
        smi_recorder : SmilesRecorder
            The recorder of library to save.
        name : str
            Name to use for saving files.
        smi_join_df : DataFrame
            Other columns to join with information table. Index should
            be canonical SMILES.
        subset : int
            How to use SMILES from index of `smi_join_df` as a subset.
            If 1, use as a subset of library.
            If 0, only append additional columns to library.
            If -1, drop subset from library.
        query : str
            A query expression used for conditional filter of library.
        X_ref_d : Dict[str, 2d-array or sparse.csr_matrix[uint8]]
            Reference fingerprint for similarity calculation. Keys are
            names of reference, values are fingerprint matrices for
            similarity comparison.
        n_plot : int
            If nonzero, plot first molecules as example.
        calc_tmap : bool
            Whether calculate TMAP of Morgan fingerprint for library
            for visualization.

        Returns
        -------
        smi_df : DataFrame
            Information table of output SMILES.
        """
        name_prefix, _, name_suffix = name.rpartition("_n")
        if name_suffix.partition("_")[0].isdigit():
            name = name_prefix
        smi_df = smi_recorder.get_library(calc_fpscore=True, X_ref_d=X_ref_d)
        self.smi_df_ = smi_df
        if smi_join_df is not None and len(smi_join_df) > 0:
            if subset < 0:
                duplicates = smi_df["SMILES"].isin(smi_join_df.index)
                smi_df = smi_df.loc[~duplicates]
                logging.info(f"Remove {duplicates.sum()} molecules overlap with input")
            else:
                diff_cols = smi_join_df.columns.difference(smi_df.columns).tolist()
                smi_df = smi_df.join(
                    smi_join_df[diff_cols],
                    on="SMILES",
                    how=["left", "inner"][int(subset)],
                )
            logging.info(f"Subset choice: {subset}, {len(smi_df)} molecules left")
        # @NOTE(Du Jiewen): Annotation by Du Jiewen 
        # if query:
        #     smi_df.query(query, inplace=True)
        #     logging.info(
        #         f"Filter by query expression: {query}, {len(smi_df)} molecules left"
        #     )
        n_smi = len(smi_df)
        if n_smi == 0:
            logging.error("No molecules to output")
            return None
        elif n_smi == len(smi_recorder):
            index = slice(None)
        else:
            index = smi_df.index
        self.eval_names_.append(f"{name}_n{n_smi}")
        if smi_recorder.model_fp_:
            ## Morgan fingerprint and statistics
            X_fp = smi_recorder.model_fp_.get_fps(index)
            bitinfo_df = smi_recorder.model_fp_.fps_summary(X_fp)
            # @NOTE(Du Jiewen): Annotation by Du Jiewen 
            # save_fps(f"{self.filepath_}{name}_n{n_smi}_fps.npz", X_fp)
            # #            bitinfo_df.sort_values(["NONZERO", "COUNT"], ascending=False, inplace=True)
            # bitinfo_df.to_csv(
            #     f"{self.filepath_}{name}_n{n_smi}_bitinfo.csv", float_format="%g"
            # )
            # count_d = {
            #     key: val["COUNT"].values for key, val in bitinfo_df.groupby("DIAMETER")
            # }
            # logging.info(
            #     "Effective number of Morgan fragment species (Hill numbers of order 0/1/2):\n"
            #     + "\n".join(
            #         [
            #             f"ECFP{key}: {len(val)} / {hill_number(val, 1):.1f} / {hill_number(val, 2):.1f}"
            #             for key, val in count_d.items()
            #         ]
            #     )
            # )
            smi_df["LIBSIM_MORGAN"] = (
                sparse_weightsum(
                    X_fp,
                    bitinfo_df.index.str[7:].astype(int).values,
                    bitinfo_df["NONZERO"].values,
                )
                / X_fp.sum(axis=1).A1
            )
            mean_pharm = X_fp.fp_pharm.mean(axis=0).round(0).astype(np.uint8)
            smi_df["LIBSIM_PHARM"] = lib_similarity(X_fp.fp_pharm, mean_pharm)
        
        return smi_df
        
        # @NOTE(Du Jiewen): Annotation by Du Jiewen 
        #     mean_scaf = X_fp.fp_scaf.mean(axis=0).round(0).astype(np.uint16)
        #     smi_df["LIBSIM_SCAF"] = lib_similarity(X_fp.fp_scaf, mean_scaf)
        #     if calc_tmap and n_smi > 10:
        #         if n_smi > 100000:
        #             tmap_dim = 128
        #         elif n_smi > 50000:
        #             tmap_dim = 256
        #         else:
        #             tmap_dim = 512
        #         try:
        #             smi_df["LIBMAP_MORGAN_X"], smi_df["LIBMAP_MORGAN_Y"], parents, smi_df["LIBMAP_MORGAN_DEGREE"] = vectools.tree_map(X_fp, tmap_dim)
        #             smi_df["LIBMAP_MORGAN_PARENT"] = np.r_[smi_df["SMILES"], [""]][parents]
        #         except Exception:
        #             logging.exception("TMAP calculation failed: ")
        # idx_generic, scafinfo_df = self.model_scaf_.transform(
        #     smi_df["MURCKO_SCAFFOLD"].astype("O").fillna("")
        # )
        # scafinfo_df.index = [smi_strhash(scaf) for scaf in idx_generic.categories]
        # scafinfo_df.rename_axis("GENERIC_SCAFFOLD_ID", inplace=True)
        # smi_df["GENERIC_SCAFFOLD_ID"] = scafinfo_df.index[idx_generic.codes]
        # scafinfo_df_ls = [scafinfo_df]
        # if "AS_ALL" in smi_df.columns:
        #     ## RENOVA.Asteroid support
        #     idx_generic, scafinfo_df = self.model_scaf_.transform(
        #         smi_df["AS_ALL"].astype("O").fillna(""), "AS"
        #     )
        #     scafinfo_df.index = [smi_strhash(scaf) for scaf in idx_generic.categories]
        #     scafinfo_df.rename_axis("GENERIC_SCAFFOLD_ID", inplace=True)
        #     smi_df["AS_GENERIC_SCAFFOLD_ID"] = scafinfo_df.index[idx_generic.codes]
        #     scafinfo_df_ls.append(scafinfo_df)
        #     scafinfo_df, idx_scaf = df_merge(scafinfo_df_ls, "GENERIC_SCAFFOLD_ID")
        #     scafinfo_df.set_index("GENERIC_SCAFFOLD_ID", inplace=True)
        # cols_count = [col for col in scafinfo_df.columns if col.startswith("COUNT")]
        # scafinfo_df.fillna({col: 0 for col in cols_count}, inplace=True)
        # scafinfo_df.sort_values(cols_count, ascending=False, inplace=True)
        # scafinfo_df.to_csv(
        #     f"{self.filepath_}{name}_n{n_smi}_scafinfo.csv", float_format="%g"
        # )
        # smi_df.reset_index(drop=True, inplace=True)
        # name_nopath = name.rpartition("/")[-1]
        # smi_df = df_fillcol(
        #     smi_df, "NAME", "SMILES", lambda x: f"{name_nopath}_{smi_strhash(x)}"
        # )
        # smi_df.to_csv(
        #     f"{self.filepath_}{name}_n{n_smi}_result.csv",
        #     index=False,
        #     float_format="%g",
        # )
        # record = summary_record(smi_df)
        # logging.info(
        #     "Summary of library properties:\n" + record.to_string(float_format="%.4g")
        # )
        # if n_plot:
        #     mols = [smi_to_mol(smi) for smi in smi_df["SMILES"].values[:n_plot]]
        #     highlight_atoms = [None] * len(mols)
        #     highlight_bonds = None
        #     if smi_recorder.core_:
        #         mol_core = smi_to_mol(smi_remove_site(smi_recorder.core_))
        #         if mol_core:
        #             for i, mol in enumerate(mols):
        #                 if mol:
        #                     highlight_atoms[i] = list(mol.GetSubstructMatch(mol_core))
        #         else:
        #             logging.info(
        #                 f"Plot core substructure failure: {smi_recorder.core_}"
        #             )
        #     if smi_recorder.model_fp_ and len(mols) > 1:
        #         highlight_bonds = set(bitinfo_df.index[(bitinfo_df["NONZERO"] >= 0.6) & (bitinfo_df["DIAMETER"] >= 4)])
        #         logging.info(f"Common-core Morgan bits: {len(highlight_bonds)}")
        #     plt = smiles_plot(
        #         mols,
        #         sanitize=True,
        #         n_column=6,
        #         n_plot=n_plot,
        #         highlight_atoms=highlight_atoms,
        #         highlight_bonds=highlight_bonds,
        #         legends=smi_df["NAME"].tolist(),
        #     )
        #     with open(f"{self.filepath_}{name}_n{n_smi}_example.svg", "w") as f:
        #         f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + plt.data)
        # logging.info(f"Finish writing output at {self.filepath_}{name}_n{n_smi}")
        # return smi_df


def summary_record(smi_df):
    record = smi_df.describe().T
    record["description"] = [
        get_column_doc(col).replace(" ", "_") for col in record.index
    ]
    return record


def lib_similarity(X_fp, x_ref):
    """
    Calculate fingerprint similarity between library and reference.

    Parameters
    ----------
    X_fp : 2d-array of shape (n, m)
        Fingerprints of molecules.
    x_ref : 1d-array of length m
        Reference fingerprint for similarity calculation.

    Returns
    -------
    s : 1d-array of length n
        Similarity between fingerprint of input molecules and reference.
    """
    s = (np.minimum(X_fp, x_ref).sum(axis=1) / np.maximum(X_fp, x_ref).sum(axis=1)).astype(np.float32)
    return s


def df_merge(df_ls, key_col, idx_col=None, diff_cols=[], fill_diff=0):
    """
    Merge multiple dataframes (with same / different rows and columns)
    by key column.

    Parameters
    ----------
    df_ls : List[DataFrame]
        Input DataFrames to merge.
    key_col : str
        The column used as index when merging, elements will be unique
        after merge.
    idx_col : str, optional
        If exist, used to indicate which input each record comes from
        as an output column. The form should be "01" strings, such as
        "000101" if input 6 DataFrames, and the record exist in
        DataFrame 3 and 5.
    diff_cols : List[str]
        The columns indicating distinctive among different input
        DataFrames. They will be added "_lib{n}" as suffix of column
        names.
    fill_diff : float or str
        For `diff_cols`, the value to be filled for NA.

    Returns
    -------
    df : DataFrame
        Merged DataFrame
    idx : 2d-array[int] of shape (len(df), len(df_ls))
        The original record index of each input DataFrames for merged
        DataFrame. -1 indicates not existence.
    """
    if not df_ls:
        return pd.DataFrame({key_col: []}), np.ones((0, 0), dtype=int)
    df = df_ls[0]
    if len(df_ls) == 1:
        return df, np.ones((len(df), 1), dtype=int)
    if not key_col:
        key_col = df.index.name
    elif key_col in df.columns:
        df.dropna(subset=[key_col], inplace=True)
        df.drop_duplicates(subset=[key_col], inplace=True)
        df.set_index(key_col, inplace=True)
    df.rename(
        columns={col: f"{col}_lib1" for col in df.columns.intersection(diff_cols)},
        inplace=True,
    )
    index_ls = [df.index.tolist()]
    for i, ref_df in enumerate(df_ls[1:]):
        if key_col in ref_df.columns:
            ref_df.dropna(subset=[key_col], inplace=True)
            ref_df.drop_duplicates(subset=[key_col], inplace=True)
            ref_df.set_index(key_col, inplace=True)
        index_ls.append(ref_df.index.tolist())
        ref_df.rename(
            columns={
                col: f"{col}_lib{i + 2}"
                for col in ref_df.columns.intersection(diff_cols)
            },
            inplace=True,
        )
        append_cols = ref_df.columns.difference(df.columns).tolist()
        df = df.merge(ref_df[append_cols], on=key_col, how="outer", copy=False)
        df.fillna(ref_df, inplace=True)

    idx = np.full((len(df), len(df_ls)), -1, dtype=int)
    key_d = {key: i for i, key in enumerate(df.index)}
    for j, index in enumerate(index_ls):
        idx[[key_d[key] for key in index], j] = np.arange(len(index), dtype=int)
    df.fillna(
        {
            f"{col}_lib{i + 1}": fill_diff
            for i in range(len(df_ls))
            for col in diff_cols
        },
        inplace=True,
        downcast="infer",
    )
    df.reset_index(inplace=True)
    if idx_col:
        s = (idx > -1).tobytes().hex()[1::2]
        df[idx_col] = [
            s[i:(i + idx.shape[1])] for i in range(0, len(s), idx.shape[1])
        ]
    return df, idx


def df_fillcol(df, fill_col, ref_col, fill_func, use_index=False):
    """
    Add / fill a column of DataFrame with converting values from
    another column.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame to add / fill a column.
    fill_col : str
        The column name to add / fill.
    ref_col : str
        The column name to give values to convertion function.
    fill_func : Callable[Any, [Any]]
        The convertion function from reference value to filling value.
        If `use_index` is False, the input is each element in
        `df[ref_col]`.
        If `use_index` is False, the input are two elements from
        `df.index` and `df[ref_col]`
    use_index : bool
        Whether use index column as another input of convertion
        function.

    Returns
    -------
    df : DataFrame
        DataFrame with the specific column filled.
    """
    if fill_col not in df.columns:
        idx_fill = slice(None)
    else:
        idx_fill = df[fill_col].isna().values
    if not use_index:
        values_fill = [fill_func(val) for val in df.loc[idx_fill, ref_col]]
    elif ref_col:
        values_fill = [
            fill_func(key, val) for key, val in df.loc[idx_fill, ref_col].items()
        ]
    else:
        values_fill = [fill_func(val) for val in df.index[idx_fill]]
    if values_fill:
        df.loc[idx_fill, fill_col] = values_fill
    return df


if __name__ == "__main__":
    from rdkit import RDLogger
    import doctest

    lg = RDLogger.logger()
    lg.setLevel(4)
    doctest.testmod()
