import os
import warnings
from collections import defaultdict

import MDAnalysis as mda
import numpy as np
import pandas as pd
import tqdm
from MDAnalysis.lib.util import convert_aa_code

from .utils import parse_id

warnings.filterwarnings("ignore", message="Biopython*")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)  # pdbfixer


def _longest_common_contiguous_substring(strs: list[str]) -> str:
    """
    Find the longest contiguous substring common to all strings in `strs`.
    """
    if not strs:
        return ""
    if len(strs) == 1:
        return strs[0]
    shortest = min(strs, key=len)
    n = len(shortest)
    for length in range(n, 0, -1):
        for start in range(n - length + 1):
            sub = shortest[start : start + length]
            if all(sub in s for s in strs):
                return sub
    return ""


def kabsch_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns the 3x3 rotation matrix R that minimizes RMSD between
    centered coordinate arrays `a` (mobile) and `b` (reference).
    Satisfies b ~ a @ R.T (i.e. b = R @ a).
    Always returns a proper rotation with det(R) = +1.
    """
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    diag = np.ones(3)
    if d < 0:
        diag[2] = -1.0
    return Vt.T @ np.diag(diag) @ U.T


def has_only_identical_subunits(u: mda.Universe) -> bool:
    """
    Check if all subunits in the universe are identical.

    Parameters
    ----------
    u : mda.Universe
        The MDAnalysis universe to check.

    Returns
    -------
    bool
        True if all subunits are identical, False otherwise.
    """
    if len(u.segments) == 0:
        return True
    seqs = [tuple(seg.residues.resnames) for seg in u.segments]
    return all(s == seqs[0] for s in seqs)


def seg_to_chain(u: mda.Universe) -> dict[int, str]:
    """
    Returns the seg id to chain id mapping.

    Parameters
    ----------
    u : mda.Universe

    Returns
    -------
    dict
        {int(segid): str(chainid)}
    """
    mapping = {}
    for i, seg in enumerate(u.segments):
        chains = [c for c in np.unique(seg.atoms.chainIDs) if c]
        mapping[i] = chains[0] if len(chains) > 0 else seg.segid
    return mapping


def chain_to_seg(u: mda.Universe) -> dict[str, int]:
    """
    Returns the chain id to seg id mapping.

    Parameters
    ----------
    u : mda.Universe

    Returns
    -------
    dict
        {str(chainid): int(segid)}
    """
    mapping = {}
    for i, seg in enumerate(u.segments):
        chains = [c for c in np.unique(seg.atoms.chainIDs) if c]
        chain_id = chains[0] if len(chains) > 0 else seg.segid
        mapping[chain_id] = i
    return mapping


def find_identical_subunits(u: mda.Universe) -> dict[int, list[int]]:
    """
    Groups segments (subunits) by identical residue sequences.

    Parameters
    ----------
    u : mda.Universe
        Protein structure.

    Returns
    -------
    dict[int, list[int]]
        Dictionary where keys are group indices (0, 1, ...) and values are
        lists of segment indices that share identical amino acid sequences.
    """
    seq_groups = defaultdict(list)
    for i, seg in enumerate(u.segments):
        seq = tuple(seg.residues.resnames)
        seq_groups[seq].append(i)
    return {idx: group for idx, group in enumerate(seq_groups.values())}


def same_list_membership(d: dict[int, list[int]], a: int, b: int) -> bool:
    """
    Check if both `a` and `b` appear in the same list of the dictionary.

    Returns True if they are in the same list, False if in different lists
    or only one is found.
    """
    for group in d.values():
        if a in group and b in group:
            return True
    return False


def align_principal_axes_to_global_frame(u: mda.Universe) -> np.ndarray:
    """
    Reorients the Universe in place to align with the global axes.
    Ensures determinant is +1 to prevent improper rotation (reflection).
    """
    universe_com = u.atoms.center_of_mass()
    coords = u.atoms.positions - universe_com
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    local_frame = eigvecs  # columns are basis vectors
    global_frame = np.eye(3)

    R = global_frame @ local_frame.T
    if np.linalg.det(R) < 0:
        R[2, :] *= -1.0

    shifted = u.atoms.positions - universe_com
    rotated = shifted @ R.T
    u.atoms.positions = rotated
    return R


def get_long_axis(subunit: mda.AtomGroup) -> np.ndarray:
    """Get the primary principal axis for an atom group."""
    coords = subunit.positions - subunit.center_of_mass()
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, np.argmax(eigvals)]
    return long_axis / np.linalg.norm(long_axis)


def align_mobile_to_ref(mobile: mda.AtomGroup, ref: mda.AtomGroup):
    """
    Aligns the long axis of mobile atom group to ref atom group's in place.
    """
    ref_com = ref.center_of_mass()
    mob_com = mobile.center_of_mass()

    ref_axis = get_long_axis(ref)
    mob_axis = get_long_axis(mobile)

    v = np.cross(mob_axis, ref_axis)
    c = np.dot(mob_axis, ref_axis)
    s = np.linalg.norm(v)

    if s == 0:
        if c > 0:
            R = np.eye(3)
        else:
            ortho = np.eye(3)[np.argmin(np.abs(mob_axis))]
            v = np.cross(mob_axis, ortho)
            v /= np.linalg.norm(v)
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + 2 * K @ K
    else:
        v /= s
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s**2))

    shifted = mobile.positions - mob_com
    rotated = shifted @ R.T
    aligned = rotated + ref_com
    mobile.positions = aligned


def get_rotation_matrix(
    chaina: mda.AtomGroup,
    chainb: mda.AtomGroup,
) -> np.ndarray:
    """
    Get the rotation matrix that aligns chaina with chainb.
    Centering is performed prior to alignment to ensure origin invariance.
    """
    com_a = chaina.center_of_mass()
    com_b = chainb.center_of_mass()
    return kabsch_rotation(chaina.positions - com_a, chainb.positions - com_b)


def get_all_rotations(
    u: mda.Universe,
    identical_subunits: dict,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Get rotation matrices for each subunit pair.
    """
    segids = list(range(len(u.segments)))
    seg_chain = seg_to_chain(u)
    rotations = {
        seg_chain[segid1]: {
            seg_chain[segid2]: np.eye(3) for segid2 in segids if segid2 != segid1
        }
        for segid1 in segids
    }
    for sega in segids:
        for segb in segids:
            if sega == segb:
                continue
            ca_a = u.segments[sega].atoms.select_atoms("name CA")
            ca_b = u.segments[segb].atoms.select_atoms("name CA")
            if len(ca_a) == len(ca_b) and len(ca_a) > 0:
                rotations[seg_chain[sega]][seg_chain[segb]] = get_rotation_matrix(
                    ca_a, ca_b
                )
    return rotations


def get_equivalent_interactions(
    u_or_rotations: mda.Universe | dict,
    identical_subunits: dict[int, list[int]],
    chain_seg_dict: dict[str, int],
    representative_chains: list[str] | None = None,
    rmsd_cutoff: float = 1.5,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """
    Map pairs of subunits to other pairs with the same spatial relationship.

    Uses coordinate superposition and CA-RMSD with `rmsd_cutoff` to identify
    truly equivalent subunit interactions. Subunit pairs with distinct
    spatial environments (e.g. in non-identically symmetric homotetramers or
    linear filaments) are kept in distinct equivalence classes.

    Parameters
    ----------
    u_or_rotations : mda.Universe | dict
        The MDAnalysis Universe (preferred) or rotation matrices dictionary.
    identical_subunits : dict[int, list[int]]
        Keys are group IDs, values are lists of segment IDs with identical sequences.
    chain_seg_dict : dict[str, int]
        Mapping from chain ID string to segment ID integer.
    representative_chains : list[str] | None, optional
        If provided, only returns equivalence classes whose representative key
        starts with a chain in `representative_chains`.
    rmsd_cutoff : float, default=1.5
        RMSD cutoff in Angstroms for determining whether two pairwise subunit
        geometries are spatially equivalent.

    Returns
    -------
    dict[tuple[str, str], list[tuple[str, str]]]
        Mapping from canonical representative chain pair (tuple) to the list
        of equivalent directed chain pairs.
    """
    if isinstance(u_or_rotations, mda.Universe):
        u = u_or_rotations
    else:
        raise TypeError(
            "get_equivalent_interactions requires an MDAnalysis Universe "
            "as first argument."
        )

    seg_chain = {v: k for k, v in chain_seg_dict.items()}
    seg_to_group = {}
    for grp, segs in identical_subunits.items():
        for s in segs:
            seg_to_group[s] = grp

    # Compute coordinates of subunit B in the standard reference frame of group(A)
    coords_map = {}
    for sega_idx, sega in enumerate(u.segments):
        grp_a = seg_to_group[sega_idx]
        ref_seg_idx = identical_subunits[grp_a][0]
        ref_ca = u.segments[ref_seg_idx].atoms.select_atoms("name CA")
        ca_a = sega.atoms.select_atoms("name CA")

        com_a = ca_a.center_of_mass()
        com_ref = ref_ca.center_of_mass()
        R = kabsch_rotation(ca_a.positions - com_a, ref_ca.positions - com_ref)

        cha = seg_chain[sega_idx]
        for segb_idx, segb in enumerate(u.segments):
            chb = seg_chain[segb_idx]
            ca_b = segb.atoms.select_atoms("name CA")
            b_coords = (ca_b.positions - com_a) @ R.T + com_ref
            coords_map[(cha, chb)] = b_coords

    all_pairs = list(coords_map.keys())
    visited = set()
    classes = []

    for p1 in all_pairs:
        if p1 in visited:
            continue
        eq_class = [p1]
        visited.add(p1)
        cha1, chb1 = p1
        grp_a1 = seg_to_group[chain_seg_dict[cha1]]
        grp_b1 = seg_to_group[chain_seg_dict[chb1]]

        for p2 in all_pairs:
            if p2 in visited:
                continue
            cha2, chb2 = p2
            grp_a2 = seg_to_group[chain_seg_dict[cha2]]
            grp_b2 = seg_to_group[chain_seg_dict[chb2]]

            if grp_a1 != grp_a2 or grp_b1 != grp_b2:
                continue

            if cha1 == chb1 and cha2 == chb2:
                # Intra-chain interactions of identical subunit types
                eq_class.append(p2)
                visited.add(p2)
            elif cha1 != chb1 and cha2 != chb2:
                # Inter-chain interactions: check CA-RMSD
                c1 = coords_map[p1]
                c2 = coords_map[p2]
                rmsd = np.sqrt(np.mean((c1 - c2) ** 2))
                if rmsd <= rmsd_cutoff:
                    eq_class.append(p2)
                    visited.add(p2)

        classes.append(eq_class)

    result = {}
    for eq_class in classes:
        eq_class_sorted = sorted(eq_class)
        if representative_chains is not None:
            rep_candidates = [
                p for p in eq_class_sorted if p[0] in representative_chains
            ]
            if not rep_candidates:
                continue
            key = rep_candidates[0]
        else:
            key = eq_class_sorted[0]
        result[key] = eq_class_sorted

    return result


def validate_group_memberships(L: list, D: dict) -> bool:
    """
    Ensure that each element of L occurs only once in
    one list value of D for all lists in D.
    """
    group_counts = {k: 0 for k in D}
    element_to_group = {}
    for k, values in D.items():
        for v in values:
            element_to_group[v] = k

    for item in L:
        if item not in element_to_group:
            return False
        group_counts[element_to_group[item]] += 1

    return all(count == 1 for count in group_counts.values())


def get_chain_group(chain, identical_subunits):
    """
    Return the identical subunit group that the subunit is in.
    """
    for group in identical_subunits:
        if chain in identical_subunits[group]:
            return group


def alphabetize(contact_name: str) -> str:
    """Ensure contact name is alphabetically ordered."""
    a, b = contact_name.split("-")
    if a < b:
        return contact_name
    else:
        return f"{b}-{a}"


def map_chains(chain_pair: tuple, pairs_list: list) -> dict:
    """
    Maps alphabetically ordered chains to their relationship order with other
    equivalently interacting chains.
    """
    for pair in pairs_list:
        if chain_pair[0] in pair and chain_pair[1] in pair:
            if chain_pair[0] == pair[0]:
                return {0: "a", 1: "b"}
            else:
                return {0: "b", 1: "a"}
    return {0: "a", 1: "b"}


def make_equivalent_contact_names(
    resinfo: dict,
    equivalent_interactions: dict[tuple[str, str], list[tuple[str, str]]],
) -> list[str]:
    """
    Uses orientation-aware equivalent_interactions to generate the names
    for all contacts equivalent to the given contact.

    Parameters
    ----------
    resinfo : dict
        Contact info dictionary from parse_id.
    equivalent_interactions : dict
        Mapping from canonical chain pairs to equivalent directed pairs.

    Returns
    -------
    list[str]
        Sorted list of unique equivalent contact names.
    """
    chaina, chainb = resinfo["chaina"], resinfo["chainb"]
    resna, resida = resinfo["resna"], resinfo["resida"]
    resnb, residb = resinfo["resnb"], resinfo["residb"]

    matched_key = None
    role_order = None

    for key, pair_list in equivalent_interactions.items():
        for p0, p1 in pair_list:
            if (chaina, chainb) == (p0, p1):
                matched_key = key
                role_order = (resna, resida, resnb, residb)
                break
            elif (chaina, chainb) == (p1, p0):
                matched_key = key
                role_order = (resnb, residb, resna, resida)
                break
        if matched_key is not None:
            break

    if matched_key is None:
        return [alphabetize(f"{chaina}:{resna}:{resida}-{chainb}:{resnb}:{residb}")]

    r1_n, r1_id, r2_n, r2_id = role_order
    contacts = []
    for p0, p1 in equivalent_interactions[matched_key]:
        contacts.append(alphabetize(f"{p0}:{r1_n}:{r1_id}-{p1}:{r2_n}:{r2_id}"))

    return sorted(set(contacts))


def get_representative_name(
    resinfo: dict,
    equivalent_interactions: dict[tuple[str, str], list[tuple[str, str]]],
) -> str:
    """
    Provide contact info and get the canonical representative contact name.
    """
    chaina, chainb = resinfo["chaina"], resinfo["chainb"]
    resna, resida = resinfo["resna"], resinfo["resida"]
    resnb, residb = resinfo["resnb"], resinfo["residb"]

    matched_key = None
    role_order = None

    for key, pair_list in equivalent_interactions.items():
        for p0, p1 in pair_list:
            if (chaina, chainb) == (p0, p1):
                matched_key = key
                role_order = (resna, resida, resnb, residb)
                break
            elif (chaina, chainb) == (p1, p0):
                matched_key = key
                role_order = (resnb, residb, resna, resida)
                break
        if matched_key is not None:
            break

    if matched_key is None:
        return alphabetize(f"{chaina}:{resna}:{resida}-{chainb}:{resnb}:{residb}")

    rep_a, rep_b = matched_key
    r1_n, r1_id, r2_n, r2_id = role_order
    return alphabetize(f"{rep_a}:{r1_n}:{r1_id}-{rep_b}:{r2_n}:{r2_id}")


def get_pair_distance(sel1: str, sel2: str, u: mda.Universe) -> float:
    """Return the distance between two atom selections."""
    a = u.select_atoms(sel1).positions[0]
    b = u.select_atoms(sel2).positions[0]
    return float(np.linalg.norm(a - b))


def trim_multimer_termini(
    structure: str | os.PathLike | mda.Universe,
    df: pd.DataFrame,
    chain_groups: list[list[str]] | None = None,
    output_structure: str | os.PathLike | None = None,
) -> tuple[mda.Universe, pd.DataFrame]:
    """
    Trim ragged termini across homomultimer subunits and synchronize contact data.

    Finds the longest common contiguous substring of amino acid residues across
    identical subunits (or specified chain groups), removes non-common
    terminal residues from the MDAnalysis Universe, and drops any contact columns
    from the frequency DataFrame that involve the discarded residues.

    Parameters
    ----------
    structure : str | os.PathLike | mda.Universe
        Path to PDB file or an existing MDAnalysis Universe.
    df : pd.DataFrame
        Contact frequency DataFrame.
    chain_groups : list[list[str]], optional
        List of lists of chain IDs to trim together (e.g. [['A', 'B'], ['C', 'D']]).
        If None, chains with high sequence similarity (longest common substring >= 50%
        of chain length) are automatically grouped as homosubunits.
    output_structure : str | os.PathLike, optional
        If provided, writes the trimmed Universe coordinates to this file path.

    Returns
    -------
    tuple[mda.Universe, pd.DataFrame]
        A tuple of (trimmed_universe, trimmed_dataframe).
    """
    if isinstance(structure, mda.Universe):
        u = structure
    else:
        u = mda.Universe(str(structure))

    protein = u.select_atoms("protein")
    if len(protein) > 0:
        u = mda.Merge(protein)

    seg_chain = seg_to_chain(u)

    chain_seqs = {}
    chain_residues = {}
    for i, seg in enumerate(u.segments):
        ch = seg_chain[i]
        seq = "".join(convert_aa_code(r.resname) for r in seg.residues)
        chain_seqs[ch] = seq
        chain_residues[ch] = list(seg.residues)

    all_chains = list(chain_seqs.keys())

    if chain_groups is None:
        chain_groups = []
        visited = set()
        for ch1 in all_chains:
            if ch1 in visited:
                continue
            group = [ch1]
            visited.add(ch1)
            s1 = chain_seqs[ch1]
            for ch2 in all_chains:
                if ch2 in visited:
                    continue
                s2 = chain_seqs[ch2]
                lcs = _longest_common_contiguous_substring([s1, s2])
                min_len = min(len(s1), len(s2))
                if min_len > 0 and len(lcs) / min_len >= 0.5:
                    group.append(ch2)
                    visited.add(ch2)
            chain_groups.append(group)

    discarded_residues = set()
    kept_atoms_sel = []

    for group in chain_groups:
        if len(group) == 1:
            ch = group[0]
            kept_atoms_sel.append(f"chainID {ch}")
            continue

        group_seqs = [chain_seqs[ch] for ch in group]
        lcs = _longest_common_contiguous_substring(group_seqs)

        if not lcs:
            warnings.warn(
                f"No common sequence found for chains {group}. "
                "Keeping all residues.",
                stacklevel=2,
            )
            for ch in group:
                kept_atoms_sel.append(f"chainID {ch}")
            continue

        for ch in group:
            seq = chain_seqs[ch]
            res_list = chain_residues[ch]
            start_idx = seq.find(lcs)
            end_idx = start_idx + len(lcs)

            kept_res = res_list[start_idx:end_idx]
            disc_res = res_list[:start_idx] + res_list[end_idx:]

            for r in disc_res:
                discarded_residues.add((ch, str(r.resid)))

            kept_resids = " ".join(str(r.resid) for r in kept_res)
            if kept_resids:
                kept_atoms_sel.append(f"(chainID {ch} and resid {kept_resids})")

    full_sel = " or ".join(kept_atoms_sel)
    selected_atoms = u.select_atoms(full_sel)
    trimmed_u = mda.Merge(selected_atoms)

    if output_structure is not None:
        trimmed_u.atoms.write(str(output_structure))

    if discarded_residues and len(df.columns) > 0:
        cols_to_drop = []
        for col in df.columns:
            info = parse_id(col)
            if (info["chaina"], str(info["resida"])) in discarded_residues or (
                info["chainb"],
                str(info["residb"]),
            ) in discarded_residues:
                cols_to_drop.append(col)
        trimmed_df = df.drop(columns=cols_to_drop)
    else:
        trimmed_df = df.copy()

    return trimmed_u, trimmed_df


def average_multimer(
    structure: str | os.PathLike | mda.Universe,
    df: pd.DataFrame,
    representative_chains: list[str] | None = None,
    return_stdev: bool = False,
    rmsd_cutoff: float = 1.5,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes a structure, contact frequency DataFrame, and representative chain(s)
    to compute averaged contact frequencies across equivalent interfaces.

    Parameters
    ----------
    structure : str | os.PathLike | mda.Universe
        Path to PDB file or existing MDAnalysis Universe.
    df : pd.DataFrame
        Contact frequency dataframe to average.
    representative_chains : list[str] | None, optional
        List of pdb chain IDs to use for depicting averaged contacts.
        If None, automatically defaults to the first chain for homomultimers
        or one representative per subunit type for heteromultimers.
    return_stdev : bool, default=False
        If True, returns (averaged_df, stdev_df).
    rmsd_cutoff : float, default=1.5
        RMSD cutoff in Angstroms for determining whether two pairwise subunit
        geometries are spatially equivalent.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
    """
    if isinstance(structure, mda.Universe):
        u = structure
    else:
        u = mda.Universe(str(structure))
    u = mda.Merge(u.select_atoms("protein"))

    identical_subunits = find_identical_subunits(u)
    chain_seg = chain_to_seg(u)
    seg_chain = seg_to_chain(u)

    identical_chains = {
        group: [seg_chain[el] for el in val]
        for group, val in identical_subunits.items()
    }

    if has_only_identical_subunits(u):
        if representative_chains is None:
            representative_chains = [seg_chain[0]]
    else:
        if len(identical_subunits) > 1:
            test_len = len(list(identical_subunits.values())[0])
            if not np.all(
                [len(val) == test_len for val in identical_subunits.values()]
            ):
                print(
                    "Can't average a heteromultimer with an unequal number "
                    "of each type of subunit."
                )
                return
            if representative_chains is None:
                representative_chains = [
                    seg_chain[val[0]] for val in identical_subunits.values()
                ]
            elif len(representative_chains) != len(identical_subunits):
                print(
                    "You need to specify 1 chain id for each type of subunit "
                    "in the representative_chains argument."
                )
                return
            elif not validate_group_memberships(
                representative_chains, identical_chains
            ):
                print(
                    "The representative chains don't come from unique sets "
                    "of subunits in the identical_subunits dictionary."
                )
                return

    equivalent_interactions = get_equivalent_interactions(
        u,
        identical_subunits,
        chain_seg,
        representative_chains,
        rmsd_cutoff=rmsd_cutoff,
    )

    remaining_cols = set(df.columns)
    averaged_data = {}
    standard_deviation = {}

    print("Just a moment.\n")
    total_count = len(df.columns)
    with tqdm.tqdm(total=total_count) as progress:
        for col in df.columns:
            if col not in remaining_cols:
                continue

            resinfo = parse_id(col)
            averaged_name = get_representative_name(resinfo, equivalent_interactions)
            to_average_all = make_equivalent_contact_names(
                resinfo, equivalent_interactions
            )

            expected_count = len(to_average_all)
            present_cols = [c for c in to_average_all if c in df.columns]

            if not present_cols:
                remaining_cols.discard(col)
                progress.update(1)
                continue

            sum_series = df[present_cols].sum(axis=1)
            averaged_data[averaged_name] = sum_series / expected_count

            if expected_count == 1:
                standard_deviation[averaged_name] = pd.Series(0.0, index=df.index)
            else:
                n_missing = expected_count - len(present_cols)
                if n_missing == 0:
                    standard_deviation[averaged_name] = df[present_cols].std(
                        axis=1, ddof=1
                    )
                else:
                    vals = df[present_cols].values
                    zeros = np.zeros((vals.shape[0], n_missing))
                    full_vals = np.hstack([vals, zeros])
                    std_vals = np.std(full_vals, axis=1, ddof=1)
                    standard_deviation[averaged_name] = pd.Series(
                        std_vals, index=df.index
                    )

            for c in to_average_all:
                if c in remaining_cols:
                    remaining_cols.remove(c)
                    progress.update(1)

    if return_stdev:
        return pd.DataFrame(averaged_data), pd.DataFrame(standard_deviation)
    else:
        return pd.DataFrame(averaged_data)


def everything_from_averaged(
    averaged_contacts: pd.DataFrame,
    u: mda.Universe,
    representative_chains: list[str] | None = None,
    as_map: bool = False,
    rmsd_cutoff: float = 1.5,
):
    """
    Take the averaged contacts and regenerate the entire protein's contacts
    using this data.

    Parameters
    ----------
    averaged_contacts : pd.DataFrame
        The averaged contact data.
    u : mda.Universe
        The same universe used for averaging contacts.
    representative_chains : list[str] | None, optional
        The list of chain IDs used when generating the averaged contact names.
    as_map : bool, default=False
        Returns a dictionary mapping each averaged contact name to a list of the
        corresponding replicated contact names.
    rmsd_cutoff : float, default=1.5
        RMSD cutoff in Angstroms for determining equivalent interactions.

    Returns
    -------
    pd.DataFrame or dict[str, list[str]]
    """
    print("Collecting some information. One moment.")
    protein = u.select_atoms("protein")
    u = mda.Merge(protein)

    identical_subunits = find_identical_subunits(u)
    chain_seg = chain_to_seg(u)
    seg_chain = seg_to_chain(u)

    if representative_chains is None:
        if has_only_identical_subunits(u):
            representative_chains = [seg_chain[0]]
        else:
            representative_chains = [
                seg_chain[val[0]] for val in identical_subunits.values()
            ]

    equivalent_interactions = get_equivalent_interactions(
        u,
        identical_subunits,
        chain_seg,
        representative_chains,
        rmsd_cutoff=rmsd_cutoff,
    )

    replicated_contacts = {}
    mapped_contacts = {}

    for averaged_contact in tqdm.tqdm(averaged_contacts.columns):
        resinfo = parse_id(averaged_contact)
        remade_contacts = make_equivalent_contact_names(
            resinfo, equivalent_interactions
        )
        for remade_contact in remade_contacts:
            replicated_contacts[remade_contact] = averaged_contacts[averaged_contact]

        mapped_contacts[averaged_contact] = remade_contacts

    if as_map:
        return mapped_contacts
    else:
        return pd.DataFrame(replicated_contacts, index=averaged_contacts.index)
