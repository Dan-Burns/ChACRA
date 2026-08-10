# -*- coding: utf-8 -*-
"""
Core analysis utilities for ChACRA.

Contains contact-name parsing helpers, set-intersection utilities, and
dictionary helpers used throughout the analysis pipeline.  This module has
no OpenMM or simulation dependencies and is safe to import in any environment.

Simulation setup helpers live in ``chacra.simulation``.
Run configuration management lives in ``chacra.run_config``.
"""

import re
import warnings

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)  # pdbfixer


def make_contact_frequency_dictionary(freq_files: list) -> dict:
    """
    Deprecated in favor of ``make_contact_dataframe()``.

    Go through a list of getcontacts frequency files and collect all contact
    frequencies for each replica into a dictionary.

    Parameters
    ----------
    freq_files : list
        List of paths to each contact frequency file, presorted.

    Returns
    -------
    dict
        Keys are contact labels (``'CH:RES:NUM-CH:RES:NUM'``), values are
        lists of frequencies across replicas.
    """
    contact_dictionary = {}

    regex = r"\w:\w+:\d+\s+\w:\w+:\d+"
    for i, file in enumerate(freq_files):
        with open(file, "r") as freqs:
            for line in freqs.readlines():
                if re.search(regex, line):
                    line = line.strip()
                    first, second, num_str = line.split()
                    label = first + "-" + second

                    if label not in contact_dictionary.keys():
                        contact_dictionary[label] = [0 for n in range(i)]
                        contact_dictionary[label].append(float(num_str))
                    else:
                        contact_dictionary[label].append(float(num_str))

        # Extend all lists before opening the next freq_file
        for key in contact_dictionary.keys():
            if i > 0 and len(contact_dictionary[key]) != i + 1:
                length = len(contact_dictionary[key])
                extend = (i + 1) - length
                contact_dictionary[key].extend([0 for n in range(extend)])

    return contact_dictionary


def sort_dictionary_values(dictionary: dict) -> dict:
    """Sort a flat dictionary by its values in descending order."""
    return dict(sorted(dictionary.items(), key=lambda item: -item[1]))


def parse_id(contact: str) -> dict:
    """
    Parse a contact name into its component residue descriptors.

    Parameters
    ----------
    contact : str
        Contact name in the form ``'CHAIN:RESNAME:RESID-CHAIN:RESNAME:RESID'``.

    Returns
    -------
    dict with keys: chaina, resna, resida, chainb, resnb, residb
    """
    chaina, resna, resida, chainb, resnb, residb = re.split(":|-", contact)

    ### for combined contact data, the prepended name needs to be removed from
    ### chain a
    ##### This might break something if multiple contacts
    #### are going into the keys of another dictionary because
    #### duplicate names will be overwritten.
    ## shouldn't be a problem for averaging functions because combined data
    ## will be produced from pre-averaged data
    ## to_heatmap() will not give correct results as is - need to prepare
    ## the data with original names for that....

    if "_" in chaina:
        chaina = chaina.split("_")[1]

    return {
        "chaina": chaina,
        "resna": resna,
        "resida": resida,
        "chainb": chainb,
        "resnb": resnb,
        "residb": residb,
    }


def split_id(contact: str) -> dict:
    """
    Split a contact name into its two residue parts.

    Returns
    -------
    dict with keys ``'resa'`` and ``'resb'``, each containing
    ``'CHAIN:RESNAME:RESID'``.
    """
    resa, resb = re.split("-", contact)
    return {"resa": resa, "resb": resb}


def multi_intersection(lists: list[list], cutoff: float | int | None = None,
                       verbose: bool = False) -> list:
    """
    Return the intersection of values across a collection of lists.

    Parameters
    ----------
    lists : list of lists
        The lists of values to find shared elements across.

    cutoff : float or int or None
        If not None, restrict the intersection to a subset of lists:
        - float < 1 : only include lists whose length is at least
          ``cutoff`` × the longest list length.
        - int > 1   : only include lists longer than ``cutoff``.

    verbose : bool
        If True, print the initial and final list counts.

    Returns
    -------
    list
        Sorted intersection of values.
    """
    initial = len(lists)
    if cutoff is not None and cutoff < 1:
        longest_len = max([len(data) for data in lists])
        lists = [data for data in lists if len(data) > longest_len * cutoff]
    elif cutoff is not None and cutoff > 1:
        lists = [data for data in lists if len(data) > cutoff]

    final = len(lists)
    set1 = set(lists[0])
    setlist = [set(data) for data in lists[1:]]
    if verbose:
        print(f"n lists initial: {initial} \nn lists final: {final}")
    return sorted(list(set1.intersection(*setlist)))


def sort_nested_dict(d: dict) -> dict:
    """
    Sort a nested dictionary whose inner keys are residue identifiers of the
    form ``'CHAIN:RESNAME:RESID'``, ordering first by chain then by residue
    number.
    """
    sorted_dict = {}
    for outer_key, nested_dict in d.items():
        sorted_keys = sorted(
            nested_dict.keys(),
            key=lambda x: (x.split(":")[0], int(x.split(":")[-1])),
        )
        sorted_dict[outer_key] = {key: nested_dict[key] for key in sorted_keys}
    return sorted_dict
