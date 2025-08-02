import MDAnalysis as mda
import nglview as nv
import numpy as np

from chacra.utils import parse_id
from chacra.visualize.colors import chacra_colors, hex_to_RGB
from chacra.visualize.pymol import get_contact_data

# class to easily depict ChACRA data in nglview
# if you want more than one atom at a time use np.isin
# np.where(np.isin(u.atoms.resids,[2,1]) & (u.atoms.segids == 'G') & (u.atoms.names == 'CA'))
# nglview reindex from 0 if you take a subset of atoms from the original universe
# use this line to get the new 0 based indices for a subselection if the entire protein is not in nv.view
# np.where(np.in1d(all_indices,indices_to_select))[0]


# offer cylinder width option to depict most sensitive contact with largest cylinder


def get_midpoint(a, b):
    """
    Get the midpoint of two vectors.

    Parameters
    ----------
    a,b : np.array
    Returns
    -------
    np.array
    """
    return np.mean([a, b], axis=0)


def get_contact_indices(contact, u, ca_only=True):
    """
    Get the atom or residue indices from an mda.Universe/ structure

    Parameters
    ----------
    contact : string
        Contact name.

    ca_only : bool
        Return the atom indices of the c-alpha atoms.
        Default is True. If you want the residue index, set ca_only=False

    Returns
    np.array of atom or residue indices
    """
    c = parse_id(contact)

    indices = []
    for res in ["a", "b"]:
        if ca_only:
            indices.append(
                np.where(
                    (u.atoms.resnums == int(c[f"resid{res}"]))
                    & (u.atoms.segids == c[f"chain{res}"])
                    & (u.atoms.names == "CA")
                )[0][0]
            )
        else:
            indices.append(
                np.where(
                    (u.residues.resids == int(c[f"resid{res}"]))
                    & (u.residues.segids == c[f"chain{res}"])
                )[0][0]
            )
    return np.asarray(indices)


def get_positions(atom_indices, u):
    """
    atom_indices : list
        list of atom indices

    u : mda.Universe


    """
    positions = []
    for atom_index in atom_indices:
        positions.append(u.atoms[atom_index].position)
    return positions


def draw_line(
    contact,
    u,
    view,
    width=0.3,
    arrows=False,
    contact_data=None,
    variable_width=False,
    width_coef=1.3,
    exp=2,
    color=None,
):
    """
    Draw a line between contacting residues with nglview

    Parameters:
    ----------
    contact : string
        Contact name.

    view : nglview.widget.NGLWidget

    width : float
        The width of the cylinder between the residues

    arrows : bool
        Draw arrowheads to indicate whether the contact has a tendency to form or break across temperature.
        If True, must provide contact_data.

    contact_data : dictionary from contacts_to_pymol.get_contact_data()
        Dictionary containing the information to depicting correct arrow direction.

    variable_width : bool
        Vary the width of the contact line according to the loading scores.
        (loading_score * width_coef) ** exp

    width_coef : float
        value by which to multiply a contact's loading score to alter the width of the line.

    exp : int or float
        value by which to rais the loading_score*width_coef.

    color : list or array or string
        A 3d rgb vector or hex string that can be converted to rgb.
        If None, contact_data is expected.

    Returns
    -------
    A view is updated with lines connecting contacting residues

    #TODO speed it up - difficult to speed up because you have to make a call to shape for each line.
    """
    # get the atom indices for specifying the end of each line.
    atom_indices = get_contact_indices(contact, u)
    # get the 3d position vectors
    positions = get_positions(atom_indices, u)

    # if a hex color is provided
    if color is not None and type(color) == str:
        color = np.array(hex_to_RGB(color)) / 256
    elif contact_data is None and color is None:
        color = [1, 0, 0]

    if arrows == True:
        midpoint = get_midpoint(positions[0], positions[1])
        if color is None:
            color = chacra_colors[contact_data[contact]["top_pc"] - 1]
            # color has to be normalized to work in nglview
            color = np.asarray(hex_to_RGB(color)) / 256
        if variable_width == True:
            width = (contact_data[contact]["loading_score"] * width_coef) ** 2
        if contact_data[contact]["slope"] < 0:
            # make arrows point away from midpoint
            view.shape.add_arrow(midpoint, positions[0], color, width)
            view.shape.add_arrow(midpoint, positions[1], color, width)
        else:
            view.shape.add_arrow(positions[0], midpoint, color, width)
            view.shape.add_arrow(positions[1], midpoint, color, width)
    else:
        if color is None:
            color = chacra_colors[contact_data[contact]["top_pc"] - 1]
            color = np.asarray(hex_to_RGB(color)) / 256
        if variable_width == True:
            width = (contact_data[contact]["loading_score"] * width_coef) ** 2

        # make arrows point toward midpoint (increasing contact_frequency)
        view.shape.add_cylinder(positions[0], positions[1], color, width)


class Visualizer:
    """
    structure : str or mda.Universe
        Path to structure file or mda.Universe corresponding to the

    cpca : ContactPCA

    cont : ContactFrequencies

    Calling Visualizer.view will reset the custom shapes (lines connecting contacting residues), so call
    Visualizer.view first and then show_chacras and the existing representation will be updated.

    """

    def __init__(
        self, structure, cpca, cont, protein_color="silver", bkgrnd="black"
    ):
        if type(structure) == mda.core.universe.Universe:
            self.u = structure
        else:
            self.u = mda.Universe(structure)

        self.view = nv.show_mdanalysis(self.u)
        self.cpca = cpca
        self.cont = cont
        self.view.background = bkgrnd
        self.protein_color = protein_color
        self.view.add_representation(
            "cartoon", selection="protein", color=self.protein_color
        )

    def view(self):
        return self.view

    def show_chacras(
        self,
        pc_range,
        cutoff=0.6,
        ca_spheres=True,
        arrows=True,
        sphere_scale=0.8,
        variable_line_width=False,
        line_width=0.3,
        clear_representation=False,
    ):
        """
        pc_range : tuple of int
            PC/chacra range to depict top contacts for (inclusive) e.g. (1,7) to see all chacras between 1 and 7.

        variable_line_width : bool
            Whether or not to have the lines connecting contacts vary in width based on the loading score

        line_width : float
            The width of the contact lines.
            If variable_line_width == True, this is a coefficient to multiply the normalized loading score against
            to determine each line's width.

            TODO: make exponential function - not enough difference in linewidth as is
        """
        # contact_data will be in order of the highest loading score last.  That should
        # be the color of sphere that is depicted.  Dictionary of resindex keys (unique) and color values
        # sort the dictionary and take a list of all the ones with the same color before calling the 'spacefill' representation
        if clear_representation == True:
            self.view.clear_representations()
            self.view.add_representation(
                "cartoon", selection="protein", color=self.protein_color
            )

        min_pc = pc_range[0]
        max_pc = pc_range[1]
        top_contacts = []
        # start taking above a loading score cutoff of 0.6
        for i in range(min_pc, max_pc + 1):
            top_contacts.extend(
                (
                    self.cpca.sorted_norm_loadings(i).loc[
                        self.cpca.sorted_norm_loadings(i)[f"PC{i}"] > cutoff
                    ]
                ).index
            )

        top_contacts = list(set(top_contacts))
        contact_data = get_contact_data(
            top_contacts, self.cont, self.cpca, pc_range=pc_range
        )

        # collect atom id keys and color values
        ca_colors = {}

        for contact in contact_data:
            if ca_spheres == True:
                # get the atom indices and corresponding top scoring pc color
                # collecting them in the order they occur in contact_data will make the last color chosen correspond to
                # the highest scoring pc for that residue
                atom_indices = get_contact_indices(contact, self.u)
                ca_colors[atom_indices[0]] = chacra_colors[
                    contact_data[contact]["top_pc"] - 1
                ]
                ca_colors[atom_indices[1]] = chacra_colors[
                    contact_data[contact]["top_pc"] - 1
                ]

            draw_line(
                contact,
                self.u,
                self.view,
                width=line_width,
                arrows=arrows,
                variable_width=variable_line_width,
                width_coef=line_width,
                contact_data=contact_data,
            )

        if ca_spheres == True:
            # get the set of colors in ca_colors and all the atom indices that should have that color in a comma separated string
            # for nglview selection input
            sphere_colors = {
                value: ",".join(
                    [
                        str(atom_index)
                        for atom_index in ca_colors.keys()
                        if ca_colors[atom_index] == value
                    ]
                )
                for value in set(ca_colors.values())
            }
            # depict them on the structure one pc at a time
            for color, indices in sphere_colors.items():
                # color requires "#" and 6 string characters
                self.view.add_representation(
                    "spacefill",
                    selection=f"@{indices}",
                    color=color[:7],
                    radius_scale=sphere_scale,
                )

    # TODO add network visualization
