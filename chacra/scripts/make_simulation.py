from chacra.utils import OMMSetup, fix_pdb
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="""
        Create an OpenMM simulation system xml file. 
        Provide a PDB file with the protein to simulate. If you provide a .cif
         and there are missing residues, they can be naively fixed with 
        pdbfixer.
        """
    )
    parser.add_argument(
        "-s", "--structure", type=str, required=True, 
        help="The structure to simulate."
    )
    parser.add_argument(
        "-f", "--fix", required=False, default="False",
        action="store_true",
        help="""
        Allow pdbfixer to fix the structure. Protonation state will be 
        set to pH 7.0 and non-standard residues will be replaced. Check the 
        output PDB file to ensure the structure is as expected.
        """
    )
    parser.add_argument(
        "-o", "--output", type=str, required=False,  
        help="""
        The directory to save the simulation system xml file and pdb 
        file. 'system' and 'structure' directories will be created here.
        """,
        default="./"
    )
    parser.add_argument(
        "-t", "--temperature", type=float, required=False,
        help="""
        The temperature that the replica exchange solvent will be run at
        in Kelvin.
        """,
        default=290.0
    )
    parser.add_argument(
        "-n", "--name", type=str, required=False,
        help="""
        Provide a name for the simulation and system to prepend to the output."
        """,
        default="chacra_simulation"
    )
    parser.add_argument(
        "-p", "--pressure", type=float, required=False,
        help="""
        Pressure to simulate the system at in bar."
        """,
        default=1
    )

    parser.add_argument(
        "-d", "--timestep", type=int, required=False,
        help="""
        Timestep in femtoseconds."
        """,
        default=4
    )

    parser.add_argument(
        "-m", "--hmass", type=float, required=False,
        help="""
        Hydrogen mass. A larger hydrogen mass allows a longer timestep."
        """,
        default=2
    )
    args = parser.parse_args()
    pressure = float(args.pressure)
    temperature = float(args.temperature)

    if args.fix:
        print("Running pdbfixer.")
        os.makedirs(f"{args.output}/structures", exist_ok=True)
        fix_pdb(args.structure, f"{args.output}/structures/{args.name}_fixed.pdb")
        structure = f"{args.output}/structures/{args.name}_fixed.pdb"
    else:
        structure = args.structure   

    setup = OMMSetup(
        structures=[structure],
        temperature=temperature,
        name=args.name,
        pressure=pressure,
        Hmass=args.hmass,
        timestep=args.timestep,
    )
    setup.model()
    print(f"Adding solvent.")
    setup.make_system()
    print("Minimizing the system.")
    setup.make_simulation()
    setup.save(args.output)

if __name__ == "__main__":
    main()
