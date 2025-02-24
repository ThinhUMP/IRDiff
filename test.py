from openbabel import openbabel
from meeko import MoleculePreparation
from meeko import obutils
# from vina import Vina
import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile
import AutoDockTools
import os
import contextlib

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper

def convert_pdb_to_pdbqt(input_file, output_file):
    # Initialize Open Babel conversion objects
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")

    # Create a new molecule
    mol = openbabel.OBMol()

    # Read the molecule from file
    if not obConversion.ReadFile(mol, input_file):
        print("Failed to read the input file")
        return

    # Add hydrogens
    mol.AddHydrogens()

    # Optionally, add polar hydrogens only if dealing with a protein
    # mol.AddPolarHydrogens()

    # Assign charges using Gasteiger-Marsili (simple charge model)
    charge_model = openbabel.OBChargeModel.FindType("gasteiger")
    if charge_model:
        charge_model.ComputeCharges(mol)

    # Write the molecule to output file
    success = obConversion.WriteFile(mol, output_file)
    if success:
        print(f"Conversion successful, output file written to {output_file}")
    else:
        print("Failed to write the output file")
    return success


def convert_sdf_to_pdbqt(input_file, output_file):
    # Initialize Open Babel conversion objects
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdbqt")

    # Create a new molecule
    mol = openbabel.OBMol()

    # Read the molecule from file
    if not obConversion.ReadFile(mol, input_file):
        print("Failed to read the input file")
        return

    # Add hydrogens
    mol.AddHydrogens()

    # Assign charges using Gasteiger-Marsili (simple charge model)
    mol.AddPolarHydrogens()
    charge_model = openbabel.OBChargeModel.FindType("gasteiger")
    if charge_model:
        charge_model.ComputeCharges(mol)
    obConversion.WriteFile(mol, output_file)

import subprocess

def run_vina(receptor, ligand, config, output, log):
    """
    Run AutoDock Vina for docking ligands to a receptor.

    Parameters:
    receptor (str): Path to the receptor file in PDBQT format.
    ligand (str): Path to the ligand file in PDBQT format.
    config (str): Path to the configuration file with docking parameters.
    output (str): Path to the output file to store docking results.
    log (str): Path to the log file to store the Vina output log.
    """
    # Construct the command to run AutoDock Vina
    vina_command = [
        'vina',                        # Vina command, ensure it's in your PATH
        '--receptor', receptor,        # Specify the receptor file
        '--ligand', ligand,            # Specify the ligand file
        '--config', config,            # Configuration file for docking
        '--out', output,               # Output file for the results
        '--log', log                   # Log file to store the docking process output
    ]

    # Run the command
    try:
        subprocess.run(vina_command, check=True)
        print("Docking completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during docking: {e}")

if __name__=='__main__':
    try:
        # lig = PrepLig('./tmp/aakzchovpfsxumvxvhzdbfluvobpbr_ligand.sdf', 'sdf')
        # lig.get_pdbqt('./tmp/aakzchovpfsxumvxvhzdbfluvobpbr_ligand.pdbqt')
        # Example usage
        # input = './tmp/abgtvbxqmtcydcdeaxurtubqvcarkv_ligand.sdf'
        # output = './tmp/abgtvbxqmtcydcdeaxurtubqvcarkv_ligand.pdbqt'
        # convert_sdf_to_pdbqt(input, output)
        # input_pro = './path/to/test_set/GLMU_STRPN_2_459_0/4aaw_A_rec.pdb'
        # output_pro = './path/to/test_set/GLMU_STRPN_2_459_0/4aaw_A_rec.pdbqt'
        # convert_pdb_to_pdbqt(input_pro, output_pro)

        # Example usage:
        run_vina(
            receptor='./path/to/test_set/ABL2_HUMAN_274_551_0/4xli_B_rec.pdbqt',
            ligand='./tmp/aappelmlcjdboabuglylfhmnltonkj_ligand.pdbqt',
            config='config.txt',
            output='docked_output.pdbqt',
            log='docking_log.txt'
)
    except Exception as e:
        print(f"An error occurred: {e}")
