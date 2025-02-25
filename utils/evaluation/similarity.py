import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    # fp1 = AllChem.GetMorganFingerprint(mol, 2)
    # fp2 = AllChem.GetMorganFingerprint(ref, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def tanimoto_sim_N_to_1(mols, ref):
    sim = [tanimoto_sim(m, ref) for m in mols]
    return sim


def batched_number_of_rings(mols):
    n = []
    for m in mols:
        n.append(Chem.rdMolDescriptors.CalcNumRings(m))
    return np.array(n)
