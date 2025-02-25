import os
import numpy as np
import torch
from glob import glob
from rdkit import Chem
from utils.evaluation.similarity import tanimoto_sim, tanimoto_sim_N_to_1
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import argparse
from collections import defaultdict
from utils.evaluation.docking_vina import VinaDockingTask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docking_mode",
        default="vina_dock",
        type=str,
    )
    parser.add_argument("--calculate_diversity", type=bool, default=True) # This process will cost an amount of computational resources
    parser.add_argument("--n_jobs", type=int, default=60)
    parser.add_argument("--protein_root", type=str, default="./path/to/test_set/")
    args = parser.parse_args()

    eval_path = "./eval_results/"
    results_fn_list = glob(os.path.join(eval_path, "metrics_*.pt"))
    print("num of results.pt: ", len(results_fn_list))

    qed_all = []
    sa_all = []
    qvina_all = []
    vina_score_all = []
    vina_min_all = []
    vina_dock_all = []

    for rfn in results_fn_list:
        result_i = torch.load(rfn)["all_results"]

    #     qed_all += [r["chem_results"]["qed"] for r in result_i]
    #     sa_all += [r["chem_results"]["sa"] for r in result_i]
    #     if args.docking_mode == "qvina":
    #         qvina_all += [r["vina"][0]["affinity"] for r in result_i]
    #     elif args.docking_mode in ["vina_dock", "vina_score"]:
    #         vina_score_all += [r["vina"]["score_only"][0]["affinity"] for r in result_i]
    #         vina_min_all += [r["vina"]["minimize"][0]["affinity"] for r in result_i]
    #         if args.docking_mode == "vina_dock":
    #             vina_dock_all += [r["vina"]["dock"][0]["affinity"] for r in result_i]

    # qed_all_mean, qed_all_median = np.mean(qed_all), np.median(qed_all)
    # sa_all_mean, sa_all_median = np.mean(sa_all), np.median(sa_all)

    # print("qed_all_mean, qed_all_median:", qed_all_mean, qed_all_median)
    # print("sa_all_mean, sa_all_median:", sa_all_mean, sa_all_median)

    # if len(qvina_all):
    #     qvina_all_mean, qvina_all_median = np.mean(qvina_all), np.median(qvina_all)
    #     print("qvina_all_mean, qvina_all_median:", qvina_all_mean, qvina_all_median)

    # if len(vina_score_all):
    #     vina_score_all_mean, vina_score_all_median = np.mean(vina_score_all), np.median(
    #         vina_score_all
    #     )
    #     print(
    #         "vina_score_all_mean, vina_score_all_median:",
    #         vina_score_all_mean,
    #         vina_score_all_median,
    #     )

    # if len(vina_min_all):
    #     vina_min_all_mean, vina_min_all_median = np.mean(vina_min_all), np.median(
    #         vina_min_all
    #     )
    #     print(
    #         "vina_min_all_mean, vina_min_all_median:",
    #         vina_min_all_mean,
    #         vina_min_all_median,
    #     )

    # if len(vina_dock_all):
    #     vina_dock_all_mean, vina_dock_all_median = np.mean(vina_dock_all), np.median(
    #         vina_dock_all
    #     )
    #     print(
    #         "vina_dock_all_mean, vina_dock_median:",
    #         vina_dock_all_mean,
    #         vina_dock_all_median,
    #     )
    test_ligand_name = []
    if args.calculate_diversity: 
        for r in result_i:
            test_ligand_name.append(r['ligand_filename'])
        test_ligand_name = list(set(test_ligand_name))
    # print(len(test_ligand_name))
    #     diversity = []
    #     for ligand_name in tqdm(test_ligand_name):
    #         mols = []
    #         for r in result_i:
    #             if r['ligand_filename'] == ligand_name:
    #                 mols.append(r['mol'])
    #         average_smi = tanimoto_parallel(args, mols)
    #         diversity.append(average_smi)
    # print('Mean diversity', 1-np.mean(diversity), 'Median diversity', 1-np.median(diversity))

    if args.calculate_diversity: 
        high_affinity = []
        for ligand_name in tqdm(test_ligand_name):
            vina_dock_list = []
            docking_results = None
            for r in result_i:
                if r['ligand_filename'] == ligand_name:
                    vina_dock_list.append(r['vina']['dock'][0]['affinity'])
                    if docking_results is None:
                        supplier = Chem.SDMolSupplier(os.path.join(args.protein_root, ligand_name))
                        mol = next(supplier, None)
                        vina_task = VinaDockingTask.from_generated_mol(
                                mol, ligand_name, protein_root=args.protein_root
                            )
                        docking_results = vina_task.run(
                                    mode="dock", exhaustiveness=16
                                )
            high_affinity.append(sum(vina_dock_list<docking_results[0]['affinity'])/len(vina_dock_list))
    print('High affinity', np.mean(high_affinity), 'High affinity', np.median(high_affinity))
    #         average_smi = tanimoto_parallel(args, mols)
    #         diversity.append(average_smi)
    # print('Mean diversity', 1-np.mean(diversity), 'Median diversity', 1-np.median(diversity))

def tanimoto_parallel(args, molecules):
    num_molecules = len(molecules)
    tasks = [(molecules[i], molecules[j]) for i in range(num_molecules) for j in range(num_molecules)]

    # Parallel computation of Tanimoto similarity
    n_jobs = args.n_jobs
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(tanimoto_sim)(mol1, mol2) for mol1, mol2 in tasks)
    return np.mean(results)
        # print("Average Tanimoto Similarity:", np.mean(results), "Mean Tanimoto Similarity", np.mean(results))

        # Save results
        # with open('./eval_results/tanimoto_similarity_results.pkl', 'wb') as f:
        #     pickle.dump(results, f)
        # Load the results from the saved file
        # with open('./eval_results/tanimoto_similarity_results.pkl', 'rb') as f:
        #     loaded_results = pickle.load(f)

        # print("Loaded Tanimoto Similarities:", np.mean(loaded_results), np.median(loaded_results))

if __name__=='__main__':
    main()