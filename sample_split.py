import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from graphbap.bapnet import BAPNet


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]
    return all_step_v


def sample_diffusion_ligand(
    model,
    data,
    prompt_data,
    prompt_data_2,
    prompt_data_3,
    num_samples,
    batch_size=16,
    device="cuda:0",
    num_steps=None,
    pos_only=False,
    center_pos_mode="protein",
    sample_num_atoms="prior",
    net_cond=None,
    cond_dim=128,
):

    assert net_cond is not None and prompt_data is not None

    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = (
            batch_size
            if i < num_batch - 1
            else num_samples - batch_size * (num_batch - 1)
        )
        batch = Batch.from_data_list(
            [data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH
        ).to(device)
        prompt_batch = Batch.from_data_list(
            [prompt_data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH
        ).to(device)
        prompt_batch_2 = Batch.from_data_list(
            [prompt_data_2.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH
        ).to(device)
        prompt_batch_3 = Batch.from_data_list(
            [prompt_data_3.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH
        ).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == "prior":
                pocket_size = atom_num.get_space_size(
                    batch.protein_pos.detach().cpu().numpy()
                )
                ligand_num_atoms = [
                    atom_num.sample_atom_num(pocket_size).astype(int)
                    for _ in range(n_data)
                ]
                batch_ligand = torch.repeat_interleave(
                    torch.arange(n_data), torch.tensor(ligand_num_atoms)
                ).to(device)
            elif sample_num_atoms == "range":
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(
                    torch.arange(n_data), torch.tensor(ligand_num_atoms)
                ).to(device)
            elif sample_num_atoms == "ref":
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(
                    torch.ones_like(batch_ligand), batch_ligand, dim=0
                ).tolist()
            else:
                raise ValueError

            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(
                    device
                )
                init_ligand_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                prompt_ligand_pos=prompt_batch.ligand_pos,
                prompt_ligand_v=prompt_batch.ligand_atom_feature_full,
                prompt_batch_ligand=prompt_batch.ligand_element_batch,
                prompt_ligand_pos_2=prompt_batch_2.ligand_pos,
                prompt_ligand_v_2=prompt_batch_2.ligand_atom_feature_full,
                prompt_batch_ligand_2=prompt_batch_2.ligand_element_batch,
                prompt_ligand_pos_3=prompt_batch_3.ligand_pos,
                prompt_ligand_v_3=prompt_batch_3.ligand_atom_feature_full,
                prompt_batch_ligand_3=prompt_batch_3.ligand_element_batch,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,
                net_cond=net_cond,
                cond_dim=cond_dim,
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = (
                r["pos"],
                r["v"],
                r["pos_traj"],
                r["v_traj"],
            )
            ligand_v0_traj, ligand_vt_traj = r["v0_traj"], r["vt_traj"]
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [
                ligand_pos_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]]
                for k in range(n_data)
            ]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(
                        p_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]]
                    )
            all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]
            all_pred_pos_traj += [p for p in all_step_pos]

            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [
                ligand_v_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]]
                for k in range(n_data)
            ]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return (
        all_pred_pos,
        all_pred_v,
        all_pred_pos_traj,
        all_pred_v_traj,
        all_pred_v0_traj,
        all_pred_vt_traj,
        time_list,
    )


if __name__ == "__main__":
    root_dir = "./"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=root_dir + "/configs/sampling.yml"
    )
    parser.add_argument(
        "--train_config", type=str, default=root_dir + "/configs/training.yml"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument(
        "--result_path", type=str, default=root_dir + "/sampled_results_test/"
    )
    parser.add_argument(
        "--test_prompt_indices_path",
        type=str,
        default=root_dir + "/src/test_prompt_ligand_indices_top3.pt",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=99)
    args = parser.parse_args()

    logger = misc.get_logger("sampling")

    config = misc.load_config(args.config)
    train_config = misc.load_config(args.train_config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {train_config}")

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = train_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose(
        [
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
        ]
    )

    dataset, subsets = get_dataset(config=train_config.data, transform=transform)
    train_set, test_set = subsets["train"], subsets["test"]
    logger.info(f"Successfully load the dataset (size: {len(test_set)})!")

    test_prompt_indices = torch.load(args.test_prompt_indices_path)

    net_cond = BAPNet(
        ckpt_path=train_config.net_cond.ckpt_path,
        hidden_nf=train_config.net_cond.hidden_dim,
    ).to(args.device)

    model = ScorePosNet3D(
        train_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Successfully load the model! {config.model.checkpoint}")

    num_test = len(test_set)
    for data_id in range(args.start_index, args.end_index + 1):
        data = test_set[data_id]
        prompt_data_id = test_prompt_indices[data_id, -1].item()
        prompt_data_id_2 = test_prompt_indices[data_id, -2].item()
        prompt_data_id_3 = test_prompt_indices[data_id, -3].item()

        prompt_data = dataset[prompt_data_id]
        prompt_data_2 = dataset[prompt_data_id_2]
        prompt_data_3 = dataset[prompt_data_id_3]

        (
            pred_pos,
            pred_v,
            pred_pos_traj,
            pred_v_traj,
            pred_v0_traj,
            pred_vt_traj,
            time_list,
        ) = sample_diffusion_ligand(
            model,
            data,
            prompt_data,
            prompt_data_2,
            prompt_data_3,
            config.sample.num_samples,
            batch_size=args.batch_size,
            device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            net_cond=net_cond,
            cond_dim=train_config.model.cond_dim,
        )
        result = {
            "data": data,
            "pred_ligand_pos": pred_pos,
            "pred_ligand_v": pred_v,
            "pred_ligand_pos_traj": pred_pos_traj,
            "pred_ligand_v_traj": pred_v_traj,
            "time": time_list,
        }
        logger.info("Sample done!")

        result_path = args.result_path
        os.makedirs(result_path, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(result_path, "sample.yml"))
        torch.save(result, os.path.join(result_path, f"result_{data_id}.pt"))
