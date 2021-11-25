""" Script for performing evaluation runs on the Ensemble models. This can also be easily used to do evaluations of
single NNs by setting the ensemble size to 1 and referencing the snapshot you want from the command line.
"""

import copy
import os
import time
from functools import partial

import numpy as np
import pandas
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter

from metrics import accuracy
from pyll.base import TorchModel, AverageMeter
from pyll.session import PyLL
from pyll.utils.workspace import Workspace


def main():
    session = PyLL()
    datasets = session.datasets
    workspace = session.workspace
    summaries = session.summaries
    config = session.config

    ###########################
    ensemble_properties = session.ensemble_properties

    if ensemble_properties:
        ensemble_size = ensemble_properties["ensemble_size"]
        if ensemble_properties["ensemble_type"] == "snapshot_ensemble":
            ensemble_size = ensemble_size - 1  # Since we discard the first member as burn-in
            model_paths = [
                os.path.join(workspace.checkpoint_dir, f"snapshot_ensemble.{i}.pth.tar")
                for i in range(1, ensemble_size + 1)
            ]
            for p in model_paths: print(f"Loading: {p}")

        elif ensemble_properties["ensemble_type"] == "deep_ensemble":
            model_paths = [
                os.path.join(workspace.checkpoint_dir, f"ensemble.{i}.pth.tar")
                for i in range(ensemble_size)
            ]
            for p in model_paths: print(f"Loading: {p}")

        else:
            raise RuntimeError("Ensemble properties missing valid ensemble_type")

        models = [copy.deepcopy(session.model) for _ in range(ensemble_size)]

        # Load the state dicts
        saved_states = [torch.load(p) for p in model_paths]

        # Update each ensemble model with the loaded state dicts
        for i, model in enumerate(models):
            model_state = model.state_dict()
            loaded_state = saved_states[i]["state_dict"]
            filtered_loaded_state = {key : loaded_state[key] for key in model_state.keys()}
            model_state.update(filtered_loaded_state)

            model.load_state_dict(model_state)

        # Make the ensemble itself a Torch model
        ensemble = torch.nn.ModuleList(models)
        session.model = ensemble

    # Should be able to eval a standard GapNet with this ...
    else:
        ensemble = torch.nn.ModuleList([session.model])
        session.model = ensemble

    ############################

    if config.has_value("evaluation") and config.evaluation.batchsize is not None:
        batchsize_eval = config.evaluation.batchsize
    else:
        batchsize_eval = config.training.batchsize

    dataset_to_eval = config.get_value("dataset_to_eval")
    if dataset_to_eval in datasets:
        loader_val = torch.utils.data.DataLoader(datasets[dataset_to_eval],
                                                 batch_size=batchsize_eval, shuffle=False,
                                                 num_workers=config.workers, pin_memory=True, drop_last=False)
        eval_featurizer = partial(featurize_ensemble,
                                  loader=loader_val,
                                  split_name=dataset_to_eval,
                                  ensemble=ensemble,
                                  config=config,
                                  summary=summaries[dataset_to_eval],
                                  workspace=workspace)

    # Training Loop
    try:
        if eval_featurizer is not None:
            # evaluate on validation set
            _ = eval_featurizer(samples_seen=0)

    finally:
        print("Closing summary writers...")
        for name, summary in summaries.items():
            summary.export_scalars_to_json(os.path.join(workspace.statistics_dir, "{}.json".format(name)))
            summary.close()
        print("Done")


def featurize_ensemble(
        loader, split_name, ensemble: TorchModel, config, samples_seen, summary: SummaryWriter, workspace: Workspace
):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    config_eval = config.get_value("evaluation", None)

    ensemble_size = len(ensemble)
    batchsize = loader.batch_size
    n_samples = len(loader.dataset)

    n_tasks = loader.dataset.num_classes
    fc_units = 2048
    model_params = config.get_value('model_params', None)
    if model_params and "fc_units" in model_params:
        fc_units = model_params["fc_units"]
            
    features = np.zeros(shape=(ensemble_size, n_samples, fc_units))
    targets = np.zeros(shape=(n_samples, n_tasks))
    sample_keys = []

    # switch to evaluate mode
    ensemble.eval()

    end = time.time()
    for i, batch in enumerate(loader):
        with torch.no_grad():
            input = batch["input"]
            target = batch["target"]
            sample_keys.extend(batch["ID"])
            target = target.cuda(non_blocking=True)

            for j, model in enumerate(ensemble):
                # compute output
                output = model(input)
                output = torch.stack([output])

                # Store predictions
                single_features = output.cpu().data.numpy()
                features[j, i * batchsize:(i + 1) * batchsize, :] = single_features
        #Store targets
        target = target.cpu().numpy()
        target_tasks = target
        targets[i * batchsize:(i + 1) * batchsize, :] = target_tasks / 2 + 0.5

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('{split}: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time, loss=losses, acc=accuracies, split=split_name
            )
            )

    # calculate mean over views for mean well predictions
    # retain breakdown by ensemble
    dfs = []
    for i in range(ensemble_size):
        df = pandas.DataFrame(data=features[i], index=sample_keys)
        df["ensemble_index"] = i
        df = df.set_index("ensemble_index", append=True)
        dfs.append(df)

    df_features_granular = pandas.concat(dfs)
    # Consolidate at group, rather than view level. Average all predictions for each component view across the ensemble
    groups = (
        df_features_granular
            .groupby(by=lambda key: "-".join(key[0].split("-")[0:2]))
            .mean()
            .sort_index(inplace=False)
    )
    ensemble_features = groups.values

    # also group targets and pick first element of group (as they should all have the same label anyway)
    df = pandas.DataFrame(data=targets, index=sample_keys)
    groups = df.groupby(by=lambda key: "-".join(key.split("-")[0:2])).first().sort_index(inplace=False)
    targets = groups.values
    sample_keys_grouped = groups.index

    # store predictions
    if workspace is not None:
        np.savez_compressed(
            file="{}/ensemble-features-{}.npz".format(workspace.results_dir, split_name),
            granular_features=features,
            features=ensemble_features,
            targets=targets,
            granular_ids=sample_keys,
            ids=sample_keys_grouped,
            ensemble_size=ensemble_size
        )


if __name__ == '__main__':
    main()
