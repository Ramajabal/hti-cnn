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
    model = session.model
    ensemble_properties = session.ensemble_properties
    workspace = session.workspace
    summaries = session.summaries
    config = session.config
    start_epoch = session.epoch
    best_performance = session.best_performance

    if config.has_value("evaluation") and config.evaluation.batchsize is not None:
        batchsize_eval = config.evaluation.batchsize
    else:
        batchsize_eval = config.training.batchsize

    # data loader
    loader_train = torch.utils.data.DataLoader(datasets["train"],
                                               batch_size=config.training.batchsize, shuffle=True,
                                               num_workers=config.workers, pin_memory=True,
                                               persistent_workers=True, drop_last=False)

    eval_val = None
    if "val" in datasets:
        loader_val = torch.utils.data.DataLoader(datasets["val"],
                                                 batch_size=batchsize_eval, shuffle=False,
                                                 num_workers=config.workers, pin_memory=False,
                                                 persistent_workers=True, drop_last=False)
        eval_val = partial(validate, loader=loader_val, split_name="val", config=config, summary=summaries["val"], workspace=workspace)
    
    if config.evaluate == "val":
        validate(loader_val, "val", model, 0, None, None)
    
    # initial evaluation
    if start_epoch == 0:
        if eval_val is not None:
            eval_val(model=model, samples_seen=0)
    
    # Training Loop
    try:
        if ensemble_properties:
            if ensemble_properties.get("ensemble_type") in ["deep_ensemble", "snapshot_ensemble"]:
                ensemble_size = ensemble_properties.get("ensemble_size")
                cycle_length = ensemble_properties.get("cycle_length")
                total_epochs = ensemble_size * cycle_length
                print(
                    f'Running Snapshot Ensemble for {ensemble_size} iterations '
                    f'of {cycle_length} cycles (Total: {total_epochs})'
                )
                # Used for tracking Ensembles to determine which component NN we are training
                m = int(start_epoch/cycle_length)
                print(f"Training Ensemble Member: {m}")

        else:
            total_epochs = config.training.epochs

        for epoch in range(start_epoch, total_epochs):
            # train for one epoch
            train(session, loader_train, epoch, summaries["train"], ensemble_properties)

            if eval_val is not None:
                # evaluate on validation set
                performance = eval_val(model=session.model, samples_seen=(epoch + 1) * len(loader_train.dataset))

                # remember best prec@1 and save checkpoint
                is_best = performance > best_performance
                best_performance = max(performance, best_performance)
                session.save_checkpoint(performance, is_best)

            # If we're running an ensemble model, we need to save our component ensembles
            # at specific checkpoints along the way
            if ensemble_properties:
                if ensemble_properties.get("ensemble_type") in ["deep_ensemble", "snapshot_ensemble"]:
                    cycle_length = ensemble_properties.get("cycle_length")
                    ensemble_size = ensemble_properties.get("ensemble_size")
                    checkpoints = [cycle_length * cp for cp in range(1, ensemble_size + 1)]

                    if (epoch + 1) in checkpoints:
                        # Save the m-th ensemble component and increment m
                        print(f"Saving Ensemble: {m} (epochs {epoch + 1})")
                        session.save_ensemble_checkpoint(performance, "ensemble", m)
                        m += 1

                        # For Deep Ensembles, we re-initialise the model (and weights for each ensemble member)
                        if ensemble_properties.get("ensemble_type") in ["deep_ensemble"]:
                            print(f"Re-initialising model weights for next cycle")
                            session.re_initialise_model(config, datasets)

    finally:
        if (epoch + 1) != total_epochs:
            print("Saving current state...")
            session.save_checkpoint(filename="user_abort.pth.tar", performance=-1, is_best=False)

        print("Closing summary writers...")
        for name, summary in summaries.items():
            summary.export_scalars_to_json(os.path.join(workspace.statistics_dir, "{}.json".format(name)))
            summary.close()
        print("Done")


def train(session: PyLL, loader, epoch, summary: SummaryWriter, ensemble_properties=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    n_tasks = loader.dataset.num_classes
    lr = session.config.optimizer_params.get("lr")

    # For snapshot ensembles ...
    if ensemble_properties:
        cycle_length = ensemble_properties.get("cycle_length")
        if ensemble_properties.get("ensemble_type") == "snapshot_ensemble":
            # Adjust learning rate
            initial_lr = ensemble_properties.get("initial_lr")
            lr = session.adjust_cyclic_annealing_lr(epoch, initial_lr, cycle_length)
            print(f"Epoch: {epoch}, Cycle length: {cycle_length}, Annealed Learning Rate: {lr:.3f}")

        elif ensemble_properties.get("ensemble_type") == "deep_ensemble":
            print(f"Epoch: {epoch}, Cycle length: {cycle_length}")

    elif session.config.has_value("lr_schedule"):
        lr = session.adjust_learning_rate(epoch)

    # switch to train mode
    session.model.train()

    end = time.time()
    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = batch["input"]
        target = batch["target"]
        target = target.cuda(non_blocking=True)

        loss, output = session.train_step(input, target, epoch)

        if output.size(1) != loader.dataset.num_classes:
            output, _ = torch.split(output, loader.dataset.num_classes, dim=1)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        target = target.cpu().numpy()
        pred = output.cpu().data.numpy()
        pred_tasks = pred
        target_tasks = target

        # measure accuracy and record loss
        acc = accuracy(pred_tasks, target_tasks)
        losses.update(loss.item(), input.size(0))
        accuracies.update(acc, input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write statistics
        samples_seen = session.samples_seen
        summary.add_scalar("Loss", losses.val, samples_seen)
        summary.add_scalar("Accuracy", accuracies.val, samples_seen)
        summary.add_scalar("Learning_Rate", session.optimizer.param_groups[0]['lr'], samples_seen)
        summary.add_scalar("Epoch", epoch, samples_seen)
        
        if i % session.config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Learning Rate {lr:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=accuracies, lr=lr))

def validate(loader, split_name, model: TorchModel, config, samples_seen, summary: SummaryWriter, workspace: Workspace):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    config_eval = config.get_value("evaluation", None)

    batchsize = loader.batch_size
    n_samples = len(loader.dataset)

    n_tasks = loader.dataset.num_classes    
    predictions = np.zeros(shape=(n_samples, n_tasks))
    targets = np.zeros(shape=(n_samples, n_tasks))
    sample_keys = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, batch in enumerate(loader):
        with torch.no_grad():
            input = batch["input"]
            target = batch["target"]
            sample_keys.extend(batch["ID"])
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.sigmoid(output)
            loss = model.module.loss(output, target)

        # store predictions and labels
        target = target.cpu().numpy()
        pred = output.cpu().data.numpy()
        pred_tasks = pred
        target_tasks = target

        # store
        predictions[i * batchsize:(i + 1) * batchsize, :] = pred_tasks
        targets[i * batchsize:(i + 1) * batchsize, :] = target_tasks / 2 + 0.5

        # measure accuracy and record loss
        acc = accuracy(pred_tasks, target_tasks)
        losses.update(loss.item(), input.size(0))
        accuracies.update(acc, input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('{split}: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time, loss=losses, acc=accuracies, split=split_name))

    # calculate mean over views for mean well predictions
    df = pandas.DataFrame(data=predictions, index=sample_keys)
    groups = df.groupby(by=lambda key: "-".join(key.split("-")[0:2])).mean().sort_index(inplace=False)
    predictions = groups.values
    # also group targets and pick first element of group (as they should all have the same label anyway)
    df = pandas.DataFrame(data=targets, index=sample_keys)
    groups = df.groupby(by=lambda key: "-".join(key.split("-")[0:2])).first().sort_index(inplace=False)
    targets = groups.values
    sample_keys = groups.index

    # store predictions
    if workspace is not None:
        np.savez_compressed(file="{}/step-{}-{}.npz".format(workspace.results_dir, samples_seen, split_name), predictions=predictions, targets=targets, ids=sample_keys)

    # AUC
    class_aucs = []
    for i in range(n_tasks):
        try:
            if np.any(targets[:, i] == 0) and np.any(targets[:, i] == 1):
                samples = list(np.where(targets[:, i] == 0)[0]) + list(np.where(targets[:, i] == 1)[0])
                class_auc = roc_auc_score(y_true=targets[samples, i], y_score=predictions[samples, i])
            else:
                class_auc = 0.5
            class_aucs.append(class_auc)
        except ValueError:
            class_aucs.append(0.5)

    mean_auc = float(np.mean(class_aucs))

    # write statistics
    if summary is not None:
        summary.add_scalar("Loss", losses.avg, samples_seen)
        summary.add_scalar("Accuracy", accuracies.avg, samples_seen)        
        summary.add_scalar("AUC", mean_auc, samples_seen)
        # AUC ROC per class
        if config_eval is not None and config_eval.get_value("class_statistics", False):
            for i, val in enumerate(class_aucs):
                summary.add_scalar("Tasks/Task_{}_AUC".format(i), val, samples_seen)

    print(' * Accuracy {acc.avg:.3f}\tAUC {auc:.3f}'.format(acc=accuracies, auc=mean_auc))
    return mean_auc


if __name__ == '__main__':
    main()
