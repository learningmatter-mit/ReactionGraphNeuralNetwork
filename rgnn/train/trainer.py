import shutil
import sys
import time
from typing import Dict
from rgnn.graph.utils import batch_to
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

MAX_EPOCHS = 100
BEST_METRIC = 1e10
BEST_LOSS = 1e10


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    def __init__(
        self,
        model_path,
        model,
        loss_fn,
        # metric_fn,
        optimizer,
        scheduler,
        train_loader,
        validation_loader,
        normalizer=None,
    ):
        self.model_path = model_path
        self.model = model
        self.loss_fn = loss_fn
        # self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.normalizer = normalizer

    def train(
        self,
        device,
        start_epoch=0,
        n_epochs=MAX_EPOCHS,
        best_loss=BEST_LOSS,
        best_metric=BEST_METRIC,
        early_stop=[50, 0.01],
        save_results=True,
        best_model_name="best_model.pth.tar",
    ):
        # switch to train mode
        # train model
        train_losses = []
        train_metrics = []
        val_losses = []
        val_metrics = []
        count = 0
        for epoch in range(start_epoch, n_epochs):
            self.model.train()
            self.to(device)
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            metrics = AverageMeter()
            end = time.time()
            for i, batch in enumerate(self.train_loader):
                self.model.zero_grad(set_to_none=True)
                batch = batch_to(batch, device)
                # print(batch["batch"])
                # measure data loading time
                data_time.update(time.time() - end)

                output = self.model(batch)
                # print(self.model.reaction_model.barrier_representation.layers[0].weight[0][:5].detach())
                # print(self.model.reaction_model.energy_output.layers[0].weight[0][:5].detach())
                loss = self.loss_fn(batch, output)
                # metric = self.metric_fn(output, batch)
                # measure accuracy and record loss
                losses.update(loss.data.cpu().item(), batch["n_atoms"].size(0))
                # metrics.update(metric.cpu().item(), target.size(0))

                # compute gradient and do optim step
                loss.backward()
                # Gradient clipping maybe helpful for spectra learning
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.2)
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if i % 10 == 0:
                #     print("Epoch: [{0}][{1}/{2}]\t"
                #           "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                #           "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                #           "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                #               epoch,
                #               i,
                #               len(self.train_loader),
                #               batch_time=batch_time,
                #               data_time=data_time,
                #               loss=losses,
                #               # metrics=metrics,
                #           ))
            train_losses.append(losses.avg)
            # train_metrics.append(metrics.avg)
            print(f"Epoch: [{epoch}]\t")
            val_loss = self.validate(device=device)
            val_losses.append(val_loss)
            # val_metrics.append(val_metric)

            if val_loss != val_loss:
                print("Exit due to NaN")
                sys.exit(1)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            is_best = val_loss <= best_loss
            best_loss = min(val_loss, best_loss)

            # best_metric = min(val_metric, best_m'etric)
            if self.normalizer is not None:
                normalizer_dict = {}
                for key, val in self.normalizer.items():
                    normalizer_dict[key] = val.state_dict()
                # TODO Save modelparams too
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "hyper_parameters": self.model.get_config(),
                        # "best_metric": best_metric,
                        "best_loss": best_loss,
                        "optimizer": self.optimizer.state_dict(),
                        "normalizer": normalizer_dict,
                    },
                    is_best,
                    self.model_path,
                    best_filename=best_model_name,
                )
            else:
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "hyper_parameters": self.model.get_config(),
                        # "best_metric": best_metric,
                        "best_loss": best_loss,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best,
                    self.model_path,
                    best_filename=best_model_name,
                )
            # Evaluate when to end training on account of no MAE improvement
            if is_best:
                count = 0
            else:
                count += 1
            if count > early_stop[0] and losses.avg < early_stop[1]:
                break

        # if save_results:
        #     with open(f"{self.model_path}/results.log", "w") as f:
        #         f.write("| Epoch | Train_L | Train_M | Vali_L | Vali_M |\n")
        #         for i in range(start_epoch, len(train_losses)):
        #             f.write(
        #                 f"{i}  {train_losses[i]:.4f}  {train_metrics[i]:.4f}  {val_losses[i]:.4f}  {val_metrics[i]:.4f}\n"
        #             )

        return best_loss

    def validate(self, device, test=False):
        """Validate the current state of the model using the validation set"""
        self.to(device=device)
        self.model.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        with torch.no_grad():
            for val_batch in self.validation_loader:
                val_batch = batch_to(val_batch, device)
                # target = val_batch["target"]

                output = self.model(val_batch)

                loss = self.loss_fn(val_batch, output)
                # metric = self.metric_fn(val_batch, output)

                losses.update(loss.data.cpu().item(), val_batch["n_atoms"].size(0))
                # metrics.update(metric.cpu().item(), batch.batch.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        print(
            # "Epoch: [{epoch:.3f}]\t"
            "*Validatoin: \t"
            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                # epoch=epoch,
                # "Metric {metrics.avg:.3f}".format(
                batch_time=batch_time,
                loss=losses,
                # metrics=metrics,
            )
        )

        return losses.avg

    def _load_model_state_dict(self, state_dict):
        if self.torch_parallel:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def save_checkpoint(
        self,
        state: Dict,
        is_best: bool,
        path: str,
        filename: str = "checkpoint.pth.tar",
        best_filename: str = "best_model.pth.tar",
    ):
        saving_filename = path + "/" + filename
        best_filename = path + "/" + best_filename
        torch.save(state, saving_filename)
        if is_best:
            shutil.copyfile(saving_filename, best_filename)

    def to(self, device):
        """Changes the device"""
        self.model.device = device
        self.model.to(device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())
