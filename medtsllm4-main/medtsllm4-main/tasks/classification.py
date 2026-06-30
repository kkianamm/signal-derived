"""
Sequence-level classification task for MedTsLLM-family models.

Cross-entropy training with accuracy / F1 / precision / recall (macro) scoring.
Records per-epoch TRAIN and TEST (and VAL) metrics into `self.history`, which
train.py writes to a JSON file at the end of the run.

Train metrics are computed from the logits already produced during the training
loop (running predictions), so they add no extra forward pass over the train set.

If the model exposes an `aux_loss` attribute (e.g. BiomedCoOpTS), it is added to
the cross-entropy loss during training. Backward compatible: models without
`aux_loss` contribute nothing.
"""

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from tqdm import tqdm

from .base import BaseTask


class ClassificationTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "classification"
        super(ClassificationTask, self).__init__(run_id, config, newrun)
        self.history = []   # per-epoch train/val/test metrics

    def _metrics(self, pred_int, target_int, prefix):
        avg = "binary" if self.train_dataset.n_classes == 2 else "macro"
        return {
            f"{prefix}/accuracy": accuracy_score(target_int, pred_int),
            f"{prefix}/f1": f1_score(target_int, pred_int, average=avg, zero_division=0),
            f"{prefix}/precision": precision_score(target_int, pred_int, average=avg, zero_division=0),
            f"{prefix}/recall": recall_score(target_int, pred_int, average=avg, zero_division=0),
        }

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()

            train_preds, train_targets = [], []
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed):
                    logits = self.model(inputs)
                    labels = inputs["labels"].long()
                    if logits.ndim == 1:
                        loss = self.loss_fn(logits, labels.to(logits.dtype))
                    else:
                        loss = self.loss_fn(logits, labels)

                    aux = getattr(self.model, "aux_loss", None)
                    if aux is not None:
                        loss = loss + aux

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.log_step(loss.item())

                # collect running train predictions (free: logits already computed)
                with torch.no_grad():
                    if logits.ndim == 1:
                        preds = (logits > 0).long()
                    else:
                        preds = logits.argmax(dim=-1)
                train_preds.append(preds.detach().cpu())
                train_targets.append(labels.detach().cpu())

            # ----- per-epoch metrics -----
            tp = torch.cat(train_preds).int().numpy()
            tt = torch.cat(train_targets).int().numpy()
            train_scores = self._metrics(tp, tt, "train")

            val_scores = self.val()
            test_scores = self.test()

            epoch_record = {"epoch": epoch + 1,
                            **train_scores, **val_scores, **test_scores}
            self.history.append(epoch_record)

            # log_epoch needs val/<eval_metric> present for best-model tracking
            self.log_epoch({**train_scores, **val_scores, **test_scores})
            self.scheduler.step()

        self.model.eval()

    def val(self):
        preds, targets = self.predict(self.val_dataloader)
        scores = {f"val/{k}": v for k, v in self.score(preds, targets).items()}
        self.log_scores(scores)
        return scores

    def test(self):
        preds, targets = self.predict(self.test_dataloader)
        scores = {f"test/{k}": v for k, v in self.score(preds, targets).items()}
        self.log_scores(scores)
        return scores

    def predict(self, dataloader):
        self.model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for inputs in tqdm(dataloader, total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                probs = self.model(inputs)
                if probs.ndim == 1:
                    probs = torch.stack([1.0 - probs, probs], dim=-1)
                all_probs.append(probs.float().cpu())
                all_targets.append(inputs["labels"].cpu())
        return torch.cat(all_probs, 0), torch.cat(all_targets, 0)

    def score(self, pred_scores, target):
        pred = pred_scores.argmax(dim=1).int().numpy()
        target = target.int().numpy()
        # reuse _metrics but strip the prefix (val()/test() add their own)
        m = self._metrics(pred, target, "x")
        return {k.split("/", 1)[1]: v for k, v in m.items()}

    def build_loss(self):
        is_binary = (self.train_dataset.n_classes == 2)
        loss_name = self.config.training.loss
        if loss_name in ("bce",) or is_binary:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif loss_name in ("ce", "cross_entropy", "auto"):
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss function selection: {loss_name}")
        return self.loss_fn
