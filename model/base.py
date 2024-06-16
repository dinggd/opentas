import logging
from abc import ABC
import os
import numpy as np
import torch
from tqdm import tqdm
import re
from eval import edit_score, f_score, read_file


class BaseTrainer(ABC):

    def __init__(self, cfg):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.init_model(self.cfg)
        self.init_criterion(self.cfg)

    def train(self, train_dataset_loader):

        self.model.train()
        self.model.cuda()

        optimizers = self.get_optimizers(self.cfg)
        schedulers = self.get_schedulers(optimizers, self.cfg)

        last_epoch = 0
        # check is resume training is enabled
        if self.cfg.TRAIN.RESUME:
            # find epoch provided if not find the last available
            for filename in os.listdir(self.cfg.TRAIN.MODEL_DIR):
                if filename.endswith("model"):
                    match = re.search(r"\d+", filename)
                    if match:
                        number = int(match.group())
                        # Ensure that optimizers are also saved for this epoch
                        if np.all([
                            f"epoch-{number}-opt{opt_idx}.opt" in os.listdir(self.cfg.TRAIN.MODEL_DIR) 
                            for opt_idx in range(len(optimizers))
                        ]): 
                            last_epoch = number if number > last_epoch else last_epoch
            logging.info(f"-------- Loaded {last_epoch}-th epoch model ---------")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.cfg.TRAIN.MODEL_DIR, f"epoch-{last_epoch}.model")
                )
            )
            for opt_idx in range(len(optimizers)):
                optimizers[opt_idx].load_state_dict(
                    torch.load(os.path.join(self.cfg.TRAIN.MODEL_DIR, f"epoch-{last_epoch}-opt{opt_idx}.opt"))
                )

        for epoch in tqdm(range(last_epoch, self.cfg.TRAIN.NUM_EPOCHS)):
            epoch_loss = 0
            metrics_accum_dict = self.get_empty_metrics_accum_dict(self.cfg)

            for batch_train_data in train_dataset_loader:

                loss, predictions = self.get_train_loss_preds(batch_train_data, self.cfg)

                epoch_loss += loss.item()

                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

                self.accumulate_metrics(metrics_accum_dict, batch_train_data, predictions, self.cfg)

            for scheduler in schedulers:
                scheduler.step(epoch_loss)

            if (
                (epoch + 1) % self.cfg.TRAIN.LOG_FREQ == 0
                or (epoch + 1) == self.cfg.TRAIN.NUM_EPOCHS
            ):
                score_dict = self.score_accumulated_metrics(metrics_accum_dict, epoch_loss, 
                                                            train_dataset_loader, self.cfg)
                scores_to_log = [f"{k} = {v}" for k,v in score_dict.items()]
                logging.info(f"[epoch {epoch + 1}]: " + f", ".join(scores_to_log))

        torch.save(
            self.model.state_dict(),
            f"{self.cfg.TRAIN.MODEL_DIR}/epoch-{self.cfg.TRAIN.NUM_EPOCHS}.model",
        )
        for opt_idx in range(len(optimizers)):
            torch.save(
                optimizers[opt_idx].state_dict(),
                f"{self.cfg.TRAIN.MODEL_DIR}/epoch-{self.cfg.TRAIN.NUM_EPOCHS}-opt{opt_idx}.opt",
            )

    def predict(self, test_dataset_loader):
        self.cfg.DATA.FEATURES_PATH,
        self.cfg.DATA.VID_LIST_FILE_TEST,
        self.cfg.DATA.ACTIONS_DICT,
        self.cfg.DATA.SAMPLE_RATE,
        self.cfg.DATA.GT_PATH,
        actions_dict = dict(self.cfg.DATA.ACTIONS_DICT) if not isinstance(self.cfg.DATA.ACTIONS_DICT, dict) \
                       else self.cfg.DATA.ACTIONS_DICT

        with open(self.cfg.DATA.VID_LIST_FILE_TEST, "r") as f:
            list_of_vids = f.read().splitlines()

        self.model.eval()
        with torch.no_grad():
            self.model.cuda()

            for test_sample in test_dataset_loader:
                video, predicted_classes = self.get_eval_preds(test_sample, 
                                                        actions_dict, self.cfg)

                f_name = video.split("/")[-1].split(".")[0]
                with open(f"{self.cfg.TRAIN.RESULT_DIR}/{f_name}", "w") as f:
                    f.write("### Frame level recognition: ###\n")
                    f.write(" ".join(predicted_classes))

        overlap = [0.1, 0.25, 0.5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct, total, edit = 0, 0, 0

        for vid in list_of_vids:
            gt_file = self.cfg.DATA.GT_PATH + vid
            gt_content = read_file(gt_file).split("\n")[0:-1]
            recog_file = os.path.join(self.cfg.TRAIN.RESULT_DIR, vid.split(".")[0])
            recog_content = read_file(recog_file).split("\n")[1].split()

            total += len(gt_content)
            correct += sum(
                1 for gt, recog in zip(gt_content, recog_content) if gt == recog
            )

            edit += edit_score(recog_content, gt_content)

            for idx, thres in enumerate(overlap):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, thres)
                tp[idx] += tp1
                fp[idx] += fp1
                fn[idx] += fn1

        final = []
        final.append((100 * float(correct) / total))
        final.append((1.0 * edit) / len(list_of_vids))

        for idx, thres in enumerate(overlap):
            precision = tp[idx] / float(tp[idx] + fp[idx])
            recall = tp[idx] / float(tp[idx] + fn[idx])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            final.append(f1)
            
        np.savetxt(self.cfg.TRAIN.RES_FILENAME, np.array([final]), fmt="%1.2f")
        return final
    
    
    def init_model(self, cfg):
        """Hook method. Initialize the model to train. Must store into `self.model`."""
        raise NotImplementedError


    def init_criterion(self, cfg):
        """Hook method. Initialize the loss terms. Must store criterion objects into `self`."""
        raise NotImplementedError


    def get_optimizers(self, cfg):
        """Hook method. Define the optimization protocols to use for training.

        Returns:
        A List of optimizers
        """
        raise NotImplementedError()
    

    def get_schedulers(self, optimizers, cfg):
        """Hook method. Define LR schedulers to use for training.
        
        Args:
        - `optimizers`: A List of optimizers

        Returns:
        A List of schedulers
        """
        raise NotImplementedError()
    

    def get_train_loss_preds(self, batch_train_data, cfg):
        """Hook method. Defines model's training protocol.
        
        Args:
        - `batch_train_data`: A batch of training data from the dataset. Default: (batch_input, batch_target, mask)

        Returns:
        A Tuple containing:
        - `loss`: The training loss for this batch
        - `predictions`: The predictions from this batch
        """
        raise NotImplementedError()
    

    def get_empty_metrics_accum_dict(self, cfg):
        """Hook method. Creates the empty `metrics_accum_dict` at each epoch. 
        Default implementation is provided."""
        return {
            "correct": 0,
            "total": 0
        }
    
    
    def accumulate_metrics(self, metrics_accum_dict, batch_train_data, predictions, cfg):
        """Hook method. Allows Trainers to accumulate metrics into the `metrics_accum_dict`. 
        Called after the model is trained on a batch. Default implementation is provided.
        
        Args:
        - `metrics_accum_dict`: Dictionary containing metrics accumulated throughout the current epoch
        - `batch_train_data`: Current batch of training data. Default: (batch_input, batch_target, mask)
        - `predictions`: Predictions for the current batch of training data
        """
        _, batch_target, mask = batch_train_data
        _, predicted = torch.max(predictions[-1].data, 1)

        metrics_accum_dict["correct"] += torch.sum(
                (predicted == batch_target).float() * mask[:, 0, :].squeeze(1)
            ).item()
        
        metrics_accum_dict["total"] += torch.sum(mask[:, 0, :]).item()
        

    def score_accumulated_metrics(self, metrics_accum_dict, epoch_loss, train_dataset_loader, cfg):
        """Hook method. Return the scores to report from the accumulated metrics after a training epoch. 
        Default implementation is provided.
        
        Args:
        - `metrics_accum_dict`: Dictionary containing metrics accumulated throughout the current epoch
        - `epoch_loss`: Total epoch loss
        - `train_dataset_loader`: The dataset loader for retrieval of dataset statistics
        """
        return {
            "epoch loss": float(epoch_loss / len(train_dataset_loader.list_of_examples)),
            "acc": float(metrics_accum_dict["correct"]) / metrics_accum_dict["total"]
        }


    def get_eval_preds(self, test_sample, actions_dict, cfg):
        """Hook method. Defines model's evaluation protocol.
        
        Args:
        - `test_sample`: The test sample. Default: (video, sampled_feats)
        - `actions_dict`: The dictionary of action indices to labels

        Returns:
        A Tuple containing:
        - a str `video` containing the name of the sample
        - a List `predicted_classes` containing the frame-wise label predictions for each frame in the test sample
        """
        raise NotImplementedError()
