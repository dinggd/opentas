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

    def train(self, batch_gen):
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
            correct = 0
            total = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(self.cfg.TRAIN.BZ)
                batch_input, batch_target, mask = (
                    batch_input.cuda(),
                    batch_target.cuda(),
                    mask.cuda(),
                )

                loss, predictions = self.get_train_loss_preds((batch_input, batch_target, mask), self.cfg)

                epoch_loss += loss.item()

                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += torch.sum(
                    (predicted == batch_target).float() * mask[:, 0, :].squeeze(1)
                ).item()
                total += torch.sum(mask[:, 0, :]).item()

            for scheduler in schedulers:
                scheduler.step(epoch_loss)

            batch_gen.reset()

            if (
                (epoch + 1) % self.cfg.TRAIN.LOG_FREQ == 0
                or (epoch + 1) == self.cfg.TRAIN.NUM_EPOCHS
            ):
                logging.info(
                    "[epoch %d]: epoch loss = %f,   acc = %f"
                    % (
                        epoch + 1,
                        epoch_loss / len(batch_gen.list_of_examples),
                        float(correct) / total,
                    )
                )

        torch.save(
            self.model.state_dict(),
            f"{self.cfg.TRAIN.MODEL_DIR}/epoch-{self.cfg.TRAIN.NUM_EPOCHS}.model",
        )
        for opt_idx in range(len(optimizers)):
            torch.save(
                optimizers[opt_idx].state_dict(),
                f"{self.cfg.TRAIN.MODEL_DIR}/epoch-{self.cfg.TRAIN.NUM_EPOCHS}-opt{opt_idx}.opt",
            )

    def predict(self):
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

            for vid in list_of_vids:
                features = np.load(
                    os.path.join(self.cfg.DATA.FEATURES_PATH, f"{vid.split('.')[0]}.npy")
                )[:, :: self.cfg.DATA.SAMPLE_RATE]
                input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).cuda()

                predicted_classes = self.get_eval_preds(input_x, actions_dict, self.cfg)

                f_name = vid.split("/")[-1].split(".")[0]
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
        """Hook method. Initialize the model to train. Must store into self.model variable."""
        raise NotImplementedError


    def init_criterion(self, cfg):
        """Hook method. Initialize the loss terms. Store criterion into the self object."""
        raise NotImplementedError


    def get_optimizers(self, cfg):
        """Hook method. Define the optimizers to use for training."""
        raise NotImplementedError()
    

    def get_schedulers(self, optimizers, cfg):
        """Hook method. Define LR schedulers to use for training."""
        raise NotImplementedError()
    
    
    def get_train_loss_preds(self, batch_train_data, cfg):
        """Hook method. Defines model's training protocol.
        
        Args:
        - `batch_train_data`: A batch of training data from the dataset

        Returns:
        A Tuple containing:
        - `loss`: The training loss for this batch
        - `predictions`: The predictions from this batch
        """
        raise NotImplementedError()
    
    
    def get_eval_preds(self, test_input, actions_dict, cfg):
        """Hook method. Defines model's evaluation protocol.
        
        Args:
        - `test_input`: The test sample
        - `actions_dict`: The dictionary of action indices to labels

        Returns:
        A List `predicted_classes` containing the frame-wise label predictions for each frame in the test sample
        """
        raise NotImplementedError()
