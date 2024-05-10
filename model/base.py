import logging
from abc import ABC
import os
import numpy as np
import torch
from tqdm import tqdm
import re
from eval import edit_score, f_score, read_file


class BaseTrainer(ABC):
    """NOTE: All concrete classes of this must initialize self.model and self.num_classes"""

    def train(self, batch_gen, cfg):
        self.model.train()
        self.model.cuda()

        optimizers = self.get_optimizers(cfg.TRAIN.LR)
        schedulers = self.get_schedulers(optimizers)

        last_epoch = 0
        # check is resume training is enabled
        if cfg.TRAIN.RESUME:
            # find epoch provided if not find the last available
            for filename in os.listdir(cfg.TRAIN.MODEL_DIR):
                if filename.endswith("model"):
                    match = re.search(r"\d+", filename)
                    if match:
                        number = int(match.group())
                        # Ensure that optimizers are also saved for this epoch
                        if np.all([
                            f"epoch-{number}-opt{opt_idx}.opt" in os.listdir(cfg.TRAIN.MODEL_DIR) 
                            for opt_idx in range(len(optimizers))
                        ]): 
                            last_epoch = number if number > last_epoch else last_epoch
            logging.info(f"-------- Loaded {last_epoch}-th epoch model ---------")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(cfg.TRAIN.MODEL_DIR, f"epoch-{last_epoch}.model")
                )
            )
            for opt_idx in range(len(optimizers)):
                optimizers[opt_idx].load_state_dict(
                    torch.load(os.path.join(cfg.TRAIN.MODEL_DIR, f"epoch-{last_epoch}-opt{opt_idx}.opt"))
                )

        for epoch in tqdm(range(last_epoch, cfg.TRAIN.NUM_EPOCHS)):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(cfg.TRAIN.BZ)
                batch_input, batch_target, mask = (
                    batch_input.cuda(),
                    batch_target.cuda(),
                    mask.cuda(),
                )
                
                predictions = self.forward_train(batch_input, mask)

                loss = self.calc_loss(predictions, batch_target, mask)

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
                (epoch + 1) % cfg.TRAIN.LOG_FREQ == 0
                or (epoch + 1) == cfg.TRAIN.NUM_EPOCHS
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
            f"{cfg.TRAIN.MODEL_DIR}/epoch-{cfg.TRAIN.NUM_EPOCHS}.model",
        )
        for opt_idx in range(len(optimizers)):
            torch.save(
                optimizers[opt_idx].state_dict(),
                f"{cfg.TRAIN.MODEL_DIR}/epoch-{cfg.TRAIN.NUM_EPOCHS}-opt{opt_idx}.opt",
            )

    def predict(self, cfg):
        cfg.DATA.FEATURES_PATH,
        cfg.DATA.VID_LIST_FILE_TEST,
        cfg.DATA.ACTIONS_DICT,
        cfg.DATA.SAMPLE_RATE,
        cfg.DATA.GT_PATH,
        if not isinstance(cfg.DATA.ACTIONS_DICT, dict):
            actions_dict = dict(cfg.DATA.ACTIONS_DICT)

        with open(cfg.DATA.VID_LIST_FILE_TEST, "r") as f:
            list_of_vids = f.read().splitlines()

        self.model.eval()
        with torch.no_grad():
            self.model.cuda()

            for vid in list_of_vids:
                features = np.load(
                    os.path.join(cfg.DATA.FEATURES_PATH, f"{vid.split('.')[0]}.npy")
                )[:, :: cfg.DATA.SAMPLE_RATE]
                input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).cuda()

                predictions = self.forward_eval(input_x, torch.ones(input_x.size()).cuda())
                predicted_classes = [
                    list(actions_dict.keys())[
                        list(actions_dict.values()).index(pred.item())
                    ]
                    for pred in torch.max(predictions[-1].data, 1)[1].squeeze()
                ] * cfg.DATA.SAMPLE_RATE

                f_name = vid.split("/")[-1].split(".")[0]
                with open(f"{cfg.TRAIN.RESULT_DIR}/{f_name}", "w") as f:
                    f.write("### Frame level recognition: ###\n")
                    f.write(" ".join(predicted_classes))

        overlap = [0.1, 0.25, 0.5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct, total, edit = 0, 0, 0

        for vid in list_of_vids:
            gt_file = cfg.DATA.GT_PATH + vid
            gt_content = read_file(gt_file).split("\n")[0:-1]
            recog_file = os.path.join(cfg.TRAIN.RESULT_DIR, vid.split(".")[0])
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
        np.savetxt(cfg.TRAIN.RES_FILENAME, np.array([final]), fmt="%1.2f")
        return final

    def get_optimizers(self, learning_rate):
        """Hook method. Define the optimizers to use for training."""
        raise NotImplementedError()

    def get_schedulers(self, optimizers):
        """Hook method. Define LR schedulers to use for training."""
        raise NotImplementedError()

    def calc_loss(self, predictions, batch_target, mask):
        """Hook method. Calculates the loss between the target and predicted values."""
        raise NotImplementedError()

    def forward_train(self, *args, **kwargs):
        """Overridable hook method. Indicate forward propagation method to use for training."""
        return self.model(*args, **kwargs)

    def forward_eval(self, *args, **kwargs):
        """Overridable hook method. Indicate forward propagation method to use for evaluation."""
        return self.model(*args, **kwargs)
