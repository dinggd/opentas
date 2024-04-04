import logging
from abc import ABC

import numpy as np
import torch
from tqdm import tqdm

from eval import edit_score, f_score, read_file


class BaseTrainer(ABC):
    """NOTE: All concrete classes of this must initialize self.model and self.num_classes"""

    def train(self, batch_gen, save_dir, num_epochs, batch_size, learning_rate):
        self.model.train()
        self.model.cuda()

        optimizers = self.get_optimizers(learning_rate)
        schedulers = self.get_schedulers(optimizers)

        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = (
                    batch_input.cuda(),
                    batch_target.cuda(),
                    mask.cuda(),
                )
                predictions = self.model(batch_input, mask)

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

        torch.save(self.model.state_dict(), f"{save_dir}/epoch-{epoch + 1}.model")
        torch.save(optimizer.state_dict(), f"{save_dir}/epoch-{epoch + 1}.opt")
        logging.info(
            "[epoch %d]: epoch loss = %f,   acc = %f"
            % (
                epoch + 1,
                epoch_loss / len(batch_gen.list_of_examples),
                float(correct) / total,
            )
        )

    def get_optimizers(self, learning_rate):
        raise NotImplementedError()

    def get_schedulers(self, optimizers):
        raise NotImplementedError()

    def calc_loss(self, predictions, batch_target, mask):
        raise NotImplementedError()

    def predict(
        self,
        results_dir,
        features_path,
        vid_list_file,
        actions_dict,
        sample_rate,
        gt_path,
    ):
        if not isinstance(actions_dict, dict):
            actions_dict = dict(actions_dict)

        with open(vid_list_file, "r") as f:
            list_of_vids = f.read().splitlines()

        self.model.eval()
        with torch.no_grad():
            self.model.cuda()

            for vid in list_of_vids:
                features = np.load(f"{features_path}{vid.split('.')[0]}.npy")[
                    :, ::sample_rate
                ]
                input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).cuda()
                predictions = self.model(input_x, torch.ones(input_x.size()).cuda())
                predicted_classes = [
                    list(actions_dict.keys())[
                        list(actions_dict.values()).index(pred.item())
                    ]
                    for pred in torch.max(predictions[-1].data, 1)[1].squeeze()
                ] * sample_rate
                f_name = vid.split("/")[-1].split(".")[0]
                with open(f"{results_dir}/{f_name}", "w") as f:
                    f.write("### Frame level recognition: ###\n")
                    f.write(" ".join(predicted_classes))

        overlap = [0.1, 0.25, 0.5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct, total, edit = 0, 0, 0

        for vid in list_of_vids:
            gt_file = gt_path + vid
            gt_content = read_file(gt_file).split("\n")[0:-1]
            recog_file = results_dir + "/" + vid.split(".")[0]
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

        return final
