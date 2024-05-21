import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import logging
import re
import os
import numpy as np
from tqdm import tqdm
from dataset.batch_gen import C2fDataset, collate_fn_override
from eval import edit_score, f_score, read_file

from model.base import BaseTrainer
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diff // 2, diff - diff //2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TPPblock(nn.Module):
    def __init__(self, in_channels):
        super(TPPblock, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )

    def forward(self, x):
        self.in_channels, t = x.size(1), x.size(2)
        self.layer1 = F.upsample(
            self.conv(self.pool1(x)), size=t, mode="linear", align_corners=True
        )
        self.layer2 = F.upsample(
            self.conv(self.pool2(x)), size=t, mode="linear", align_corners=True
        )
        self.layer3 = F.upsample(
            self.conv(self.pool3(x)), size=t, mode="linear", align_corners=True
        )
        self.layer4 = F.upsample(
            self.conv(self.pool4(x)), size=t, mode="linear", align_corners=True
        )

        out = torch.cat([self.layer1, self.layer2,
                         self.layer3, self.layer4, x], 1)

        return out


class C2F_TCN(nn.Module):
    '''
        Features are extracted at the last layer of decoder.
    '''
    def __init__(self, n_channels, n_classes):
        super(C2F_TCN, self).__init__()
        self.inc = inconv(n_channels, 256)
        self.down1 = down(256, 256)
        self.down2 = down(256, 256)
        self.down3 = down(256, 128)
        self.down4 = down(128, 128)
        self.down5 = down(128, 128)
        self.down6 = down(128, 128)
        self.up = up(260, 128)
        self.outcc0 = outconv(128, n_classes)
        self.up0 = up(256, 128)
        self.outcc1 = outconv(128, n_classes)
        self.up1 = up(256, 128)
        self.outcc2 = outconv(128, n_classes)
        self.up2 = up(384, 128)
        self.outcc3 = outconv(128, n_classes)
        self.up3 = up(384, 128)
        self.outcc4 = outconv(128, n_classes)
        self.up4 = up(384, 128)
        self.outcc = outconv(128, n_classes)
        self.tpp = TPPblock(128)
        self.weights = torch.nn.Parameter(torch.ones(6))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        # x7 = self.dac(x7)
        x7 = self.tpp(x7)
        x = self.up(x7, x6)
        y1 = self.outcc0(F.relu(x))
        # print("y1.shape=", y1.shape)
        x = self.up0(x, x5)
        y2 = self.outcc1(F.relu(x))
        # print("y2.shape=", y2.shape)
        x = self.up1(x, x4)
        y3 = self.outcc2(F.relu(x))
        # print("y3.shape=", y3.shape)
        x = self.up2(x, x3)
        y4 = self.outcc3(F.relu(x))
        # print("y4.shape=", y4.shape)
        x = self.up3(x, x2)
        y5 = self.outcc4(F.relu(x))
        # print("y5.shape=", y5.shape)
        x = self.up4(x, x1)
        y = self.outcc(x)
        # print("y.shape=", y.shape)
        return y, [y5, y4, y3, y2, y1], x


class PostProcess(nn.Module):
    def __init__(self, actions_dict, chunk_size, gt_path):
        super().__init__()
        self.labels_dict_id2name = {}
        self.labels_dict_name2id = {}
        for act, ind in actions_dict.items():
            self.labels_dict_id2name[ind] = act
            self.labels_dict_name2id[act] = ind

        self.results_dict = dict()
        self.chunk_size = chunk_size
        self.gd_path = gt_path
        self.count = 0

    def start(self):
        self.results_dict = dict()
        self.count = 0

    def upsample_video_value(self, predictions, video_len, chunk_size):
        new_label_name_expanded = []
        prediction_swap = predictions.permute(1, 0)
        for i, ele in enumerate(prediction_swap):
            st = i * chunk_size
            end = st + chunk_size
            for j in range(st, end):
                new_label_name_expanded.append(ele)
        out_p = torch.stack(new_label_name_expanded).permute(1, 0)[:, :video_len]
        return out_p

    def dump_to_directory(self, path):
        print("Number of cats =", self.count)
        if len(self.results_dict.items()) == 0:
            return

        ne_dict = {}
        for video_id, video_value in self.results_dict.items():
            # pred_value = video_value[0]
            # label_count = video_value[1]
            # label_gt = video_value[2]
            # video_len = video_value[3]

            upped_pred_logit = self.upsample_video_value(video_value[0][:, :video_value[1]],
                                                         video_value[3], self.chunk_size)

            upped_pred_logit = torch.argmax(upped_pred_logit, dim=0)
            ne_dict[video_id] = upped_pred_logit

        for video_id, video_value in ne_dict.items():
            pred_value = video_value.detach().cpu().numpy()
            label_name_arr = [self.labels_dict_id2name[i.item()] for i in pred_value]

            out_path = os.path.join(path, video_id)
            with open(out_path, "w") as fp:
                fp.write("### Frame level recognition: ###\n")
                fp.write(" ".join(label_name_arr))


    @torch.no_grad()
    def forward(self, outputs, video_names, framewise_labels, counts, vid_len_arr):
        """ Perform the computation
        Parameters:
            :param outputs: raw outputs of the model
            :param start_frame:
            :param video_names:
            :param clip_length:
        """
        for output, vn, framewise_label, count, vid_len in zip(outputs, video_names, framewise_labels, counts, vid_len_arr):
            if vn in self.results_dict:
                self.count += 1

                prev_tensor, prev_count, prev_gt_labels, vid_len = self.results_dict[vn]
                output = torch.cat([prev_tensor, output], dim=1)
                framewise_label = torch.cat([prev_gt_labels, framewise_label])
                count = count + prev_count

            self.results_dict[vn] = [output, count, framewise_label, vid_len]

class C2FTrainer(BaseTrainer):

    def init_model(self, cfg):
        self.model = C2F_TCN(cfg.DATA.FEATURE_DIM, cfg.DATA.NUM_CLASSES)

    def init_criterion(self, cfg):
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction="none")

    # Override
    def get_optimizers(self, cfg):
        return [optim.Adam(self.model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)]

    # Override
    def get_schedulers(self, optimizers, cfg):
        optimizer = optimizers[0]  # Only one optmizer for C2F
        return [
            optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.TRAIN.STEP_SIZE, gamma=cfg.TRAIN.GAMMA
            )
        ]

    def get_c2f_ensemble_output(self, outp, weights):
        ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)

        for i, outp_ele in enumerate(outp[1]):
            upped_logit = F.upsample(outp_ele, size=outp[0].shape[-1], mode='linear', align_corners=True)
            ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)

        return ensemble_prob

    def get_train_loss_preds(self, batch_train_data, cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Unpack
        samples = batch_train_data[0].to(device).permute(0, 2, 1)
        count = batch_train_data[1].to(device)
        labels = batch_train_data[2].to(device)
        src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
        src_mask = src_mask.to(device)
        src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

        # Forward pass
        outputs_list = self.model(samples)
        outputs_ensemble = self.get_c2f_ensemble_output(outputs_list, cfg.MODEL.PARAMS.ENSEMBLE_WEIGHTS)
        predictions = torch.log(outputs_ensemble + 1e-10)  # log is necessary because ensemble gives softmax output
        ce_l = self.ce(predictions, labels)
        mse_l = 0.15 * torch.mean(torch.clamp(self.mse(predictions[:, :, 1:], predictions.detach()[:, :, :-1]),
                                              min=0, max=16) * src_msk_send[:, :, 1:])
        loss = ce_l + mse_l

        return loss, outputs_ensemble


    def accumulate_metrics(self, metrics_accum_dict, batch_train_data, predictions, cfg):
        """Hook method. Accumulate the necessary metrics into the metrics accumulator dict."""
        predicted = torch.argmax(predictions, dim=1)
        batch_target = batch_train_data[2]
        batch_target = batch_target.to(predicted.device)
        count = batch_train_data[1].to(predicted.device)
        src_mask = torch.arange(batch_target.shape[1], device=batch_target.device)[None, :] < count[:, None]
        src_mask = src_mask.to(predicted.device)


        metrics_accum_dict["correct"] += torch.sum(
            (predicted == batch_target).float()* src_mask).item()

        metrics_accum_dict["total"] += torch.sum(src_mask).item()

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
                scheduler.step()

            if (
                    (epoch + 1) % self.cfg.TRAIN.LOG_FREQ == 0
                    or (epoch + 1) == self.cfg.TRAIN.NUM_EPOCHS
            ):
                scores_to_log = [f"epoch loss = {epoch_loss / len(train_dataset_loader.dataset)}"]

                score_dict = self.score_accumulated_metrics(metrics_accum_dict, self.cfg)
                scores_to_log.extend([f"{k} = {v}" for k, v in score_dict.items()])

                logging.info(f"[epoch {epoch + 1}]: " + f", ".join(scores_to_log))

                # scores = self.predict()
                # logging.info(f"{scores}")

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
        actions_dict = dict(self.cfg.DATA.ACTIONS_DICT) if not isinstance(self.cfg.DATA.ACTIONS_DICT, dict) \
            else self.cfg.DATA.ACTIONS_DICT
        postprocessor = PostProcess(actions_dict, chunk_size=self.cfg.DATA.CHUNK_SIZE, gt_path=self.cfg.DATA.GT_PATH)

        def _init_fn(worker_id):
            np.random.seed(int(self.cfg.TRAIN.SEED))
        test_dataset = C2fDataset(self.cfg, is_train=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.cfg.TRAIN.BZ, shuffle=False,
                                                   pin_memory=True, num_workers=self.cfg.TRAIN.NUM_WORKER,
                                                   collate_fn=collate_fn_override,
                                                   worker_init_fn=_init_fn)

        self.model.eval()
        with torch.no_grad():
            self.model.cuda()
            for batch_train_data in test_loader:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                samples = batch_train_data[0].to(device).permute(0, 2, 1)
                vid_id = batch_train_data[5]
                count = batch_train_data[1].to(device)
                labels = batch_train_data[2].to(device)
                vid_lens = batch_train_data[3].to(device)

                outputs_list = self.model(samples)
                predictions = self.get_c2f_ensemble_output(outputs_list, self.cfg.MODEL.PARAMS.ENSEMBLE_WEIGHTS)

                postprocessor(predictions, vid_id, labels, count, vid_lens)

            postprocessor.dump_to_directory(f"{self.cfg.TRAIN.RESULT_DIR}")
        overlap = [0.1, 0.25, 0.5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct, total, edit = 0, 0, 0

        with open(self.cfg.DATA.VID_LIST_FILE_TEST, "r") as f:
            list_of_vids = f.read().splitlines()

        for vid in list_of_vids:
            gt_file = self.cfg.DATA.GT_PATH + vid
            gt_content = read_file(gt_file).split("\n")[0:-1]
            recog_file = os.path.join(self.cfg.TRAIN.RESULT_DIR, vid.split(".")[0])
            recog_content = read_file(recog_file).split("\n")[1].split()
            assert len(gt_content) == len(recog_content)

            total += len(gt_content)
            correct += sum(
                1 for gt, recog in zip(gt_content, recog_content) if gt == recog
            )

            edit += edit_score(recog_content, gt_content, self.cfg.DATA.BG_CLASS)

            for idx, thres in enumerate(overlap):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, thres, self.cfg.DATA.BG_CLASS)
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