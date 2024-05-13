import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import logging

from model.base import BaseTrainer


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)
                )
                for s in range(num_stages - 1)
            ]
        )

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class MSTCNTrainer(BaseTrainer):
    def __init__(self, cfg):

        self.model = MultiStageModel(
            cfg.MODEL.PARAMS.NUM_STAGES,
            cfg.MODEL.PARAMS.NUM_LAYERS,
            cfg.MODEL.PARAMS.NUM_F_MAPS,
            cfg.DATA.FEATURE_DIM,
            cfg.DATA.NUM_CLASSES,
        )
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction="none")
        self.num_classes = cfg.DATA.NUM_CLASSES

    # Override
    def get_optimizers(self, learning_rate):
        return [optim.Adam(self.model.parameters(), lr=learning_rate)]

    # Override
    def get_schedulers(self, optimizers):
        return []

    # Override
    def get_train_loss_preds(self, batch_train_data):

        # Unpack
        batch_input, batch_target, mask = batch_train_data

        # Forward pass
        predictions = self.model(batch_input, mask)

        # predictions_old is for LwF
        loss = 0
        for i, p in enumerate(predictions):
            loss += self.ce(
                p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                batch_target.view(-1),
            )
            loss += 0.15 * torch.mean(
                torch.clamp(
                    self.mse(
                        F.log_softmax(p[:, :, 1:], dim=1),
                        F.log_softmax(p.detach()[:, :, :-1], dim=1),
                    ),
                    min=0,
                    max=16,
                )
                * mask[:, :, 1:]
            )
        return loss, predictions
        
    # Override
    def get_eval_preds(self, test_input, actions_dict, cfg):

        predictions = self.model(test_input, torch.ones(test_input.size()).cuda())
        predicted_classes = [
            list(actions_dict.keys())[
                list(actions_dict.values()).index(pred.item())
            ]
            for pred in torch.max(predictions[-1].data, 1)[1].squeeze()
        ] * cfg.DATA.SAMPLE_RATE

        return predicted_classes
