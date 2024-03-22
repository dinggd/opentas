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
        self.stage1 = SingleStageModel(
            num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(
            num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

    def update_fcs(self, num_classes):
        input_out, input_in, _ = self.stages[0].conv_1x1.weight.shape
        logging.info(
            f'Updating the stage input heads form {input_in} to {num_classes}')
        #  input shape change: no need for first stage, but necessary for following stages
        for layer in self.stages:
            weight = copy.deepcopy(layer.conv_1x1.weight.data)
            bias = copy.deepcopy(layer.conv_1x1.bias.data)
            new_conv_1x1 = nn.Conv1d(num_classes, input_out, 1)
            new_conv_1x1.weight.data[:,:input_in,:] = weight
            new_conv_1x1.bias.data = bias
            del layer.conv_1x1
            layer.conv_1x1 = new_conv_1x1

        
        output_out, output_in, _ = self.stages[0].conv_out.weight.shape
        logging.info(
            f'Updating the stage output heads form {output_out} to {num_classes}')
        #  out shape change: first stage
        weight = copy.deepcopy(self.stage1.conv_out.weight.data)
        bias = copy.deepcopy(self.stage1.conv_out.bias.data)
        new_fc = nn.Conv1d(output_in, num_classes, 1)
        new_fc.weight.data[:output_out] = weight
        new_fc.bias.data[:output_out] = bias
        del self.stage1.conv_out
        self.stage1.conv_out = new_fc
        # out shape change: following stages
        for layer in self.stages:
            weight = copy.deepcopy(layer.conv_out.weight.data)
            bias = copy.deepcopy(layer.conv_out.bias.data)
            new_fc = nn.Conv1d(output_in, num_classes, 1)
            new_fc.weight.data[:output_out] = weight
            new_fc.bias.data[:output_out] = bias
            del layer.conv_out
            layer.conv_out = new_fc


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(
            2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
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
            in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class MSTCNTrainer(BaseTrainer):
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(
            num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    # Override
    def get_optimizers(self, learning_rate):
        return [ optim.Adam(self.model.parameters(), lr=learning_rate) ]

    # Override
    def get_schedulers(self, optimizers):
        return []

    # Override
    def calc_loss(self, predictions, batch_target, mask, predictions_old=None):
        # predictions_old is for LwF
        loss = 0
        for i,p in enumerate(predictions):
            loss += self.ce(p.transpose(2, 1).contiguous().view(-1,
                            self.num_classes), batch_target.view(-1))
            loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(
                p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
            if predictions_old is not None:
                p_flat = p.transpose(2,1).contiguous.view(-1,self.num_classes)
                known_classes = predictions_old.shape[1]
                p_old_flat = predictions_old[i].transpose(2,1).contiguous.view(-1,known_classes)
                loss -= (torch.mul(torch.softmax(p_old_flat[:,:known_classes], dim=1), 
                                  torch.log_softmax(p_flat, dim=1)
                                  )*mask.flatten()).sum() / p_flat.shape[0]
        return loss
