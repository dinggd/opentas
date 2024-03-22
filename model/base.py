from tqdm import tqdm
import logging
import torch
import numpy as np
from copy import deepcopy

from eval import read_file, edit_score, f_score

class BaseTrainer:
    '''NOTE: All concrete classes of this must initialize self.model and self.num_classes'''
    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
            
        self.model.train()
        self.model.to(device)

        optimizers = self.get_optimizers(learning_rate)
        schedulers = self.get_schedulers(optimizers)
        
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(
                    batch_size)
                batch_input, batch_target, mask = batch_input.to(
                    device), batch_target.to(device), mask.to(device)
                predictions = self.model(batch_input, mask)

                loss = self.calc_loss(predictions, batch_target, mask)

                epoch_loss += loss.item()
                
                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() *
                            mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            for scheduler in schedulers:
                scheduler.step(epoch_loss)
            batch_gen.reset()

        torch.save(self.model.state_dict(),
                   f'{save_dir}/epoch-{epoch + 1}.model')
        torch.save(optimizer.state_dict(),
                   f'{save_dir}/epoch-{epoch + 1}.opt')
        logging.info("[epoch %d]: epoch loss = %f,   acc = %f" % (
            epoch + 1, epoch_loss / len(batch_gen.list_of_examples), float(correct)/total))

    def get_optimizers(self, learning_rate):
        raise NotImplementedError()

    def get_schedulers(self, optimizers):
        raise NotImplementedError()

    def calc_loss(self, predictions, batch_target, mask):
        raise NotImplementedError()

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, 
                sample_rate, gt_path):

        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(
                    input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(
                        actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct,total,edit = 0, 0, 0

        for vid in list_of_vids:
            gt_file = gt_path + vid
            gt_content = read_file(gt_file).split('\n')[0:-1]
            recog_file = results_dir+'/' + vid.split('.')[0]
            recog_content = read_file(recog_file).split('\n')[1].split()

            for i in range(len(gt_content)):
                total += 1
                if gt_content[i] == recog_content[i]:
                    correct += 1

            edit += edit_score(recog_content, gt_content)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1

        final = []
        final.append((100*float(correct)/total))
        final.append((1.0*edit)/len(list_of_vids))
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])

            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100
            final.append(f1)

        return final