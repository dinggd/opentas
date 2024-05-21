#!/usr/bin/python2.7

import torch
import numpy as np
import random
import os
import logging

class BatchGeneratorIterator(object):
    def __init__(self, batch_gen):
        self.batch_gen = batch_gen

    def __iter__(self):
        return self

    def __next__(self):
        if not self.batch_gen.has_next():
            raise StopIteration()
        return self.batch_gen.next_batch()
        

class BatchGenerator(object):
    def __init__(self, cfg):

        self.list_of_examples = list()
        self.index = 0
        self.num_classes = cfg.DATA.NUM_CLASSES
        self.actions_dict = (
            dict(cfg.DATA.ACTIONS_DICT)
            if not isinstance(cfg.DATA.ACTIONS_DICT, dict)
            else cfg.DATA.ACTIONS_DICT
        )
        self.gt_path = cfg.DATA.GT_PATH
        self.features_path = cfg.DATA.FEATURES_PATH
        self.sample_rate = cfg.DATA.SAMPLE_RATE
        self.batch_size = cfg.TRAIN.BZ

    def __iter__(self):
        # Note: Very rough wrapper around BatchGenerator. Cannot have multiple iterators over BatchGenerator as self.reset() will affect all iterators.
        self.reset()
        return BatchGeneratorIterator(self)

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file, aug_list=[]):
        file_ptr = open(vid_list_file, "r")
        self.list_of_examples = file_ptr.read().split("\n")[:-1]
        file_ptr.close()
        self.list_of_examples.extend(aug_list)

    def next_batch(self):
        batch = self.list_of_examples[self.index : self.index + self.batch_size]
        self.index += self.batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split(".")[0] + ".npy")
            with open(self.gt_path + vid, "r") as f:
                content = f.read().splitlines()
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input.append(features[:, :: self.sample_rate])
            batch_target.append(classes[:: self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(
            len(batch_input),
            np.shape(batch_input[0])[0],
            max(length_of_sequences),
            dtype=torch.float,
        )
        batch_target_tensor = torch.ones(
            len(batch_input), max(length_of_sequences), dtype=torch.long
        ) * (-100)
        mask = torch.zeros(
            len(batch_input),
            self.num_classes,
            max(length_of_sequences),
            dtype=torch.float,
        )
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, : np.shape(batch_input[i])[1]] = torch.from_numpy(
                batch_input[i]
            )
            batch_target_tensor[i, : np.shape(batch_target[i])[0]] = torch.from_numpy(
                batch_target[i]
            )
            mask[i, :, : np.shape(batch_target[i])[0]] = torch.ones(
                self.num_classes, np.shape(batch_target[i])[0]
            )

        batch_input_tensor = batch_input_tensor.cuda()
        batch_target_tensor = batch_target_tensor.cuda()
        mask = mask.cuda()

        return batch_input_tensor, batch_target_tensor, mask


##############################################
# Functions below are only for C2F
def collate_fn_override(data):
    """
       data:
    """
    data = list(filter(lambda x: x is not None, data))
    data_arr, count, labels, clip_length, start, video_id, labels_present_arr, aug_chunk_size = zip(*data)

    return torch.stack(data_arr), torch.tensor(count), torch.stack(labels), torch.tensor(clip_length),\
            torch.tensor(start), video_id, torch.stack(labels_present_arr), torch.tensor(aug_chunk_size, dtype=torch.int)


class C2fDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train=True, zoom_crop=(0.5, 2), smallest_cut=1.0):
        self.max_frames_per_video = cfg.DATA.MAX_FRAME_PER_VIDEO
        self.feature_size = cfg.DATA.FEATURE_DIM
        self.base_dir_name = cfg.DATA.FEATURES_PATH
        self.ground_truth_files_dir = cfg.DATA.GT_PATH
        self.chunk_size = cfg.DATA.CHUNK_SIZE
        self.num_class = cfg.DATA.NUM_CLASSES
        self.zoom_crop = zoom_crop
        self.smallest_cut = smallest_cut
        self.is_train = is_train
        self.actions_dict = cfg.DATA.ACTIONS_DICT
        self.cfg = cfg
        self.data = self.make_data_set(cfg.DATA.VID_LIST_FILE if self.is_train else cfg.DATA.VID_LIST_FILE_TEST)

    def make_data_set(self, fold_file_name):
        label_id_to_label_name = {}
        label_name_to_label_id_dict = {}
        for act, ind in self.actions_dict:
            label_id_to_label_name[ind] = act
            label_name_to_label_id_dict[act] = ind

        data = open(fold_file_name).read().split("\n")[:-1]
        data_arr = []
        num_video_not_found = 0
        for i, video_id in enumerate(data):
            video_id = video_id.split(".txt")[0]
            filename = os.path.join(self.ground_truth_files_dir, video_id + ".txt")

            with open(filename, 'r') as f:
                recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
                f.close()

            recog_content = [label_name_to_label_id_dict[e] for e in recog_content]
            total_frames = len(recog_content)

            if not os.path.exists(os.path.join(self.base_dir_name, video_id + ".npy")):
                logging.info("Not found video with id", os.path.join(self.base_dir_name, video_id + ".npy"))
                num_video_not_found += 1
                continue

            start_frame_arr = []
            end_frame_arr = []
            for st in range(0, total_frames, self.max_frames_per_video * self.chunk_size):
                start_frame_arr.append(st)
                max_end = st + (self.max_frames_per_video * self.chunk_size)
                end_frame = max_end if max_end < total_frames else total_frames
                end_frame_arr.append(end_frame)

            for st_frame, end_frame in zip(start_frame_arr, end_frame_arr):
                ele_dict = {'st_frame': st_frame, 'end_frame': end_frame, 'video_id': video_id,
                            'tot_frames': (end_frame - st_frame)}

                ele_dict["labels"] = np.array(recog_content, dtype=int)

                data_arr.append(ele_dict)

        logging.info("Number of videos logged in {} fold is {}".format('train' if self.is_train else 'test', len(data_arr)))
        logging.info("Number of videos not found in {} fold is {}".format('train' if self.is_train else 'test', num_video_not_found))
        return data_arr

    def getitem(self, index):  # Try to use this for debugging purpose
        ele_dict = self.data[index]
        st_frame = ele_dict['st_frame']
        end_frame = ele_dict['end_frame']

        data_arr = torch.zeros((self.max_frames_per_video, self.feature_size))
        label_arr = torch.ones(self.max_frames_per_video, dtype=torch.long) * -100

        image_path = os.path.join(self.base_dir_name, ele_dict['video_id'] + ".npy")
        elements = np.load(image_path)

        end_frame = min(end_frame, st_frame + (self.max_frames_per_video * self.chunk_size))
        len_video = end_frame - st_frame
        assert ele_dict['tot_frames'] == elements.shape[1]

        if np.random.randint(low=0, high=2) == 0 and (self.is_train):
            aug_start = np.random.uniform(low=0.0, high=1 - self.smallest_cut)
            aug_len = np.random.uniform(low=self.smallest_cut, high=1 - aug_start)
            aug_end = aug_start + aug_len
            min_possible_chunk_size = np.ceil(len_video / self.max_frames_per_video)
            max_chunk_size = int(1.0 * self.chunk_size / self.zoom_crop[0])
            min_chunk_size = max(int(1.0 * self.chunk_size / self.zoom_crop[1]), min_possible_chunk_size)
            aug_chunk_size = int(np.exp(np.random.uniform(low=np.log(min_chunk_size), high=np.log(max_chunk_size))))
            num_aug_frames = np.ceil(int(aug_len * len_video) / aug_chunk_size)
            if num_aug_frames > self.max_frames_per_video:
                num_aug_frames = self.max_frames_per_video
                aug_chunk_size = int(np.ceil(aug_len * len_video / num_aug_frames))

            aug_translate = 0
            aug_start_frame = st_frame + int(len_video * aug_start)
            aug_end_frame = st_frame + int(len_video * aug_end)
        else:
            aug_translate, aug_start_frame, aug_end_frame, aug_chunk_size = 0, st_frame, end_frame, self.chunk_size

        count = 0
        labels_present_arr = torch.zeros(self.num_class, dtype=torch.float32)
        for i in range(aug_start_frame, aug_end_frame, aug_chunk_size):
            end = min(aug_end_frame, i + aug_chunk_size)
            key = elements[:, i: end]
            values, counts = np.unique(ele_dict["labels"][i: end], return_counts=True)
            label_arr[count] = values[np.argmax(counts)]
            labels_present_arr[label_arr[count]] = 1
            data_arr[aug_translate + count, :] = torch.tensor(np.max(key, axis=-1), dtype=torch.float32)
            count += 1

        return data_arr, count, label_arr, ele_dict['tot_frames'], st_frame, ele_dict['video_id'], \
               labels_present_arr, aug_chunk_size

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data)
