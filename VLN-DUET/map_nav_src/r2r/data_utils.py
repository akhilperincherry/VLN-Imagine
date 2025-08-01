import os
import json
import numpy as np
import h5py
import jsonlines

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if dataset == 'r2r':
                with open(os.path.join(anno_dir, 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'fgr2r':
                # print('Reading FGR2R')
                with open(os.path.join(anno_dir, 'FGR2R_%s.json' % split)) as f:
                    new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

            if not is_test:
                if dataset == 'r4r' and split == 'val_unseen':
                    ridxs = np.random.permutation(len(new_data))[:200]
                    new_data = [new_data[ridx] for ridx in ridxs]
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True, aug_flag = False):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            if j>2 and not aug_flag:
                print('construct_instrs() for aug: Skipping split: ', splits, 'pathid_instrid: ',  item['path_id'], '-', j)
                continue
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j]#[:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data


class ImaginationImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, path_id_instr_idx):
        # key = '%s_%s' % (path_id, instr_idx) #instr_idx is 0, 1 or 2.
        key = path_id_instr_idx #This should be a string.
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft