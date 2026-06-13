"""
================================================================

"""

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold
import os
import copy
import numpy as np
import cv2
import pickle
import glob
import h5py

class Dataset(TorchDataset):
    def __init__(self, dataset_location, input_size=128):
        self.images = []
        self.mask_labels = []
        self.series_uid = []
        self.split_tags = []
        self.aug_ids = []
        self.case_ids = []

        max_bytes = 2 ** 31 - 1
        data = {}
        print("Loading file", dataset_location)
        bytes_in = bytearray(0)
        file_size = os.path.getsize(dataset_location)
        with open(dataset_location, 'rb') as f_in:
            for _ in range(0, file_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        new_data = pickle.loads(bytes_in)
        data.update(new_data)

        for key, value in data.items():
            image = value['image']
            masks = value['masks']

            image = self._normalize_image(image)
            normalized_masks = []
            for mask in masks:
                normalized_masks.append(self._normalize_mask(mask, input_size))

            self.images.append(pad_im(image, input_size))
            self.mask_labels.append(normalized_masks)
            self.series_uid.append(value['series_uid'])
            self.split_tags.append(value.get('split_tag', None))
            self.aug_ids.append(value.get('aug_id', 0))
            self.case_ids.append(value.get('case_id', value['series_uid'].split('_')[0]))

        assert (len(self.images) == len(self.mask_labels) == len(self.series_uid))

        for i, image in enumerate(self.images):
            img_min, img_max = np.min(image), np.max(image)
            assert img_max <= 255.01 and img_min >= -0.01, \
                f"Image {i} out of expected range: [{img_min:.2f}, {img_max:.2f}]"
        for i, mask_list in enumerate(self.mask_labels):
            for j, mask in enumerate(mask_list):
                m_min, m_max = np.min(mask), np.max(mask)
                assert m_max <= 1.01 and m_min >= -0.01, \
                    f"Mask {i}-{j} out of expected range: [{m_min:.2f}, {m_max:.2f}]"

        self.num_annotators = max((len(m) for m in self.mask_labels), default=4)

        n_train_val = sum(1 for t in self.split_tags if t == 'train_val')
        n_test_dis = sum(1 for t in self.split_tags if t == 'test_disagree')
        n_legacy = sum(1 for t in self.split_tags if t is None)
        print(f"Detected {self.num_annotators} annotators, {len(self.images)} samples "
              f"(train_val={n_train_val}, test_disagree={n_test_dis}, legacy={n_legacy})")

        del new_data, data

    @staticmethod
    def _normalize_image(image):
        image = image.astype(np.float64)
        vmin, vmax = image.min(), image.max()
        if vmin >= -0.5 and vmax <= 255.5 and vmax > 1.5:
            return np.clip(image, 0, 255).astype(np.float64)
        if vmin >= -0.01 and vmax <= 1.01:
            return (image * 255.0).clip(0, 255).astype(np.float64)
        if vmax - vmin < 1e-8:
            return np.zeros_like(image, dtype=np.float64)
        normalized = (image - vmin) / (vmax - vmin) * 255.0
        return normalized.clip(0, 255).astype(np.float64)

    @staticmethod
    def _normalize_mask(mask, input_size):
        mask = mask.astype(np.float64)
        vmin, vmax = mask.min(), mask.max()
        if vmin >= -0.01 and vmax <= 1.01:
            mask = np.clip(mask, 0, 1)
        elif vmin >= -0.5 and vmax <= 255.5:
            mask = mask / 255.0
            mask = np.clip(mask, 0, 1)
        else:
            threshold = (vmin + vmax) / 2.0
            mask = (mask > threshold).astype(np.float64)
        return pad_im(mask, input_size)

    def __getitem__(self, index):
        image = copy.deepcopy(self.images[index])
        mask_labels = copy.deepcopy(self.mask_labels[index])
        series_uid = self.series_uid[index]

        target = self.num_annotators
        cur = len(mask_labels)
        if cur < target:
            last = mask_labels[-1]
            mask_labels = mask_labels + [copy.deepcopy(last) for _ in range(target - cur)]
        elif cur > target:
            mask_labels = mask_labels[:target]

        return image, mask_labels, series_uid

    def __len__(self):
        return len(self.images)

def pad_im(image, size, value=0):
    shape = image.shape
    if len(shape) == 2:
        h, w = shape
    else:
        h, w, c = shape

    if h == w:
        if h == size:
            padded_im = image
        else:
            padded_im = cv2.resize(image, (size, size), cv2.INTER_CUBIC)
    else:
        if h > w:
            pad_1 = (h - w) // 2
            pad_2 = (h - w) - pad_1
            padded_im = cv2.copyMakeBorder(image, 0, 0, pad_1, pad_2, cv2.BORDER_CONSTANT, value=value)
        else:
            pad_1 = (w - h) // 2
            pad_2 = (w - h) - pad_1
            padded_im = cv2.copyMakeBorder(image, pad_1, pad_2, 0, 0, cv2.BORDER_CONSTANT, value=value)
    if padded_im.shape[0] != size:
        padded_im = cv2.resize(padded_im, (size, size), cv2.INTER_CUBIC)
    return padded_im

class NPCDataset(TorchDataset):
    """NPC-170 dataset in h5 format."""

    _MODALITY_KEYS = ('t1', 't1c', 't2')
    _LABEL_KEYS = ('label_a1', 'label_a2', 'label_a3', 'label_a4')

    def __init__(self, dataset_root, input_size=128, split_dir='training_2d', drop_empty=True):
        self.input_size = input_size
        self.images = []
        self.mask_labels = []
        self.series_uid = []
        self.split_tags = []
        self.aug_ids = []
        self.case_ids = []

        data_dir = os.path.join(dataset_root, split_dir)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"NPC data directory not found: {data_dir}")

        files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        if not files:
            raise RuntimeError(f"No .h5 files found under {data_dir}")
        print(f"[NPCDataset] Found {len(files)} h5 files in {data_dir}")

        n_dropped = 0
        for fp in files:
            with h5py.File(fp, 'r') as f:
                image = np.stack([np.asarray(f[k]) for k in self._MODALITY_KEYS], axis=-1).astype(np.float64)
                masks = [np.asarray(f[k]).astype(np.float64) for k in self._LABEL_KEYS]
            if drop_empty and all(m.sum() == 0 for m in masks):
                n_dropped += 1
                continue
            image = self._normalize_image_npc(image)
            masks = [Dataset._normalize_mask(m, input_size) for m in masks]
            self.images.append(pad_im(image, input_size))
            self.mask_labels.append(masks)
            uid = os.path.splitext(os.path.basename(fp))[0].replace('Sample_', '')
            uid = uid.replace('_slice_', '_')
            self.series_uid.append(uid)
            self.split_tags.append(None)
            self.aug_ids.append(0)
            self.case_ids.append(uid.split('_')[0])

        print(f"[NPCDataset] Dropped {n_dropped} empty, kept {len(self.images)}")
        self.num_annotators = len(self._LABEL_KEYS)

    @staticmethod
    def _normalize_image_npc(image_hwc):
        out = np.zeros_like(image_hwc, dtype=np.float64)
        for c in range(image_hwc.shape[-1]):
            ch = image_hwc[..., c]
            vmin, vmax = ch.min(), ch.max()
            if vmax - vmin < 1e-8:
                out[..., c] = 0
            else:
                out[..., c] = (ch - vmin) / (vmax - vmin) * 255.0
        return out.clip(0, 255)

    def __len__(self):
        return len(self.images)

    # NPCDataset.__getitem__
    def __getitem__(self, index):
        return (copy.deepcopy(self.images[index]),
                copy.deepcopy(self.mask_labels[index]),
                self.series_uid[index])

class DatasetSpliter():
    _PATIENT_SPLIT_DATASETS = ('LIDC', 'LungNodule', 'NPC', 'MMIS', 'NPC170', 'QUBIQ')
    _NPC_DATASETS = ('NPC', 'MMIS', 'NPC170')

    def __init__(self, opt, input_size):
        self.opt = opt
        print(f"[DatasetSpliter] DATASET={opt.DATASET!r}, DATA_PATH={opt.DATA_PATH!r}")

        if opt.DATASET in self._NPC_DATASETS:
            self.dataset = NPCDataset(
                dataset_root=opt.DATA_PATH,
                input_size=input_size,
                split_dir=getattr(opt, 'SPLIT_DIR', 'training_2d'),
                drop_empty=getattr(opt, 'DROP_EMPTY', True),
            )
        else:
            self.dataset = Dataset(dataset_location=opt.DATA_PATH, input_size=input_size)

        self.splits = []

        has_split_tag = any(t == 'train_val' or t == 'test_disagree' for t in self.dataset.split_tags)

        if has_split_tag:
            print("[DatasetSpliter] Using split_tag-based split (lung_nodule_v2/v3)")
            train_val_idx = [i for i, t in enumerate(self.dataset.split_tags) if t == 'train_val']
            test_disagree_idx = [i for i, t in enumerate(self.dataset.split_tags) if t == 'test_disagree']
            print(f"  train_val pool: {len(train_val_idx)} samples")
            print(f"  test_disagree (fixed): {len(test_disagree_idx)} samples")

            uid_dict = {}
            for idx in train_val_idx:
                cid = self.dataset.case_ids[idx]
                uid_dict.setdefault(cid, []).append(idx)
            cids = list(uid_dict.keys())
            np.random.seed(opt.RANDOM_SEED)
            np.random.shuffle(cids)
            print(f"  Unique cases in train_val: {len(cids)}")

            if opt.KFOLD == 1:
                split_point = int(len(cids) * 0.8)
                train_cids = cids[:split_point]
                val_cids = cids[split_point:]
                train_index = []
                val_index = []
                for c in train_cids:
                    train_index += uid_dict[c]
                for c in val_cids:
                    val_index += uid_dict[c]
                test_index = val_index + test_disagree_idx
                self.splits.append({
                    'train_index': train_index,
                    'test_index': test_index,
                    'val_index': val_index,
                    'test_disagree_index': test_disagree_idx,
                })
                print(f"  Single fold: {len(train_cids)} train cases ({len(train_index)} samples), "
                      f"{len(val_cids)} val cases ({len(val_index)} samples), "
                      f"+ {len(test_disagree_idx)} test_disagree → test_total={len(test_index)}")
            else:
                self.kf = KFold(n_splits=opt.KFOLD, shuffle=False)
                for (tr_pid_idx, te_pid_idx) in self.kf.split(np.arange(len(cids))):
                    train_index = []
                    val_index = []
                    for pi in tr_pid_idx:
                        train_index += uid_dict[cids[pi]]
                    for pi in te_pid_idx:
                        val_index += uid_dict[cids[pi]]
                    test_index = val_index + test_disagree_idx
                    self.splits.append({
                        'train_index': train_index,
                        'test_index': test_index,
                        'val_index': val_index,
                        'test_disagree_index': test_disagree_idx,
                    })

        elif opt.DATASET in self._PATIENT_SPLIT_DATASETS:
            uid_dict = {}
            for idx, uid in enumerate(self.dataset.series_uid):
                pid = uid.split('_')[0]
                uid_dict.setdefault(pid, []).append(idx)
            pids = list(uid_dict.keys())
            np.random.seed(opt.RANDOM_SEED)
            np.random.shuffle(pids)

            if opt.KFOLD == 1:
                split_point = int(len(pids) * 0.8)
                train_pids = pids[:split_point]
                test_pids = pids[split_point:]
                train_index = []; test_index = []
                for pid in train_pids:
                    train_index += uid_dict[pid]
                for pid in test_pids:
                    test_index += uid_dict[pid]
                self.splits.append({'train_index': train_index, 'test_index': test_index})
                print("Single fold: {} train pids, {} test pids".format(len(train_pids), len(test_pids)))
            else:
                self.kf = KFold(n_splits=opt.KFOLD, shuffle=False)
                for (tr_pi, te_pi) in self.kf.split(np.arange(len(pids))):
                    train_index = []; test_index = []
                    for pi in tr_pi:
                        train_index += uid_dict[pids[pi]]
                    for pi in te_pi:
                        test_index += uid_dict[pids[pi]]
                    self.splits.append({'train_index': train_index, 'test_index': test_index})
        else:
            indices = list(range(len(self.dataset)))
            np.random.seed(opt.RANDOM_SEED)
            np.random.shuffle(indices)
            if opt.KFOLD == 1:
                split_point = int(len(indices) * 0.8)
                self.splits.append({
                    'train_index': indices[:split_point],
                    'test_index': indices[split_point:]})
            else:
                self.kf = KFold(n_splits=opt.KFOLD, shuffle=False)
                for (train_index, test_index) in self.kf.split(np.arange(len(self.dataset))):
                    self.splits.append({
                        'train_index': [indices[i] for i in train_index.tolist()],
                        'test_index': [indices[i] for i in test_index.tolist()]})

    def get_datasets(self, fold_idx, split_mode='all'):
        train_indices = self.splits[fold_idx]['train_index']
        if split_mode == 'val_only' and 'val_index' in self.splits[fold_idx]:
            test_indices = self.splits[fold_idx]['val_index']
            print(f"  [SPLIT_MODE=val_only] using {len(test_indices)} consensus samples")
        elif split_mode == 'disagree' and 'test_disagree_index' in self.splits[fold_idx]:
            test_indices = self.splits[fold_idx]['test_disagree_index']
            print(f"  [SPLIT_MODE=disagree] using {len(test_indices)} disagreement samples")
        else:
            test_indices = self.splits[fold_idx]['test_index']
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(self.dataset, batch_size=self.opt.TRAIN_BATCHSIZE, sampler=train_sampler)
        test_loader = DataLoader(self.dataset, batch_size=self.opt.VAL_BATCHSIZE, sampler=test_sampler)
        print("Number of training/test patches:", (len(train_indices), len(test_indices)))
        return train_loader, test_loader