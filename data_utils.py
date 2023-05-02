# coding = utf-8

import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, entity1, entity2, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            entity1: string. The untokenized text of the first entity. 
            entity2: string. The untokenized text of the second entity.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.entity1 = entity1
        self.entity2 = entity2
        self.label = label

class DiaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class WN18RRProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir, dataset):
        self.data_dir = data_dir
        self.dataset = dataset

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "train.txt")), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "valid.txt")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["_hypernym",
                "_derivationally_related_form",
                "_instance_hypernym",
                "_also_see",
                "_member_meronym",
                "_synset_domain_topic_of",
                "_has_part",
                "_member_of_domain_usage",
                "_member_of_domain_region",
                "_verb_group",
                "_similar_to"
        ]

    def get_label_map(self):
        label_map = {label: i for i, label in enumerate(self.get_labels(), 1)}
        label_map['[unk]'] = 0
        return label_map

    def get_id2label_map(self):
        return {i: label for label, i in self.get_label_map().items()}

    def get_label_size(self):
        return len(self.get_labels()) + 1

    def _create_examples(self, lines, set_type):
        examples = []
        entity_pair = lines[0][0]
        labels = lines[0][1]
        assert len(entity_pair) == len(labels)
        length = len(labels)
        for i in range(length):
            guid = "%s-%s" % (set_type, i+1)
            e_p = entity_pair[i]
            label = labels[i]
            examples.append(InputExample(guid=guid, entity1=e_p[0], entity2=e_p[1], label=label))
        return examples

    @classmethod
    def _read_txt(cls, input_file):
        '''
        read file
        return format :
        '''
        if os.path.exists(input_file) is False:
            return []
        data = []
        entity_pair = []
        label = []
        
        with open(input_file, "r", encoding = "utf-8") as f:
            for line in f:
                if len(line) == 0:
                    continue
                splits = line.strip().split('\t')
                # entity_pair.append([splits[0], splits[-1]])
                # Ignore the tag for different meaning
                entity_pair.append([splits[0].split('.')[0], splits[-1].split('.')[0]])
                label.append(splits[1])

            if len(entity_pair) > 0:
                data.append((entity_pair, label))
        return data

    def convert_to_feature(self, tokenizer, examples, max_seq_length=16):
        label_map = self.get_label_map()
        features = []
        for ex_index, example in enumerate(examples):

            labels = []
            valid_ids1 = []
            valid_ids2 = []

            entity1 = example.entity1
            entity2 = example.entity2
            label = example.label

            token1 = tokenizer.tokenize(entity1)
            token2 = tokenizer.tokenize(entity2)
            if len(token1) > max_seq_length - 2 or len(token2) > max_seq_length - 2:
                token1 = token1[:max_seq_length - 2]
                token2 = token2[:max_seq_length - 2]

            labels.append(label)
            for m in range(len(token1)):
                if m == 0:
                    valid_ids1.append(1)
                else:
                    valid_ids1.append(0)
            for m in range(len(token2)):
                if m == 0:
                    valid_ids2.append(1)
                else:
                    valid_ids2.append(0)

            tokens1 = ["[CLS]"] + token1 + ["[SEP]"]
            tokens2 = ["[CLS]"] + token2 + ["[SEP]"]
            valid_ids1 = [1] + valid_ids1 + [1]
            valid_ids2 = [1] + valid_ids2 + [1]
            input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
            input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
            label_ids = label_map[label]
            segment_ids = [0] * max_seq_length

            if len(input_ids1) < max_seq_length:
                input_ids1 += [0] * (max_seq_length - len(input_ids1))
                valid_ids1 += [0] * (max_seq_length - len(valid_ids1))

            if len(input_ids2) < max_seq_length:
                input_ids2 += [0] * (max_seq_length - len(input_ids2))
                valid_ids2 += [0] * (max_seq_length - len(valid_ids2))

            # if len(label_ids) <  max_seq_length:
            #     label_ids += [0] * (max_seq_length - len(label_ids))

            assert len(input_ids1) == max_seq_length
            assert len(input_ids2) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert len(label_ids) == max_seq_length
            assert len(valid_ids1) == max_seq_length
            assert len(valid_ids2) == max_seq_length
        

            features.append({
                "entity1_ids": torch.tensor(input_ids1, dtype=torch.long),
                "entity2_ids": torch.tensor(input_ids2, dtype=torch.long),
                "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
                "label_ids": torch.tensor(label_ids, dtype=torch.long),
                "valid_ids1": torch.tensor(valid_ids1, dtype=torch.long),
                "valid_ids2": torch.tensor(valid_ids2, dtype=torch.long),
                "entity1": example.entity1,
                "entity2": example.entity2,
                "label": example.label,
            })

        return features

    def get_dataloader(self, features, batch_size, mode='train', rank=0,  world_size=1):
        if mode == "train" and world_size > 1:
            features = features[rank::world_size]

        data_set = DiaDataset(features)
        sampler = RandomSampler(data_set)
        return DataLoader(data_set, sampler=sampler, batch_size=batch_size)

    def get_all_dataloader(self, tokenizer, args):
        #train
        train_examples = self.get_train_examples()
        train_features = self.convert_to_feature(tokenizer, train_examples, args.max_seq_len)
        train_dataloader = self.get_dataloader(train_features, mode="train", rank=args.rank,
                                                    world_size=args.world_size, batch_size=args.batch_size)

        #test
        test_examples = self.get_test_examples()
        test_features = self.convert_to_feature(tokenizer, test_examples, args.max_seq_len)
        test_dataloader = self.get_dataloader(test_features, mode="test", batch_size=args.batch_size)

        #dev
        dev_examples = self.get_dev_examples()
        dev_features = self.convert_to_feature(tokenizer, dev_examples, args.max_seq_len)
        dev_dataloader = self.get_dataloader(dev_features, mode="dev", batch_size=args.batch_size)

        return train_dataloader, test_dataloader, dev_dataloader