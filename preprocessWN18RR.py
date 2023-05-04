# WN18RR preprocessing

import numpy as np

file_path = "../Dataset/WN18RR/text/train.txt"
output_path = "../Dataset/WN18RR/text/wordDict.txt"

word_dict = set()

with open(file_path, "r", encoding = "utf-8") as f:
    for line in f:
        if len(line) == 0:
            continue
        splits = line.strip().split('\t')
        word_dict.add(splits[0].split('.')[0])
        word_dict.add(splits[-1].split('.')[0])


with open(output_path, "w", encoding = "utf-8") as f:
    for word in word_dict:
        f.write("{}\n".format(word))

