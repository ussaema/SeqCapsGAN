import numpy as np
import pickle as pickle
import hickle
import time
import os
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json
from matplotlib.pyplot import imread


def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)

    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
        caption = " ".join(caption.split())  # replace multiple spaces

        caption_data.at[i, 'caption'] = caption.lower()
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)

    # delete captions if size is larger than max_length
    print("The number of captions before deletion: %d" % len(caption_data))
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print("The number of captions after deletion: %d" % len(caption_data))
    return caption_data


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            counter[w] += 1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print(('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold)))

    word_to_idx = {'<NULL>': 0, '<START>': 1, '<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print("Max length of caption: ", max_len)
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ")  # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec)
    print("Finished building caption vectors")
    return captions

def _build_emotion_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    emotions_rand = np.random.randint(0, 3, (n_examples,)).astype(np.int32)
    emotions = np.squeeze(np.eye(3)[emotions_rand.reshape(-1)])
    print("Finished building emotion vectors")
    return emotions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs

def combine_vocab(vocabs):
    vocab = []
    for v in vocabs:
        vocab += list(v.keys())
    vocab = list(set(vocab))
    vocab.remove('<NULL>')
    vocab.remove('<START>')
    vocab.remove('<END>')
    word_to_idx = {'<NULL>': 0, '<START>': 1, '<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    return word_to_idx


def load_coco_data(image_dir='image/train2014_resized/', caption_file='data/annotations/captions_train2014.json', splits=[1.], max_length=15):
    start_t = time.time()
    n_splits = len(splits)
    starts = [0.0]
    ends = []
    for idx in range(n_splits-1):
        starts.append(np.array(splits[:idx+1]).sum())
    for idx in range(n_splits):
        ends.append(splits[idx] + starts[idx])

    # about 80000 images and 400000 captions for train dataset
    annotations = _process_caption_data(caption_file=caption_file, image_dir=image_dir, max_length=max_length) #maximum length of caption(number of word). if caption is longer than max_length, deleted.
    annotations = [annotations[int(len(annotations)*start): int(len(annotations)*end)] for (start, end) in zip(starts, ends)]

    data = [{} for _ in splits]
    for data_idx, annotation in enumerate(annotations):
        data[data_idx]['word_to_idx'] = _build_vocab(annotations=annotation, threshold=1) # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.

        data[data_idx]['captions'] = _build_caption_vector(annotations=annotation, word_to_idx=data[data_idx]['word_to_idx'], max_length=max_length)

        data[data_idx]['emotions'] = _build_emotion_vector(annotations=annotation, word_to_idx=data[data_idx]['word_to_idx'], max_length=max_length)

        data[data_idx]['file_names'], id_to_idx = _build_file_names(annotation)

        data[data_idx]['image_idxs'] = _build_image_idxs(annotation, id_to_idx)

        # prepare reference captions to compute bleu scores later
        image_ids = {}
        references = {}
        references_emotions = []
        i = -1
        for caption, emotion, image_id in zip(annotation['caption'], data[data_idx]['emotions'], annotation['image_id']):
            if not image_id in image_ids:
                references_emotions.append(emotion)
                image_ids[image_id] = 0
                i += 1
                references[i] = []
            references[i].append(caption.lower() + ' .')
        data[data_idx]['references'] = references
        data[data_idx]['references_emotions'] = references_emotions
        data[data_idx]['image_files_names'] = np.array([data[data_idx]['file_names'][idx] for idx in data[data_idx]['image_idxs']])

    end_t = time.time()
    print("Elapse time: %.2f" %(end_t - start_t))
    return data if len(data) > 1 else data[0]

def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def sample_coco_minibatch(data, batch_size):
    data_size = data['features'].shape[0]
    mask = np.random.choice(data_size, batch_size)
    features = data['features'][mask]
    file_names = data['file_names'][mask]
    return features, file_names

def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' %(epoch+1))
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])
        f.write('Bleu_4: %f\n' %scores['Bleu_4'])
        #f.write('METEOR: %f\n' %scores['METEOR'])
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])
        f.write('CIDEr: %f\n\n' %scores['CIDEr'])

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print(('Loaded %s..' %path))
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print(('Saved %s..' %path))