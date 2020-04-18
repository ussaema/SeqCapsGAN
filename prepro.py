from core.utils import load_coco_data, load_senticap_data, build_vocab
import pickle
import os
import argparse

parser = argparse.ArgumentParser(description='Dataset preprocessing')

parser.add_argument('--train_image_dir', type=str, default='data/images/train2014_resized/')
parser.add_argument('--val_image_dir', type=str, default='data/images/val2014_resized/')
parser.add_argument('--coco_dataset_train_dir', type=str, default='data/annotations/captions_train2014.json')
parser.add_argument('--coco_dataset_val_dir', type=str, default='data/annotations/captions_val2014.json')
parser.add_argument('--senticap_dataset_dir', type=str, default='data/annotations/senticap_dataset.json')
parser.add_argument('--output_dir', type=str, default='data/')

parser.add_argument('--max_length', type=int, default=25)

parser.add_argument('--coco_dataset_portions', nargs='+', help='train, validation and test portions between 0 and 1, the sum of the second and third entries must not be larger than 1.', type=float, default=[1.,0.8,0.2])
parser.add_argument('--senticap_dataset_portions', nargs='+', help='train, validation and test portions between 0 and 1, the sum of the first second and third entries must not be larger than 1.', type=float, default=[0.8,0.19,0.01])

args = parser.parse_args()

def main():
    # create vocab
    word_to_idx = build_vocab(train_image_dir=args.train_image_dir, val_image_dir=args.val_image_dir,
                              coco_dataset_files=[args.coco_dataset_train_dir, args.coco_dataset_val_dir],
                              senticap_dataset_files=[args.senticap_dataset_dir], max_length=args.max_length)
    with open(os.path.join(args.output_dir, 'word_to_idx.pkl'), 'wb') as f:
        pickle.dump(word_to_idx, f)
    print("*" * 16, "Vocabulary built", "*" * 16)

    # load senticap dataset
    train_senticap_data, val_senticap_data, test_senticap_data = load_senticap_data(vocab=word_to_idx, train_image_dir=args.train_image_dir,
                                                              val_image_dir=args.val_image_dir,
                                                              caption_file=args.senticap_dataset_dir, splits=args.senticap_dataset_portions,
                                                              max_length=args.max_length)
    with open(os.path.join(args.output_dir, 'train_senticap_data.pkl'), 'wb') as f:
        pickle.dump(train_senticap_data, f)
    with open(os.path.join(args.output_dir, 'val_senticap_data.pkl'), 'wb') as f:
        pickle.dump(val_senticap_data, f)
    with open(os.path.join(args.output_dir, 'test_senticap_data.pkl'), 'wb') as f:
        pickle.dump(test_senticap_data, f)

    # load senticap dataset
    train_coco_data = load_coco_data(vocab=word_to_idx, image_dir=args.train_image_dir,
                                     caption_file=args.coco_dataset_train_dir, splits=[args.coco_dataset_portions[0]],
                                     max_length=args.max_length)
    val_coco_data, test_coco_data = load_coco_data(vocab=word_to_idx, image_dir=args.val_image_dir,
                                                   caption_file=args.coco_dataset_val_dir, splits=args.coco_dataset_portions[1:],
                                                   max_length=args.max_length)
    with open(os.path.join(args.output_dir, 'train_coco_data.pkl'), 'wb') as f:
        pickle.dump(train_coco_data, f)
    with open(os.path.join(args.output_dir, 'val_coco_data.pkl'), 'wb') as f:
        pickle.dump(val_coco_data, f)
    with open(os.path.join(args.output_dir, 'test_coco_data.pkl'), 'wb') as f:
        pickle.dump(test_coco_data, f)
    print("*" * 16, "Dataset loaded", "*" * 16)

if __name__ == "__main__":
    main()