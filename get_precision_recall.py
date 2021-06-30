import json
from os.path import join, isdir, dirname, realpath

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from esnli import *
from latent_rationale.bert.bert_with_mask import RationaleSequenceClassifierOutput

_DIR = dirname(realpath(__file__))


@dataclass
class Arguments:
    model_name_or_path: str = field(
        default='/home/michael.me/interpretable_predictions/models/minibert_mask_all_l01_seed_42')
    generate_predictions: bool = False


def get_samples(inputs: dict, output: RationaleSequenceClassifierOutput):
    batch_input_ids = inputs['input_ids']
    batch_labels = inputs['labels']
    batch_attention_mask = inputs['attention_mask']
    masks = output.z
    assert batch_input_ids.size(0) == masks.size(0)
    result = []
    for input_ids, label, attention_mask, mask in zip(batch_input_ids, batch_labels, batch_attention_mask, masks):
        assert attention_mask.size() == input_ids.size()
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        masked_ids = input_ids.clone()
        masked_ids[masked_ids == tokenizer.pad_token_id] = -1
        masked_ids = input_ids * mask
        # masked_ids = masked_ids[attention_mask]
        masked_ids[masked_ids == 0] = tokenizer.mask_token_id
        masked_ids = masked_ids.int()
        # print(masked_ids)
        masked = tokenizer.decode(masked_ids[attention_mask == 1])
        result.append({
            'original': tokenizer.decode(input_ids[attention_mask == 1]),
            'masked': masked,
            'label': ['entailment', 'neutral', 'contradiction'][label]
        })
    return result


max_seq_length = 128


def preprocess_function(examples):
    # Tokenize the texts
    args = (examples['premise'], examples['hypothesis'])
    result = tokenizer(*args, max_length=max_seq_length, truncation=True, return_length=True)
    return result


def main(args: Arguments):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare(x):
        return {
            **{k: v.to(device) for k, v in x.items() if isinstance(v, torch.Tensor)},
            **{k: v for k, v in x.items() if isinstance(v, list)}
        }

    model_name_or_path = args.model_name_or_path
    model = BertWithRationaleForSequenceClassification.from_pretrained(model_name_or_path)
    model = model.eval().to(device)

    # # Hard Subset
    # predictions_file_name = 'masked_examples_hard.json'
    # dataset_name = 'latent_rationale/dataset/snli_hard.py'
    # dataset_directory = join(_DIR, f'data/snli_hard')
    # dataset_tag = 'test'

    # eSNLI
    predictions_file_name = 'masked_examples.json'
    dataset_name = 'latent_rationale/esnli/esnli_dataset.py'
    dataset_directory = join(_DIR, f'data/esnli_all')
    dataset_tag = 'test'

    if not isdir(dataset_directory):
        # ds = load_from_disk('data/esnli_all')
        ds = load_dataset(dataset_name)
        ds = ds.map(preprocess_function, batched=True, load_from_cache_file=True)
    else:
        ds = load_from_disk(dataset_directory)

    test_dataset = ds[dataset_tag]
    print(test_dataset[0])
    test_dl = DataLoader(test_dataset, batch_size=32, collate_fn=Collator(tokenizer), shuffle=True)

    precision = 0
    recall = 0
    count = 0
    samples = []
    all_overlap = 0
    all_overlap_hypothesis = 0
    all_overlap_premise = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dl)):
            inputs = prepare(batch)
            model = model.eval()
            output: RationaleSequenceClassifierOutput = model(**inputs, aggregate_mask_metrics=False)
            if output.precision:
                precision += output.precision
            if output.recall:
                recall += output.recall

            z_mask = output.z  # [B, T]
            attention_mask = inputs['attention_mask']
            hypothesis_mask = inputs['token_type_ids']  # [B, T]
            premise_mask = attention_mask - hypothesis_mask  # [B, T]
            lengths = attention_mask.sum(-1)  # [B]
            hypothesis_lengths = hypothesis_mask.sum(-1)  # [B]
            premise_lengths = premise_mask.sum(-1)  # [B]
            all_overlap += (z_mask.sum(-1) / lengths).mean()
            all_overlap_hypothesis += ((hypothesis_mask * z_mask).sum(-1) / hypothesis_lengths).mean()
            all_overlap_premise += ((premise_mask * z_mask).sum(-1) / premise_lengths).mean()

            count += 1
            if args.generate_predictions:
                if i < 10:
                    samples += get_samples(inputs, output)
                elif i == 10:
                    with open(join(model_name_or_path, predictions_file_name), 'w+') as f:
                        json.dump(samples, f, indent=4, ensure_ascii=False)

    all_overlap /= count
    all_overlap_hypothesis /= count
    all_overlap_premise /= count
    precision /= count
    recall /= count
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Selected: {all_overlap}')
    print(f'Hypothesis Selected: {all_overlap_hypothesis}')
    print(f'Premise Selected: {all_overlap_premise}')

    f1 = 2 * (precision * recall) / (precision + recall)
    print(f'F1 = {f1}')


if __name__ == '__main__':
    # noinspection PyTypeChecker
    parser = HfArgumentParser(Arguments)
    # noinspection PyTypeChecker
    _args: Arguments = parser.parse_args()
    main(_args)
