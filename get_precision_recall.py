import json

from torch.utils.data import DataLoader

from esnli import *
import torch
from tqdm import tqdm

from latent_rationale.bert.bert_with_mask import RationaleSequenceClassifierOutput


@dataclass
class Arguments:
    model_name_or_path: str = field(
        default='/home/michael.me/interpretable_predictions/models/minibert_mask_all_l01_seed_42')
    generate_predictions: bool = False


def get_samples(inputs: dict, output: RationaleSequenceClassifierOutput):
    batch_input_ids = inputs['input_ids']
    batch_attention_mask = inputs['attention_mask']
    masks = output.z
    assert batch_input_ids.size(0) == masks.size(0)
    result = []
    for input_ids, attention_mask, mask in zip(batch_input_ids, batch_attention_mask, masks):
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
            'masked': masked
        })
    return result


def main(args: Arguments):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare(x):
        return {
            **{k: v.to(device) for k, v in x.items() if isinstance(v, torch.Tensor)},
            **{k: v for k, v in x.items() if isinstance(v, list)}
        }

    model = BertWithRationaleForSequenceClassification.from_pretrained(args.model_name_or_path)
    model = model.eval().to(device)

    ds = load_from_disk('data/esnli_all')
    test_dataset = ds['test']
    print(test_dataset[0])
    test_dl = DataLoader(test_dataset, batch_size=32, collate_fn=Collator(tokenizer), shuffle=True)

    precision = 0
    recall = 0
    count = 0
    samples = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dl)):
            inputs = prepare(batch)
            model = model.eval()
            output = model(**inputs, aggregate_mask_metrics=False)
            precision += output.precision
            recall += output.recall
            count += 1
            if args.generate_predictions:
                if i < 10:
                    samples += get_samples(inputs, output)
                elif i == 10:
                    with open('masked_examples.json', 'w+') as f:
                        json.dump(samples, f, indent=4, ensure_ascii=False)

    precision /= count
    recall /= count
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    f1 = 2 * (precision * recall) / (precision + recall)
    print(f'F1 = {f1}')


if __name__ == '__main__':
    # noinspection PyTypeChecker
    parser = HfArgumentParser(Arguments)
    # noinspection PyTypeChecker
    _args: Arguments = parser.parse_args()
    main(_args)
