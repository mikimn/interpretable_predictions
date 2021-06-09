from dataclasses import dataclass, field
from os.path import join, abspath, isdir, dirname, realpath
from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoConfig, is_wandb_available, set_seed, IntervalStrategy, EvalPrediction, HfArgumentParser, \
    TrainerCallback, TrainerState, TrainerControl, PreTrainedModel
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer import Trainer, TrainingArguments

from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

# Faster tokenizer for optimization
from latent_rationale.bert.bert_with_mask import BertWithRationaleForSequenceClassification

_DIR = dirname(realpath(__file__))
WANDB_PROJECT_NAME = 'interpretable-transformer'
MINI_BERT = 'google/bert_uncased_L-4_H-256_A-4'

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer.model_input_names = ["input_ids", "token_type_ids", "attention_mask", "rationale_mask"]


@dataclass
class Arguments:
    tag: str = None
    model_name_or_path: str = field(default='bert-base-uncased')
    seed: int = field(default=42)
    max_seq_length: int = field(default=128)
    batch_size: int = field(default=32)
    premise_key: str = field(default='premise')
    hypothesis_key: str = field(default='hypothesis')
    wandb_project_name: str = field(default=WANDB_PROJECT_NAME)
    report_to_wandb: bool = field(default=True)
    rationale_type: str = field(default='all', metadata={
        'choices': ['all', 'premise', 'hypothesis', 'none', 'supervised']
    })
    rationale_strategy: str = field(default='independent', metadata={
        'choices': ['independent', 'contextual']
    })
    lambda_init: float = field(default=0.1)
    config_dir: str = field(default='configs')
    logging_dir: str = field(default='outputs')
    override_dataset: bool = False
    do_train: bool = True
    do_test: bool = True


class Bunch:
    def __init__(self, **kwds) -> None:
        self.__dict__.update(kwds)


@dataclass
class Collator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, List[int]]]):
        ignore_keys = ['premise_highlight', 'hypothesis_highlight']
        drop_keys = [key for key in features[0].keys() if isinstance(features[0][key], str)] + ['length']
        # print(drop_keys)
        encoded_inputs = {key: [example[key] for example in features] for key in features[0].keys()
                          if key not in (ignore_keys + drop_keys)}
        ignored_encoded_inputs = {key: [example[key] for example in features] for key in features[0].keys()
                                  if key in ignore_keys}
        batch = self.tokenizer.pad(
            encoded_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        batch.update(ignored_encoded_inputs)
        return batch


def per_class_accuracy_with_names(id_to_label: Dict = None):
    def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
        classes = np.unique(y_true)
        acc_dict = {}
        for c in classes:
            indices = (y_true == c)
            y_true_c = y_true[indices]
            y_pred_c = y_pred[indices]
            class_name = id_to_label[c] if id_to_label is not None else c
            acc_dict[f'accuracy_{class_name}'] = accuracy_score(y_true=y_true_c, y_pred=y_pred_c)
        return acc_dict

    return _per_class_accuracy


per_class_accuracy = per_class_accuracy_with_names()


def compute_metrics_default(pred: EvalPrediction):
    labels = pred.label_ids
    logits = pred.predictions
    # print(type(logits))
    if isinstance(logits, tuple):
        logits, = logits[:1]
    # print(logits.shape)
    if logits.shape != labels.shape:
        preds = logits.argmax(-1)
    else:
        preds = logits
    # print(labels)
    # print(preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        **per_class_accuracy(labels, preds)
    }


# def compute_metrics_binerized(pred):
#     y_true = pred.label_ids
#     y_pred = pred.predictions
#     accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
#     return {
#         'accuracy': accuracy,
#         **per_class_accuracy(y_true, y_pred),
#     }


def compute_metrics_wrap(compute_metrics_fn, preprocess_fn):
    def wrapper(pred):
        new_pred = preprocess_fn(pred)
        return compute_metrics_fn(new_pred)

    return wrapper


class ReportCallback(TrainerCallback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model: PreTrainedModel = kwargs.pop('model', None)
        if model is not None and 'mask_metrics' in model.__dict__:
            mask_metrics = model.mask_metrics
            if mask_metrics is not None and is_wandb_available():
                for k, v in mask_metrics.items():
                    mask_metrics[k] = mask_metrics[k] / model.mask_metrics_count
                self.logger.log(mask_metrics)
                model.mask_metrics = None
                model.mask_metrics_count = 0


def extract_rationale(sentence, flag_token='*'):
    if isinstance(sentence, list):
        return [extract_rationale(s, flag_token) for s in sentence]
    candidate = ' '.join([w[1:-1] for w in sentence.split() if w[0] == flag_token and w[-1] == flag_token])
    if len(candidate) == 0:
        return sentence
    return candidate


def main(config: Arguments):
    max_seq_length = config.max_seq_length
    batch_size = config.batch_size
    model_name_or_path = config.model_name_or_path
    premise_key = config.premise_key
    hypothesis_key = config.hypothesis_key
    report_to_wandb = config.report_to_wandb
    wandb_project_name = config.wandb_project_name
    seed = config.seed
    override_dataset = config.override_dataset
    rationale_type = config.rationale_type
    rationale_strategy = config.rationale_strategy
    lambda_init = config.lambda_init
    dataset_directory = join(_DIR, f'data/esnli_{rationale_type}')
    should_load_dataset = isdir(dataset_directory) and not override_dataset

    test_dataset_names = [('hans', 'validation')]

    # For reproducibility
    set_seed(seed)

    if should_load_dataset:
        print(f'Load preprocessed dataset from {abspath(dataset_directory)}')
        ds = load_from_disk(dataset_directory)
        save_dataset = False
    else:
        ds = load_dataset('latent_rationale/esnli/esnli_dataset.py')
        save_dataset = True

    def preprocess_function(examples):
        # Tokenize the texts
        if rationale_type == 'supervised':
            args = (extract_rationale(examples['premise_highlighted']),
                    extract_rationale(examples['hypothesis_highlighted']))
        else:
            args = (examples[premise_key], examples[hypothesis_key])
        result = tokenizer(*args, max_length=max_seq_length, truncation=True, return_length=True)
        result.update({
            'premise_highlight': [tokenizer.encode(x, add_special_tokens=False) for x in
                                  examples['premise_highlighted']],
            'hypothesis_highlight': [tokenizer.encode(x, add_special_tokens=False) for x in
                                     examples['hypothesis_highlighted']]
        })
        return result

    if save_dataset:
        assert not should_load_dataset
        ds = ds.map(preprocess_function, batched=True, load_from_cache_file=True)
        print('Saving preprocessed dataset...')
        ds.save_to_disk(dataset_directory)
    print(ds)

    train_dataset = ds['train']
    dev_dataset = ds['validation']
    test_dataset = ds['test']
    num_labels = train_dataset.features['label'].num_classes
    print('Sample:')
    print(train_dataset[0])

    bert_config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    bert_config.rationale_type = rationale_type
    bert_config.rationale_strategy = rationale_strategy
    bert_config.lambda_init = lambda_init
    bert_config.highlight_token = tokenizer.encode('*', add_special_tokens=False)[0]
    model = BertWithRationaleForSequenceClassification.from_pretrained(model_name_or_path, config=bert_config)

    report_to_wandb = report_to_wandb and is_wandb_available()
    if report_to_wandb:
        import wandb

        wandb.init(project=wandb_project_name, name=config.tag)
        wandb.config.update({
            k: v for k, v in config.__dict__.items()
        })

    training_args = TrainingArguments(
        output_dir='outputs',
        logging_steps=1000,
        eval_steps=2000,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        group_by_length=True,
        skip_memory_metrics=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        seed=seed,
        report_to=['wandb'] if report_to_wandb else []
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        data_collator=Collator(tokenizer=tokenizer, padding='max_length', max_length=max_seq_length),
        compute_metrics=compute_metrics_default
    )
    if report_to_wandb:
        trainer.add_callback(ReportCallback(wandb))

    if config.tag is None:
        config.tag = 'unspecified'
    model_dir = join('models', f'{config.tag.replace("-", "_")}_seed_{seed}')
    print(f'Model will be saved at {abspath(model_dir)}')

    if config.do_train:
        train_output = trainer.train()
        print(f'Train outputs:\n{train_output}')
        print(f'Saving model to {abspath(model_dir)}')
        trainer.save_model(model_dir)

    def test_preprocess_function(examples):
        # Tokenize the texts
        args = (examples[premise_key], examples[hypothesis_key])
        result = tokenizer(*args, max_length=max_seq_length, truncation=True, padding='max_length')
        return result

    if config.do_test:
        eval_results = trainer.evaluate(test_dataset, metric_key_prefix='snli_eval')
        print('Test Results:')
        print(eval_results)
        for test_ds_name, key in test_dataset_names:
            test_ds: Dataset = load_dataset(test_ds_name)[key]
            test_ds = test_ds.map(test_preprocess_function, batched=True)

            compute_metrics_old = trainer.compute_metrics
            if test_ds_name in ['hans']:
                # Binerization is needed because some datasets (like HANS, FEVER-Symmetric)
                # have 2 classes, while the model is trained on standard NLI (3 classes)
                def binerize_fn(pred: EvalPrediction):
                    print(f'Binerizing dataset {test_ds_name}')
                    logits = pred.predictions
                    if isinstance(logits, tuple):
                        logits, = logits[:1]
                    preds = logits.argmax(-1)
                    # (Entailment, Neutral, Contradiction)

                    # Neutral => Contradiction
                    preds[preds == 1] = 2
                    # Contradiction (2) => Contradiction (1)
                    preds[preds == 2] = 1

                    return EvalPrediction(predictions=preds, label_ids=pred.label_ids)

                trainer.compute_metrics = compute_metrics_wrap(compute_metrics_default, binerize_fn)

            eval_results = trainer.evaluate(test_ds, metric_key_prefix=f'{test_ds_name}_eval')
            # test_dl = trainer.get_test_dataloader(test_ds)
            print(f'Test Results for {test_ds_name}:')
            print(eval_results)
            # Restore
            trainer.compute_metrics = compute_metrics_old


if __name__ == '__main__':
    # noinspection PyTypeChecker
    parser = HfArgumentParser(Arguments)
    # noinspection PyTypeChecker
    _args: Arguments = parser.parse_args()
    main(_args)
