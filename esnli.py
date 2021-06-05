from os.path import join, abspath, isdir, dirname, realpath

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, is_wandb_available, set_seed, IntervalStrategy
from transformers.trainer import Trainer, TrainingArguments

from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

# Faster tokenizer for optimization
from latent_rationale.bert.bert_with_mask import BertWithRationaleForSequenceClassification

_DIR = dirname(realpath(__file__))

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


class Bunch:
    def __init__(self, **kwds) -> None:
        self.__dict__.update(kwds)


def main(config):
    max_seq_length = config.max_seq_length
    model_name_or_path = config.model_name_or_path
    premise_key = config.premise_key
    hypothesis_key = config.hypothesis_key
    report_to_wandb = config.report_to_wandb
    wandb_project_name = config.wandb_project_name
    seed = config.seed
    override_dataset = True
    mask_premise = True
    mask_hypothesis = False
    dataset_directory = join(_DIR, 'data/esnli')
    should_load_dataset = isdir(dataset_directory) and not override_dataset

    set_seed(seed)

    if should_load_dataset:
        print(f'Load preprocessed dataset from {abspath(dataset_directory)}')
        ds = load_from_disk(dataset_directory)
        save_dataset = False
    else:
        ds = load_dataset('latent_rationale/esnli/esnli_dataset.py')
        save_dataset = True

    def unary_not(lst):
        return [1 - i for i in lst]

    def list_plus(lst1, lst2):
        return [i + j for i, j in zip(lst1, lst2)]

    def preprocess_function(examples):
        # Tokenize the texts
        args = (examples[premise_key], examples[hypothesis_key])
        result = tokenizer(*args, max_length=max_seq_length, truncation=True, return_length=True)
        # [B, T]
        type_ids = result['token_type_ids']
        premise_rationale_mask = [unary_not(i) for i in type_ids] if mask_premise else [[0] * len(i) for i in type_ids]
        hypothesis_rationale_mask = type_ids if mask_hypothesis else [[0] * len(i) for i in type_ids]
        # Logical OR
        # result['rationale_mask'] = [list_plus(i, j) for i, j in zip(premise_rationale_mask, hypothesis_rationale_mask)]
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
    model = BertWithRationaleForSequenceClassification.from_pretrained(model_name_or_path, config=bert_config)

    report_to_wandb = report_to_wandb and is_wandb_available()
    if report_to_wandb:
        import wandb

        wandb.init(project=wandb_project_name)
        wandb.config.update({
            k: v for k, v in config.__dict__.items()
        })

    training_args = TrainingArguments(
        output_dir='outputs',
        logging_steps=1000,
        eval_steps=2000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        # remove_unused_columns=True,
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
        tokenizer=tokenizer
    )

    model_dir = join('models', f'bert_interpretable_seed_{seed}')
    print(f'Model will be saved at {abspath(model_dir)}')

    train_output = trainer.train()
    print(f'Train outputs:\n{train_output}')
    print(f'Saving model to {abspath(model_dir)}')
    trainer.save_model(model_dir)


if __name__ == '__main__':
    # noinspection PyTypeChecker
    main(Bunch(
        max_seq_length=128,
        model_name_or_path='bert-base-uncased',
        premise_key='premise',
        hypothesis_key='hypothesis',
        report_to_wandb=True,
        wandb_project_name='interpretable-transformer',
        seed=42
    ))
