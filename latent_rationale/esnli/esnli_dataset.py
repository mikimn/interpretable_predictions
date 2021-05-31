import csv
import datasets

_URL = "https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/"


class Esnli(datasets.GeneratorBasedBuilder):
    """e-SNLI: Natural Language Inference with Natural Language Explanations corpus."""

    def _info(self):
        return datasets.DatasetInfo(
            description="eSNLI",
            features=datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                    "explanation_1": datasets.Value("string"),
                    "explanation_2": datasets.Value("string"),
                    "explanation_3": datasets.Value("string"),
                    "premise_highlighted": datasets.Value("string"),
                    "hypothesis_highlighted": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/OanaMariaCamburu/e-SNLI",
            citation="eSNLI",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        files = dl_manager.download_and_extract(
            {
                "train": [_URL + "esnli_train_1.csv", _URL + "esnli_train_2.csv"],
                "validation": [_URL + "esnli_dev.csv"],
                "test": [_URL + "esnli_test.csv"],
            }
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"files": files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"files": files["test"]},
            ),
        ]

    def _generate_examples(self, files):
        """Yields examples."""
        for filepath in files:
            with open(filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for _, row in enumerate(reader):
                    yield row["pairID"], {
                        "premise": row["Sentence1"],
                        "hypothesis": row["Sentence2"],
                        "label": row["gold_label"],
                        "explanation_1": row["Explanation_1"],
                        "explanation_2": row.get("Explanation_2", ""),
                        "explanation_3": row.get("Explanation_3", ""),
                        "premise_highlighted": row["Sentence1_marked_1"],
                        "hypothesis_highlighted": row["Sentence2_marked_1"],
                    }


