import datasets


class HansConfig(datasets.BuilderConfig):
    """BuilderConfig for HANS."""

    def __init__(self, **kwargs):
        """BuilderConfig for HANS.
            Args:
        .
              **kwargs: keyword arguments forwarded to super.
        """
        super(HansConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class Hans(datasets.GeneratorBasedBuilder):
    """Hans: Heuristic Analysis for NLI Systems."""

    def _info(self):
        return datasets.DatasetInfo(
            description="HANS",
            features=datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["entailment", "non-entailment"]),
                    "parse_premise": datasets.Value("string"),
                    "parse_hypothesis": datasets.Value("string"),
                    "binary_parse_premise": datasets.Value("string"),
                    "binary_parse_hypothesis": datasets.Value("string"),
                    "heuristic": datasets.Value("string"),
                    "subcase": datasets.Value("string"),
                    "template": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/tommccoy1/hans",
            citation="HANS",
        )

    def _vocab_text_gen(self, filepath):
        for _, ex in self._generate_examples(filepath):
            yield " ".join([ex["premise"], ex["hypothesis"]])

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(
            "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_train_set.txt"
        )
        valid_path = dl_manager.download_and_extract(
            "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
        )

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": valid_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate hans examples.
        Args:
          filepath: a string
        Yields:
          dictionaries containing "premise", "hypothesis" and "label" strings
        """
        for idx, line in enumerate(open(filepath, "rb")):
            if idx == 0:
                continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Examples not marked with a three out of five consensus are marked with
            # "-" and should not be used in standard evaluations.
            if split_line[0] == "-":
                continue
            # Works for both splits even though dev has some extra human labels.
            yield idx, {
                "premise": split_line[5],
                "hypothesis": split_line[6],
                "label": split_line[0],
                "binary_parse_premise": split_line[1],
                "binary_parse_hypothesis": split_line[2],
                "parse_premise": split_line[3],
                "parse_hypothesis": split_line[4],
                "heuristic": split_line[8],
                "subcase": split_line[9],
                "template": split_line[10],
            }