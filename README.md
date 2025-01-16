 #  The Jungle of Generative Drug Discovery :deciduous_tree: :pill:  :monkey: :dna: :snake: :bar_chart: :palm_tree: :chart_with_upwards_trend: :elephant: 

This repository contains the code and data to reproduce the results of the paper "*The Jungle of Generative Drug Discovery: Traps, Treasures, and Ways Out*". Per the nature of our work, this is **NOT** a repository that presents a new python library or a new model. Instead, it fulfills the reproducibility requirement of a scientific work.

To learn more about our work, you can read the [preprint](https://arxiv.org/abs/2501.05457). Please consider citing us if our findings contribute to your research:

```
@article{ozccelik2024jungle,
  title={The Jungle of Generative Drug Discovery: Traps, Treasures, and Ways Out},
  author={{\"O}z{\c{c}}elik, R{\i}za and Grisoni, Francesca},
  journal={arXiv preprint arXiv:2501.05457},
  year={2024}
}
```

In short, we dig into de novo design evaluation problem and evaluate around 1B designs, generated for bioactive molecule task. We used three protein targets (DRD3, PIN1, VDR), five different splits per target, three architectures (GPT, LSTM, S4), and three sampling strategies (temperature, topk, topp) with 22 different parameters in total. 

## :gear: Installation 

We recommend using a virtual environment to install the dependencies. You can create a new conda environment with the following command:

```bash
conda create -n jungle python=3.8
conda activate jungle
```

Then, you can install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The created environment is sufficient to reproduce all experiments and results.

## :file_folder: Folder Structure

### :pill: Designs 
Results for an experiment can be found under `designs/{protein_target}/setup-{setup_idx}/{architecture}/{sampling_strategy}/t={temperature}-k={topk}-p={topp}/`, *e.g.,* `designs/DRD3/setup-0/gpt/temperature/t=1.0-k=33-p=1.0`.

Each such folder contains the following files:

- `designs.txt`: A list of SMILES generated in the experiment.
- `lls.txt`: log-likelihood of the designs (in the same order).
- `valid_unique_novel_designs.smiles`: A list of valid, unique, and novel designs, in canonical SMILES. No order is preserved between `designs.txt` and this file.
- `syntactic_scores.json`: A dictionary of syntactic scores (validity, uniqueness, and novelty) measured in increasing number of designs.
- `n_unique_substructures.json`: A dictionary of number of unique substructures measured in increasing number of designs. See the paper for more details about this metric.
- `descriptors/tanimoto_to_train.txt`: Minimum distance of each novel design to the fine-tuning set. The order is preserved with `valid_unique_novel_designs.smiles`.

### :bar_chart: Datasets 

Pre-training and fine-tuning datasets are available under `data` folder. Fine-tuning datasets contain five splits each. Descriptors, scaffolds, and distances are also available under `data` folder, as computed in our work.

### :computer: Models 
All fine-tuned models can be found under `models` folder. The file paths are constructed as `models/{protein_target}/setup-{setup_idx}/{architecture}/`, *e.g.,* `models/DRD3/setup-0/gpt/`.

Each such folder contains three files:

- `last-epoch/init_arguments.json`: Hyperparameters used to initialize the model in the codebase.
- `last-epoch/model.pt`: The model weights.
- `history.json`: Training history of the model.

These folders can be used to load the models and generate designs. See `runners/design.py` for an example.

### :running: Runners 
All experiments and evaluations can be restarted using the code under `runners` folder. Each script is expecting a set of terminal arguments to run. These arguments can be seen at the top of each file. The allowed/expected values per argument are available under `runners/setup.py`. Example commands are available under `scripts/` folder.


## :mailbox: Contact 
If you have any questions or need further information, please feel free to contact us via issues or email!