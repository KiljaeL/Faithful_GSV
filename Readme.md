# Faithful Group Shapley Value Estimation

This repository contains the implementation of the paper **"Faithful Group Shapley Value Estimation"**, submitted to NeurIPS 2025 main track.

## Requirements

- Python 3.10+
- Required libraries:
  - `numpy==2.2.4`
  - `scipy==1.15.2`
  - `pandas==2.2.3`
  - `matplotlib==3.10.1`
  - `seaborn==0.13.2`
  - `tqdm==4.67.1`
  - `scikit-learn==1.6.1`
  - `torch==2.6.0+cu124` *(required only for Section 4.2)*
  - `peft==0.14.0`

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

If you are using a CUDA-enabled GPU, you may install PyTorch with:

```bash
pip install torch==2.6.0+cu124 -f https://download.pytorch.org/whl/torch_stable.html
```

## Folder Structure

- `exp0_motivation/` – Motivational example to showcase shell company attack (Section 1)
- `exp1_benchmark/` – Benchmark comparison on SOU cooperative game (Section 4.1)
- `exp2_copyright/` – Faithful copyright attribution experiment using generative AI (Section 4.2)
- `exp3_xai/` – Faithful explainable AI experiment using the Diabetes dataset (Section 4.3)

## Reproducing the Experiments

### 1. Motivational Example (Section 1)

This experiment demonstrates the vulnerability of standard group Shapley value (GSV) to the shell company attack using a synthetic binary classification task.

To run the experiment and generate the corresponding figure:

```bash
cd exp0_motivation
python main.py
```

### 2. Benchmark Comparison (Section 4.1)

This experiment compares FGSV with state-of-the-art individual Shapley estimators in terms of approximation accuracy and computational efficiency on the Sum-of-Unanimity (SOU) cooperative game.

To run the experiment:

```bash
cd exp1_benchmark
python main.py
```

### 3. Faithful Copyright Attribution (Section 4.2)

This experiment evaluates the robustness of FSRS (Faithful Shapley Royalty Share) under shell company attacks in the context of generative AI.

To run the full pipeline for the experiment, go to the corresponding folder and run the shell script:

```bash
cd exp2_copyright
bash run.sh
```

(Note) This experiment requires a CUDA-compatible GPU with at least 32GB VRAM, and may take several hours to complete.

### 4. Faithful Explainable AI (Section 4.3)

This experiment evaluates the robustness of FGSV for explainable AI by quantifying the contributions of demographic groups to diabetes progression prediction.

To run the experiment:

```bash
cd exp3_xai
python main.py
```