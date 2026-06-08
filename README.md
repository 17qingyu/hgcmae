# HGCMAE: Heterogeneous Graph Contrastive Masked Autoencoder

This repository implements **HGCMAE**.


## 🧰 Environment

- Python 3.10.16
- PyTorch 2.6.0+cu124
- torch-geometric 2.6.1
- torch-scatter 2.1.2+pt26cu124
- torch-sparse 0.6.18+pt26cu124
- torch-cluster 1.6.3+pt26cu124
- torch-spline-conv 1.2.2+pt26cu124
- DGL 1.0.4 (CUDA 11.3 build + pip version available)
- NumPy 2.2.4
- SciPy 1.12.0
- scikit-learn 1.5.0
- Matplotlib 3.8.3
- tqdm 4.66.2
- PyTorch Lightning 2.5.1
- Hydra-core 1.3.2

---


## ⚙️ Running the Code

### 1. Pretraining

```bash
python pretrain.py --config config/pretrain/acm.yaml
python pretrain.py --config config/pretrain/dblp.yaml
python pretrain.py --config config/pretrain/aminer.yaml
python pretrain.py --config config/pretrain/freebase.yaml
```
### 2. Node Classification
Node classification requires loading a pretrained encoder checkpoint.

This is specified in the config file via:

```yaml
checkpoint_path: "logs/pretrain/<dataset>/version_xx/last_encoder.ckpt"
```

```bash
python classification.py --config config/classification/acm.yaml
python classification.py --config config/classification/dblp.yaml
python classification.py --config config/classification/aminer.yaml
python classification.py --config config/classification/freebase.yaml
```
