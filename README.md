# WIC-WriterIDConfusion-MachineUnlearning
> Official Implementation for Pattern Recognition 2025 paper "Preserving Privacy Without Compromising Accuracy: Machine Unlearning for Handwritten Text Recognition"

[![Paper](https://img.shields.io/badge/Paper-PR2026-red)](https://www.sciencedirect.com/science/article/pii/S0031320325010726)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()

---

## 🚀 Overview

Modern handwritten text recognition (HTR) systems can inadvertently encode sensitive writer identity information.  
This repository introduces **WIC (Writer-ID Confusion)** — a machine unlearning framework that:

- Reduces writer identity leakage (handwriting style cues)
- Preserves transcription accuracy (retain set, and maintained on forget set)
- Enables GDPR-compliant model updates (user's right to erasure)

<img width="1498" height="629" alt="image" src="https://github.com/user-attachments/assets/91975c8a-de80-4997-922a-99bf46345879" />

---


## 📦 Repository Structure

```
WIC-WriterIDConfusion-MachineUnlearning/
│
├── dataset.py              # IAM dataset loader for full pretraining
├── dataset_unlearn.py      # IAM dataloader with retain/forget splits for unlearning
├── model.py                # Backbone handwriting recognition architecture
├── ours_WIC_main.py        # WIC loss implementation and training pipeline
├── prune_fine_grained.py   # Stage 1: fine-grained pruning procedure
├── train.py                # Backbone model training script
├── wids_trainset.py        # Backup file storing writer IDs for training set
├── RWTH_partition/         # Official IAM data partition used in this paper
└── README.md
```

---

## 📚 Dataset Preparation

We evaluate on standard handwritten text datasets (e.g., IAM, CVL).

1. Download dataset manually.
2. Organize the data correctly on your local machine, and remember to update the data directory path in `dataset.py` and `dataset_unlearn.py`.


---

## 📈 Results

<img width="1508" height="522" alt="image" src="https://github.com/user-attachments/assets/2add5d16-b423-47a9-8008-6f57d902de2c" />

<img width="1581" height="484" alt="image" src="https://github.com/user-attachments/assets/8c323c39-6459-4f0e-91c0-2ef407ab78e0" />


---

## 📌 Citation

If you find this work useful, please cite:

```bibtex
@article{kang2025preserving,
  title={Preserving privacy without compromising accuracy: Machine unlearning for handwritten text recognition},
  author={Kang, Lei and Fu, Xuanshuo and Gomez, Lluis and Forn{\'e}s, Alicia and Valveny, Ernest and Karatzas, Dimosthenis},
  journal={Pattern Recognition},
  pages={112411},
  year={2025},
  publisher={Elsevier}
}
```

---

## 📜 License

This project is released under the MIT License.

---

## 📬 Contact

For questions, please open an issue or contact us.

---
