# SEG-EEND: SEGMENT-LEVEL END-TO-END NEURAL DIARIZATION WITH SPEAKER STATE CHANGE DETECTION

This repository extends the PyTorch implementation of End-to-End Neural Diarization (EEND) originally developed by [BUT Speech@FIT](https://github.com/BUTSpeechFIT/EEND).  
**SEG_EEND** introduces a *segment-level* approach that incorporates a **Speaker State Change Detector (SSCD)** to enhance diarization accuracy, especially under multi-speaker and overlapping speech conditions.

---

## 🔍 Overview

SEG_EEND integrates a **Speaker State Change Detector** into the standard EEND framework, enabling the model to explicitly learn transitions in speaker activity.  
This helps improve temporal consistency and boundary detection, leading to better DER and speaker change localization.

---

## 🧩 Directory Structure

```
SEG_EEND/
├─ eend/ # Core source code (model, trainer, data loader)
├─ examples/ # Example configuration files for training/inference
├─ .gitignore
├─ README.md
├─ requirements.txt

```

## ⚙️ Installation

```bash
git clone https://github.com/SEG-EEND/SEG_EEND.git
cd SEG_EEND
conda create -n seg_eend python=3.10
conda activate seg_eend
pip install -r requirements.txt
```

## 🚀 Training
To start distributed multi-GPU training:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDs> torchrun --nproc_per_node=<NUM_GPUs> \
    eend/train.py -c <path_to_config.yaml> --ddp
```

Example:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    eend/train.py -c examples/train_scd.yaml --ddp --noam-k 1
```

**Notes**
- `--ddp` enables DistributedDataParallel training.  
- `--noam-k` sets the learning rate scale factor.  
  (The `--noam-k` option was added to prevent learning rate scaling with the number of GPUs.  
  Setting it to 1 disables the default behavior where the learning rate is divided by the number of GPUs.)  
- Modify paths for training/validation data and `save_dir` inside the YAML config before training.  
- All GPUs listed in `CUDA_VISIBLE_DEVICES` will be used automatically.

## 🔁 Fine-tuning
To fine-tune from a pre-trained model, specify the checkpoint path either in the config file or via command-line argument:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    eend/train.py -c examples/adapt_scd.yaml --ddp --noam-k 1
```

## 🎧 Inference
To perform diarization inference:

```bash
python eend/infer.py -c examples/infer_scd.yaml
```

Define the input audio directory, model checkpoint, and output directory in the configuration file.

## 🧮 Evaluation
For evaluation metrics such as DER, JER, and speaker change detection (SCD):  
In this code, DER is a simplified metric. For more accurate measurement, it is recommended to use the official [dscore](https://github.com/nryant/dscore) tool.

## 📚 References

[1] C. Moon, M. H. Han, J. Park, N. S. Kim,
“SEG-EEND: SEGMENT-LEVEL END-TO-END NEURAL DIARIZATION WITH SPEAKER STATE CHANGE DETECTION,”
submitted to ICASSP 2026.

[2] S. Horiguchi, Y. Fujita, S. Watanabe, Y. Xue, and P. García,
“Encoder-decoder based attractors for end-to-end neural diarization,”
IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 1493–1507, 2022.

[3] Original EEND repository (BUT Speech@FIT):
https://github.com/BUTSpeechFIT/EEND


## License
This repository is released under the MIT License.