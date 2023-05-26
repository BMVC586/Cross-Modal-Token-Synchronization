# Cross-Modal Token Synchronization: Enhancing Visual Speech Recognition via Quantized Audio Generation

### Installation

Download dependencies with the shell command below.
```shell
git clone https://github.com/BMVC586/Cross-Modal-Token-Synchronization
cd Cross-Modal-Token-Synchronization
git clone https://github.com/pytorch/audio
cd audio
git reset --hard e77b8f909154d0361afe9a3420a17fc41e74e9d6
python setup.py install
cd ..
apt install libturbojpeg
pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt -P ./LRW2
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
pip install -r requirements.txt
```

### Dataset Preparation

1. Get authentification for Lip Reading in the Wild Dataset via https://www.bbc.co.uk/rd/projects/lip-reading-datasets
2. Download dataset using the shell command below

```shell
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaa
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partab
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partac
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partad
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partae
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaf
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partag
```
3. Extract region of interest and convert mp4 file into pkl file with the commands below.
```shell
python ./src/preprocess_roi.py
python ./src/preprocess_pkl.py
```

### Train
For training with our methodology, please run the command below after preprocessing the dataset. You may change configurations in yaml files.
```shell
python ./src/train.py ./config/bert-12l-512d.yaml devices=[0] # Transformer backbone
python ./src/train.py ./config/dc-tcn-base.yaml devices=[0] # DC-TCN backbone
```

### Inference

For inference, please download the pretrained checkpoint from the repository's [release section](https://github.com/BMVC586/Cross-Modal-Token-Synchronization/releases)(or from url attached on the table below) and run the code with the following command.
```shell
python ./src/inference.py ./config/bert-12l-512d.yaml devices=[0] # Transformer backbone
python ./src/inference.py ./config/dc-tcn-base.yaml devices=[0] # DC-TCN backbone
```

### Abstract

We investigate the role of cross-modal token synchronization in visual speech recognition (VSR), focusing on the interplay between visual representations and temporally aligned quantized audio tokens. Our method generates discrete audio tokens from silent video frames in a non-autoregressive manner. Unlike previous approaches that heavily rely on raw audio waveforms and graphemes, our method utilizes quantized audio tokens, exploiting their advantages in terms of improved linguistic representation and efficient computation. We further introduce a joint training process, combining category classification, audio classification, and KL-Divergence losses to ensure consistency between original and horizontally flipped predictions. Our approach improves VSR performance by +2.8\%p over existing methods on the Lip Reading in the Wild benchmark by adding a single forward pass in inference and a negligible increase in learnable parameters. Importantly, our model, trained on one third of the dataset size, outperforms ensemble models trained on larger datasets.

| Method              | Spatial        | Temporal    | Supervised | Top-1 (%) | 
| ------------------- | -------------- | ----------- | :----------: | :---------: |
| Xu et al.  (2020)          | ResNet50       | BiLSTM      | ✓          | 84.8      |
| Martinez et al. (2020)     | ResNet18       | MS-TCN      | ✓          | 85.3      |
| Kim et al. (2021) | ResNet18       | BiGRU       | ✓          | 85.4      |
| LiRA    (2021)   | ResNet18       | Conformer   | ✗          | 88.1      |
| MVM        (2022)    | ResNet18       | MS-TCN      | ✗          | 88.5      |
| Koumparoulis et al. (2022) | EfficientNetV2 | Transformer | ✓          | 89.5      |
| Ma et al.  (2022)    | ResNet18       | DC-TCN      | ✓          | 90.4      |
| ---                 | ---            | ---         | ---        | ---       |
| Ours                | ResNet18       | DC-TCN      | ✓          | **91.2**  |
| Ours                | ResNet18       | Transformer | ✓          | **93.2**  |

Above is performance comparison on the Lip Reading in the Wild benchmark. The proposed Cross-Modal Token Synchronization method significantly outperforms existing state-of-the-art techniques. The term "Supervised" indicates whether the model was only trained with ground truth transcriptions or trained with self-supervised learning methods.


| Model       | Sync | Training Set     | Data Size | Ensemble | WB  | Top-1 (%)       | Checkpoints |
| :-----------: | :----: | :----------------: | :---------: | :--------: | :---: | :---------------: | :---: |
| Transformer | ✗    | LRW              | 165h      | ✗        | ✗   | 89.5            | - |
| DC-TCN      | ✗    | LRW              | 165h      | ✗        | ✗   | 90.4            | - |
| DC-TCN      | ✗    | LRW              | 165h      | ✗        | ✓   | 92.1            | - |
| DC-TCN      | ✗    | LRW              | 165h      | ✓        | ✓   | 93.4            | - |
| DC-TCN      | ✗    | LRW, LRS2&3, AVS | 1504h     | ✗        | ✗   | 91.1            | - |
| DC-TCN      | ✗    | LRW, LRS2&3, AVS | 1504h     | ✗        | ✓   | 92.9            | - |
| DC-TCN      | ✗    | LRW, LRS2&3, AVS | 1504h     | ✓        | ✓   | 94.1            | - |
| ---      | ---    | --- | ---     | ---        | ---   | ---            | --- |
| DC-TCN      | ✓    | LRW              | 165h      | ✗        | ✗   | **91.2 (+0.8)** | [🔗](https://github.com/BMVC586/Cross-Modal-Token-Synchronization/releases/download/v1/LRW-Checkpoints_tcn-epoch.74-step.95475-audioloss10.ckpt) |
| DC-TCN      | ✓    | LRW              | 165h      | ✗        | ✓   | **93.4 (+1.3)** | [🔗](https://github.com/BMVC586/Cross-Modal-Token-Synchronization/releases/download/v1/LRW-Checkpoints_dc-tcn-resnet18-base-audio10-fixmixup-WB-4GPU-BEST-epoch.75-step.96748.ckpt) |
| Transformer | ✓    | LRW              | 165h      | ✗        | ✗   | **93.2 (+3.7)** | [🔗](https://github.com/BMVC586/Cross-Modal-Token-Synchronization/releases/download/v1/LRW-Checkpoints_xtransformer-epoch.144-step.184585-0.9319.ckpt) |
| Transformer | ✓    | LRW              | 165h      | ✗        | ✓   | **94.9**        | [🔗](https://github.com/BMVC586/Cross-Modal-Token-Synchronization/releases/download/v1/LRW-Checkpoints_xtransformer-wb-epoch.148-step.189677-0.9493.ckpt)|
| Transformer | ✓    | LRW, LRS2        | 554h      | ✗        | ✓   | **95.0**        | [🔗](https://github.com/BMVC586/Cross-Modal-Token-Synchronization/releases/download/v1/LRW-Checkpoints_xtransformer-wb-epoch.146-step.187131-0.9497.ckpt) |

Above is performance on LRW test set according to applied methodologies. **WB** indicates word boundary introduced by which is word's temporal appearance indicator in the video. The term **Sync** denotes whether Cross-Modal Token Synchronization is applied during the training. **Ensemble** implies whether multiple models' outputs were ensembled for the prediction. The results without synchronization method are reported from previous works. Using the same spatial front-end with the previous works, we showcase that our approach benefits both DC-TCN and Transformer backbones even on low-resource settings.

