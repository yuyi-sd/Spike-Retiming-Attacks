# Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks (ICLR 2026)

This repo is the official implementation of "[Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](https://openreview.net/forum?id=b107VY19Id&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))" (ICLR 2026)

## Requirements
````
python version: 3.11.9
CUDA version: 12.4
````

````
numpy==2.1.3
PyYAML==6.0.1
spikingjelly==0.0.0.0.14
timm==1.0.11
torch==2.3.0
torchattacks==3.5.1
torchvision==0.18.0
````

## Prepare

````
conda create --name snn_attack python=3.11.9
conda activate snn_attack
pip install -r requirements.txt
````

## Checkpoints & Datasets
Some pretrained checkpoints are in [checkpoints](./checkpoints).
Please download the corresponding datasets and put it into "../datasets/".

## Run
We provide some scripts in [some_scripts.sh](./some_scripts.sh).

## Acknowledgments
The frame of this code is altered from [PDSG-SDA](https://github.com/ryime/PDSG-SDA). We thank the authors for their contribution.

## Citation
If you find this paper useful, please consider staring this repository and citing our paper:
````
@inproceedings{yu2026spikeretiming,
  title={Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks},
  author={Yu, Yi and Zhang, Qixin and Ye, Shuhan and Lin, Xun and Wei, Qianshan and Wang, Kun and Yang, Wenhan and Tao, Dacheng and Jiang, Xudong},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
````