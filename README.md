## CLIP-Mesh

Official implementation of [**CLIP-Mesh: Generating textured meshes from text using pretrained image-text models**](https://www.nasir.lol/clipmesh)<br/>

[Nasir Mohammad Khalid](https://www.nasir.lol/),
[Tianhao Xie](https://www.linkedin.com/in/tianhao-xie-440b20186/),
[Eugene Belilovsky](http://eugenium.github.io/),
[Tiberiu Popa](https://users.encs.concordia.ca/~stpopa/index.html)<br/>
_[SIGGRAPH ASIA 2022]() | [arXiv](https://arxiv.org/abs/2203.13333) | [Project page](https://www.nasir.lol/clipmesh)_

![CLIP-Mesh header image](./assets/header.jpeg)

## Quickstart

(Work in Progress!) - The following Google Collab notebooks are the best way to get started:

|<img src="./assets/single.gif" width="310"/>|<img src="./assets/multi.gif" width="310"/>|
|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------:|
| [Text to Mesh](https://colab.research.google.com/drive/15Fm4EhLlB20EugLUnTdhSJElvGVCU7Ys?usp=sharing)| [Multi Mesh Generation](https://colab.research.google.com/drive/12MbQTjiQ6fyCo4x6Gtp119LFksE8PGkW?usp=sharing) |

|<img src="./assets/cow.gif" width="310"/>|<img src="./assets/smpl.gif" width="310"/>|
|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------:|
| [Stylizing a Mesh](https://colab.research.google.com/drive/1df5yzS2vmqyYko016tVYXYZKt2Hrmy7Q?usp=sharing)| [Apply CLIP-Mesh to Human Models](https://github.com/NasirKhalid24/CLIP-Mesh-SMPLX)|

## Setup

Clone this repository recursively to get all submodules - use submodule update to get downstream submodules

```
git clone --recurse-submodules https://github.com/chacorp/CLIP-Mesh-test.git
cd CLIP-Mesh-test
git submodule update --init --recursive
```

Setup Conda environment and install packages

```
conda create -n clip-mesh python=3.7
conda activate clip-mesh
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
# python -c "import torch; print(torch.cuda.is_available())"
# python -c "import torch; print(torch.cuda.get_arch_list())"
pip install -r requirements.txt
```

Install loop subdivison code and DALLE-2 (not that DALLE-2 is from an earlier commit so existing install may not work)

```
<!-- Install Loop Subdivision -->
cd loop_limitation
pip install .
cd ..

<!-- Install DALLE2 - Diffusion Prior -->
cd DALLE2-pytorch
pip install .
cd ..

<!-- Get DALLE2 Prior Weights -->
mkdir weights
wget https://huggingface.co/spaces/NasirKhalid24/Dalle2-Diffusion-Prior/resolve/main/larger-model.pth -O ./weights/model.pth
```

## Usage

This repo comes with some configs that are passed to ```main.py``` using the ```--config``` flag

Any of the config paramaters can be overriden by passing them to as arguments to the ```main.py``` file so you can have a base .yml file with all your parameters and just update the text prompt to generate something new

An example would be using the given config file for single mesh generation ```single.yml```

```
# Use all parameters in file
python main.py --config configs/single.yml      

# Use all parameters in file but change text prompt
python main.py --config configs/single.yml  --text_prompt "a hamburger"    

# Use all parameters in file but change text prompt, batch, texture resolution
python main.py \
--config configs/single.yml \
--text_prompt "a hamburger" \
--batch_size 5 \
--texture_resolution 1024
```

## Tips, Tricks, FAQs etc

I recommend checking out the [following document](./assets/FAQ.md) as it could answer any doubts that come up (will be updated regularly) - if you still have questions reach out [@Nymarius_](https://twitter.com/Nymarius_) or open an issue
