# ESTR-CoT

**ESTR-CoT: Towards Explainable and Accurate Event Stream based Scene Text Recognition with Chain-of-Thought Reasoning**, 
Xiao Wang, Jingtao Jiang, Qiang Chen, Lan Chen*, Lin Zhu, Yaowei Wang, Yonghong Tian, Jin Tang
arXiv Pre-Print 
[[arXiv](https://arxiv.org/abs/2507.02200)] 
[[Code](https://github.com/Event-AHU/ESTR-CoT/)]





### Abstract 
Event stream based scene text recognition is a newly arising research topic in recent years which performs better than the widely used RGB cameras in extremely challenging scenarios, especially the low illumination, fast motion. Existing works either adopt end-to-end encoder-decoder framework or large language models for enhanced recognition, however, they are still limited by the challenges of insufficient interpretability and weak contextual logical reasoning. In this work, we propose a novel chain-of-thought reasoning based event stream scene text recognition framework, termed ESTR-CoT. Specifically, we first adopt the vision encoder EVA-CLIP (ViT-G/14) to transform the input event stream into tokens and utilize a Llama tokenizer to encode the given generation prompt. A Q-former is used to align the vision token to the pre-trained large language model Vicuna-7B and output both the answer and chain-of-thought (CoT) reasoning process simultaneously. Our framework can be optimized using supervised fine-tuning in an end-to-end manner. In addition, we also propose a large-scale CoT dataset to train our framework via a three stage processing (i.e., generation, polish, and expert verification). This dataset provides a solid data foundation for the development of subsequent reasoning-based large models. Extensive experiments on three event stream STR benchmark datasets (i.e., EventSTR, WordArt*, IC15*) fully validated the effectiveness and interpretability of our proposed framework. The source code and pre-trained models will be released on https://github.com/Event-AHU/ESTR-CoT
<div align="center">
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/CoT_STR_2model.png" width="800">
</div>



### ▶️ Demo Video

 
<div align="center">
  <video src="https://github.com/user-attachments/assets/5b0aef1a-704b-4708-94d3-f7a25acb5854" width="100%" poster=""> </video>
</div>


### :dvd:  Dataset Download 

<div align="center">
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/Data_Generation.png" width="800">
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/CoT_dataset.png" width="800">
</div>

* **Download from Baidu Drive:**
```
URL：https://pan.baidu.com/s/1XN8MfK1cKrqaSOo3e2oD3A?pwd=2l7c     Code：2l7c
```

* **Download from DropBox:** 
```
https://www.dropbox.com/scl/fo/s31llbv7bshz2xj4mf2gm/AFP1AGDcSoY0mk-fcyfL7jw?rlkey=p25w7366lzex7qe3pdgz96ec4&st=afcymd0x&dl=0
```

### :hammer: Environment Configuration 
1. Creating conda environment
```
conda create -n estr python=3.9
conda activate etsr
```

2. Build from source
```
git clone https://github.com/Event-AHU/ESTR-CoT
cd ESTR-CoT
pip install -e .
```
📌 *For the full list of dependencies, see [`environment_version.txt`](./environment_version.txt)*

### :hammer: Prepare Weight 
Our Vicuna version model is released at [here](https://huggingface.co/mlpc-lab/BLIVA_Vicuna). Download our model weight and specify the path in the model config [here](https://github.com/Event-AHU/EventSTR/blob/384d37bececfc166d32d40c6fcd0ce64e1e16573/bliva/configs/models/bliva_vicuna7b.yaml#L8C4-L8C53) at line 8.

The LLM we used is the v0.1 version from Vicuna-7B. To prepare Vicuna's weight, please refer to our instruction [here](https://github.com/mlpc-ucsd/BLIVA/blob/main/PrepareVicuna.md). Then, set the path to the vicuna weight in the model config file [here](https://github.com/Event-AHU/EventSTR/blob/384d37bececfc166d32d40c6fcd0ce64e1e16573/bliva/configs/models/bliva_vicuna7b.yaml#L21) at Line 21.



### :hammer: Training & Testing 
**Training**
```
bash train.sh
```

**Testing**
```
python test.py
```
### Results 
<div align="center">
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/text_recognition.jpg" width="800">
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/Visualization.jpg" width="800">
</div>


### Acknowledgement 
We sincerely thank the contributors of the following open-source projects and datasets that made this work possible:

- [**BLIVA**](https://github.com/mlpc-ucsd/BLIVA): for providing the base multimodal LLM architecture.  
- [**WordArt**](https://github.com/xdxie/WordArt): for the WordArt dataset used in training and evaluation.  
- [**IC15**](https://github.com/MhLiao/DB): for the ICDAR2015 dataset used in our benchmark.

### Citation 
```

```





