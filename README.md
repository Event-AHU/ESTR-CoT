# ESTR-CoT
<div align="center">
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/Data_Generation.png" width="600">
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/CoT_STR_2model.png" width="600">

</div>

### ▶️ Demo Video

 
<div align="center">
  <video src="https://github.com/user-attachments/assets/5b0aef1a-704b-4708-94d3-f7a25acb5854" width="100%" poster=""> </video>
</div>


### :dvd:  Dataset Download 
<div align="center">
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/CoT_dataset.png" width="600">
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
conda create -n bliva python=3.9
conda activate bliva
```

2. Build from source
```
git clone https://github.com/Event-AHU/ESTR-CoT
cd ESTR-CoT
pip install -e .
```

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
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/text_recognition.jpg" width="600">
<img src="https://github.com/Event-AHU/ESTR-CoT/blob/main/images/Visualization.jpg" width="600">
</div>
