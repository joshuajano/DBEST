# On ModulAting TExt Scene (MATES)
![teaser](assets/teaser.png)

## News

## Run MATES

#### Step 1: Generate Synthesized Text Scene Dataset 
You can download our SynText dataset on [this](https://drive.google.com/drive/folders/10DPeYjcpqO_Pxi3KC4kGitGW0Ytc3Eo1?usp=sharing).
Alternatively, you can generate by your own. Please refer to [srnet](https://github.com/Niwhskal/SRNet). We slightly change the code from its origin. Please refer to [generate-syntext](generate-syntext/) directory.

Please go to `generate-syntext/` directory and run 
```
python datagen.py
```

#### Step 2: Outer Loop Training
Before training the noise model, please initialize the weight from the pretrained [text2img](https://github.com/CompVis/latent-diffusion) from Latent Diffusion Model. Then, go to `outer-loop/` directory and run
```
python finetune.py
```
