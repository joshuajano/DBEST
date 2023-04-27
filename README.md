# On ModulAting TExt Scene (MATES)
![teaser](assets/teaser.png)

## News
**2023.02.05** Pre-Release code :partying_face: :partying_face:

## TO-DO Lists
- [ ] Upload pre-trained weight 

## Demo
go to `inner-loop/` directory and run
```
python prototype.py
```

## Run DBEST

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

#### Step 2: Inner Loop Finetuning
For text recognition model, please use the origin weight from [ABINet](https://github.com/FangShancheng/ABINet). 
By using the pre-traind from *outer loop* process, go to `inner-loop/` directory and run
```
python prototype.py
```
