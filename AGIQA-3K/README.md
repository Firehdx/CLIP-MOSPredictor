
# AGIQA-3K Database

The AGIQA-3K is a fine-grained AI-Generated Image (AGI) subjective quality assessment database. Please refer to our paper (AGIQA-3K: An Open Database for AI-Generated Image Quality Assessment) [here](https://arxiv.org/abs/2306.04717) for more details.

## 1. Introduction

With the rapid advancements of the text-to-image generative model, AI-generated images (AGIs) have been widely applied to entertainment, education, social media, etc. However, considering the large quality variance among different AGIs, there is an urgent need for quality models that are consistent with human subjective ratings. 

To address this issue, we extensively consider various popular AGI models, generated AGI through different prompts and model parameters, and collected subjective scores at the perceptual quality and text-to-image alignment level, thus building the most comprehensive AGI subjective quality database AGIQA-3K so far. 

We believe that the fine-grained subjective scores in AGIQA-3K will inspire subsequent AGI quality models to fit human subjective perception mechanisms at both perception and alignment levels and to optimize the generation result of future AGI models.

## 2. Database Description

The **AGIQA-3K** database contains 2 types of files:

A. AGIQA-3K.zip (AI-Generated Images)

The quark downloadlink can be accessed [here](https://pan.quark.cn/s/10187e65d5c1).

The googledrive downloadlink is [here](https://drive.google.com/file/d/1ObuOZ6YZqZuxe4oRlaf3kdOBlTRg2GE4/view?usp=sharing).

B. data.csv (Prompt, perception, alignemnt)

Column1: Image name

Column2: Input prompt for the generative model

Column3-4: The two adj in the prompt. (10 adj in total)

Column5: The style in the prompt. (5 style in total)

Column6-7: Normalized MOS and STD for perception subjective score.

Column8-9: Normalized MOS and STD for alignment subjective score.


## 3. Citation

If you find our work useful, please cite our paper as:
```
@ARTICLE{10262331,
  author={Li, Chunyi and Zhang, Zicheng and Wu, Haoning and Sun, Wei and Min, Xiongkuo and Liu, Xiaohong and Zhai, Guangtao and Lin, Weisi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={AGIQA-3K: An Open Database for AI-Generated Image Quality Assessment}, 
  year={2023},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3319020}}
```

## 4. License

The database is distributed under the MIT license.
```
## Contact
Chunyi Li, lcysyzxdxc@sjtu.edu.cn
