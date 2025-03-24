[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkeonlee9420%2FDailyTalk&count_bg=%23707EE1&title_bg=%23555555&icon=pytorch.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
# Intra- and Inter-modal Context Interaction Modeling for Conversational Speech Synthesis (III-CSS)




## Introduction

This is an implementation of the following paper. ["Intra- and Inter-modal Context Interaction Modeling for Conversational Speech Synthesis"](https://ieeexplore.ieee.org/document/10890216) (Accepted by ICASSP 2025)

[**Zhenqi Jia**](https://coder-jzq.github.io/), [Rui Liu](https://ttslr.github.io/people.html)

Corresponding Author: Rui Liu




## Demo Page

[Speech Demo](https://coder-jzq.github.io/ICASSP2025-IIICSS-Website/)




## Dataset

You can download the [dataset](https://drive.google.com/drive/folders/1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL) from DailyTalk.




## Pre-trained models

The Hugging Face URL of Sentence-BERT:  https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1

The Hugging Face URL of Wav2Vec2-IEMOCAP: https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP




## Preprocessing

Run

> python3 prepare_align.py --dataset DailyTalk

for some preparations.

For the forced alignment, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences. Pre-extracted alignments for the datasets are provided [here](https://drive.google.com/drive/folders/1fizpyOiQ1lG2UDaMlXnT3Ll4_j6Xwg7K?usp=sharing). You have to unzip the files in `preprocessed_data/DailyTalk/TextGrid/`. Alternately, you can [run the aligner by yourself](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/index.html). Please note that our pretrained models are not trained with supervised duration modeling (they are trained with `learn_alignment: True`).

After that, run the preprocessing script by

> python3 preprocess.py --dataset DailyTalk




## Training

Train III-CSS with

> python3 train.py --dataset DailyTalk




## Inference

Only the batch inference is supported as the generation of a turn may need contextual history of the conversation. Try

> python3 synthesize.py --source preprocessed_data/DailyTalk/test_*.txt --restore_step RESTORE_STEP --mode batch --dataset DailyTalk

to synthesize all utterances in `preprocessed_data/DailyTalk/test_*.txt`.




## Citation

If you would like to use our dataset and code or refer to our paper, please cite as follows.
```bash
@INPROCEEDINGS{10890216,
  author={Jia, Zhenqi and Liu, Rui},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Intra- and Inter-modal Context Interaction Modeling for Conversational Speech Synthesis}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Codes;Speech coding;Signal processing;Acoustics;Speech synthesis;History;Context modeling;Conversational Speech Synthesis;Contrastive Learning;Conversational Prosody;Intra-modal Interaction;Inter-modal Interaction},
  doi={10.1109/ICASSP49660.2025.10890216}}

```




## Contact the Author

E-mail：jiazhenqi7@163.com

Homepage: https://coder-jzq.github.io/

S2LAB Homepage: https://ttslr.github.io/

