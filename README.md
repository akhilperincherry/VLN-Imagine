# Do Visual Imaginations Improve Vision-and-Language Navigation Agents?

Official implementation of the **CVPR 2025** paper: **Do Visual Imaginations Improve Vision-and-Language Navigation Agents?**

[Akhil Perincherry](http://akhilperincherry.com/), [Jacob Krantz](https://jacobkrantz.github.io/) and [Stefan Lee](http://web.engr.oregonstate.edu/~leestef/)

[[Project Page](https://www.akhilperincherry.com/VLN-Imagine-website/)] [[Paper](https://arxiv.org/pdf/2503.16394)]

Vision-and-Language Navigation (VLN) agents are tasked with navigating an unseen environment using natural language instructions. In this work, we study if visual representations of sub-goals implied by the instructions can serve as navigational cues and lead to increased navigation performance. To synthesize these visual representations or “imaginations”, we leverage a text-to-image diffusion model on landmark references contained in segmented instructions. These imaginations are provided to VLN agents as an added modality to act as landmark cues and an auxiliary loss is added to explicitly encourage relating these with their corresponding referring expressions. Our findings reveal an increase in success rate (SR) of ∼1 point and up to ∼0.5 points in success scaled by inverse path length (SPL) across agents. These results suggest that the proposed approach reinforces visual understanding compared to relying on language instructions alone.

<p align="center">
  <img src="teaser.png" alt="teaser" width="600"/>
</p>

## Installation

1. Install Matterport3D simulators: follow instructions from [here](https://github.com/peteanderson80/Matterport3DSimulator) to install the latest version.
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Setup [VLN-DUET](https://github.com/cshizhe/VLN-DUET) and [VLN-HAMT](https://github.com/cshizhe/VLN-HAMT) using their official instructions.

3. Install requirements:
```setup
conda create --name vln-imagine python=3.8.5
conda activate vln-imagine
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# install timm
git clone https://github.com/rwightman/pytorch-image-models.git
cd pytorch-image-models
git checkout 9cc7dda6e5fcbbc7ac5ba5d2d44050d2a8e3e38d
```

4. Download checkpoints and features from [here](https://oregonstate.box.com/s/97n3i25m45wkrr1ivt3stah2x9cqabv1). Files include:
 - off-the-shelf ViT features for R2R-Imagine.
 - HAMT ViT features for R2R-Imagine.
 - HAMT-Imagine R2R checkpoint.
 - DUET-Imagine R2R checkpoint.

5. (optional) Download imagination generations for R2R from [here-TODO](https://github.com/akhilperincherry/VLN-Imagine) and metadata of generations and noun-phrase segments of R2R instructions from [here](https://oregonstate.box.com/s/97n3i25m45wkrr1ivt3stah2x9cqabv1).

6. Run - adjust paths of downloaded files and run following for training/inference for HAMT and DUET from the respective folders:
```
cd <folder-of-HAMT/DUET src>
bash scripts/run_r2r.sh
```


## License

Our code is [MIT licensed](LICENSE). Trained models are considered data derived from the Matterport3D scene dataset and are distributed according to the [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).

## Citing

```tex
@InProceedings{aperinch_2025_VLN_Imagine,
          title={Do Visual Imaginations Improve Vision-and-Language Navigation Agents?},
          author={Akhil Perincherry and Jacob Krantz and Stefan Lee},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
          month={June},
          year={2025},
}
```
