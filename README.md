# DeepFuse 
This repository contains an unofficial Pytorch implementation of the DeepFuse network for image fusion of extreme exposure image pairs, published in ICCV 2017.

## Prerequisits
This release of DeepFuse implementation was tested in Google Colab with
- Python = 3.8
- PyTorch = 1.10.0
- torchvision = 0.11.1

## Dataset
The dataset used for this experiment is the subset of the SICE dataset given here: [https://github.com/ytZhang99/CF-Net](https://github.com/ytZhang99/CF-Net)

The SICE dataset can be found here: [https://github.com/csjcai/SICE](https://github.com/csjcai/SICE)

## Training
### Folder structure

Place ground truth, over exposed, and under exposed images for training in the following folder structure.

```
  - SICE_subset
      - test_data
          - GT
          - OE
          - UE
      - train_data
          - GT
          - OE
          - UE
      - val_data
          - GT
          - OE
          - UE
```

### Training
```
    python Main.py --train True --use_cuda True --trainset "./SICE_subset/train_data/"
```

## Testing
Set the `use_cuda` flag to `False` if necessary.
```
    python Main.py --train False --use_cuda True --testset "./SICE_subet/test_data/"
```
## References
[1] [K. Ram Prabhakar, V Sai Srikar, R. Venkatesh Babu. DeepFuse: A Deep Unsupervised Approach for Exposure Fusion with Extreme Exposure Image Pairs, ICCV2017, pp. 4714-4722](https://openaccess.thecvf.com/content_iccv_2017/html/Prabhakar_DeepFuse_A_Deep_ICCV_2017_paper.html)
