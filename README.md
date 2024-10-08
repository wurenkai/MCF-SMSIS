<p align="center">
  <h1 align="center">MCF-SMSIS:Multi-tasking with complementary
functions for stereo matching and surgical
instrument segmentation</h1>
  <p align="center">
    Renkai Wu, Changyu He, Pengchen Liang, Yinghao Liu, Yiqi Huang, Weiping Liu, 
    Biao Shu, Panlong Xu, Qing Chang*    
  </p>
    <p align="center">
      1. Shanghai University, Shanghai, China</br>
      2. Ruijin Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai, China</br>
      3. Imperial College London, London, United Kingdom</br>
      4. Shanghai Microport Medbot (Group) Co.,Ltd., Shanghai, China</br>
  </p>
</p>

https://github.com/wurenkai/MCF-SMSIS/assets/124028634/a8c9ffdb-0373-462e-b095-3fa04e139bff


# How to run the code

## Environment
* NVIDIA GeForce RTX 4080 Laptop GPU (12GB)
* Python 3.8
* Pytorch 1.12

## Install

### Create a virtual environment and activate it.

```
conda create -n MCF python=3.8
conda activate MCF
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm==0.5.4
```

## Data Preparation
The SCARED dataset should be obtained from the [official website](https://endovissub2019-scared.grand-challenge.org) under a confidentiality agreement. The dataset also needs to be corrected and the correction toolkit is available at [scared-toolkit](https://github.com/dimitrisPs/scared_toolkit).

## Model weights for each stage
* S1-d1 1.[Google drive](https://drive.google.com/drive/folders/1B6wKN1_tN73lIU8A7fqttS1o_dYuuHjV?usp=sharing) 2.[Baidu drive](https://pan.baidu.com/s/10pc3kzAjKox0-X3tAr09AQ) Link Password:89ja
* S2-d1d2 1.[Google drive](https://drive.google.com/drive/folders/15zBJQlJAY9fUVwt30jIhhd3C3zKaEk3r?usp=sharing) 2.[Baidu drive](https://pan.baidu.com/s/1tN-gPFF5vpsvA2nDUxkIiQ) Link Password:nv68
* S2-d1s 1.[Google drive](https://drive.google.com/drive/folders/1Sp5056xOPavbX5PVq34qGqRqjU-gKls8?usp=sharing) 2.[Baidu drive](https://pan.baidu.com/s/1hsVqstrlpyKTrm7z9Z4AhQ) Link Password:ttr4
* S3-d1d2s 1.[Google drive](https://drive.google.com/drive/folders/1oPuwWfIxo7OuY_UtyATKb6K2vqvbOL5z?usp=sharing) 2.[Baidu drive](https://pan.baidu.com/s/14vDAdKWgtQ0yjMUQm7LwhQ) Link Password:9tn7
* S3-d1sd2 1.[Google drive](https://drive.google.com/drive/folders/1PbDCOqdT6aeYEJAxG_C3GXNXAhr1FobR?usp=sharing) 2.[Baidu drive](https://pan.baidu.com/s/1oHL0VyzA-Q5QwketT_BH6Q) Link Password:z4b4
* S1-s 1.[Google drive](https://drive.google.com/drive/folders/1hv40OOrS5A1585spTk9FswUvguYpLx3F?usp=sharing) 2.[Baidu drive](https://pan.baidu.com/s/1MHDzuRN_aBYXBx8BOpBQKA) Link Password:nwuj
* S2-sd1 1.[Google drive](https://drive.google.com/drive/folders/1RRWrirmgoC71DX-JCXfjGv2yftXUCfFQ?usp=sharing) 2.[Baidu drive](https://pan.baidu.com/s/1uf9lwODpyFqHwVjZSBI8OQ) Link Password:vbrg
* S3-sd1d2 1.[Google drive](https://drive.google.com/drive/folders/1MLZ1u2s1gljPPEwfI4u_Qo8wMLoNL9NV?usp=sharing) 2.[Baidu drive](https://pan.baidu.com/s/1n56244Umq3MVemcgQgBYQg) Link Password:vtlp


## Validation and Testing
* You can proceed to run the test_seg_depth.py and test_seg.py files to verify the parallax and segmentation performance of the model, respectively.
* The test_only_depth.py file is needed to verify the model performance without combining the features in the segmentation and decoding part.
* To output the parallax result plot, you can run the test_save_disp.py file and save the result to the test folder.
* To output the segmentation results, run test_save_seg.py and save the results to the results folder.

## Citation
If you find this repository helpful, please consider citing:
```
@article{wu2024mcf,
  title={MCF-SMSIS: Multi-tasking with complementary functions for stereo matching and surgical instrument segmentation},
  author={Wu, Renkai and He, Changyu and Liang, Pengchen and Liu, Yinghao and Huang, Yiqi and Liu, Weiping and Shu, Biao and Xu, Panlong and Chang, Qing},
  journal={Computers in Biology and Medicine},
  volume={179},
  pages={108923},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgements
Thank you for the help provided by these outstanding efforts.
* [MSDESIS:](https://github.com/dimitrisPs/msdesis)
* [CGI-Stereo:](https://github.com/gangweiX/CGI-Stereo)
