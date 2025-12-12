# Retinex-MEF
Codes for ***Retinex-MEF: Retinex-based Glare Effects Aware
Unsupervised Multi-Exposure Image Fusion (ICCV 2025)***

[Haowen Bai](https://haowenbai.github.io/), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang), [Zixiang Zhao](https://zhaozixiang1228.github.io/), [Lilun Deng](), [Yukun Cui](), [Shuang Xu](https://shuangxu96.github.io/).

-[*[Paper]*]()

-[*[ArXiv]*](https://arxiv.org/pdf/2503.07235)  


## Abstract

Multi-exposure image fusion (MEF) synthesizes multiple, differently exposed images of the same scene into a single, well-exposed composite. Retinex theory, which separates image illumination from scene reflectance, provides a natural framework to ensure consistent scene representation and effective information fusion across varied exposure levels. However, the conventional pixel-wise multiplication of illumination and reflectance inadequately models the glare effect induced by overexposure. To address this limitation, we introduce an unsupervised and controllable method termed Retinex-MEF. Specifically, our method decomposes multi-exposure images into separate illumination components with a shared reflectance component, and effectively models the glare induced by overexposure. The shared reflectance is learned via a bidirectional loss, which enables our approach to effectively mitigate the glare effect. Furthermore, we introduce a controllable exposure fusion criterion, enabling global exposure adjustments while preserving contrast, thus overcoming the constraints of a fixed exposure level. Extensive experiments on diverse datasets, including underexposure-overexposure fusion, exposure controlled fusion, and homogeneous extreme exposure fusion, demonstrate the effective decomposition and flexible fusion capability of our model


## Update
- [2025/9] Release the code.

## Citation

```
@inproceedings{bai2024retinex,
  title={Retinex-MEF: Retinex-based Glare Effects Aware Unsupervised Multi-Exposure Image Fusion},
  author={Bai, Haowen and Zhang, Jiangshe and Zhao, Zixiang and Deng, Lilun and Cui, Yukun and Xu, Shuang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```


## 🌐 Usage

### 🏊 Training
**1. Data Preparation**

Modify 'raw_path' in ``'./data/SICE_arrange.py'`` to the path for the raw SICE training set.

run

```
python ./data/SICE_arrange.py
```

to sort the data by exposure from low to high.

run

```
python ./data/SICE_processing.py
```

to extract patches from the sorted data. The cropped and processed data will be stored in ``'./data/SICE_training'``. Where ``img_1`` is the underexposed image, ``img_2`` is the overexposed image, ``img_3`` is other images from the sequence with different exposure levels.

[*Here*](https://drive.google.com/file/d/1MSLVbKzOtwNnkPSI4PZZ5qZX-4j-o_MZ/view?usp=drive_link) is a sample dataset containing 10 patches. (Note: In this example, img_3 is the same as img_1.)


**2. Commence Training**

run
```
python train.py
``` 
The models will be saved in the ``./exp`` directory based on the timestamp.


### 🏄 Testing

**1. Pretrained models**

Pretrained models are available in ``./model/`` 

**2. Test cases**

The 'test_cases' folder contains four examples that appear in the main paper.
Running 
```
python test.py
``` 
will fuse these cases, and the fusion results will be saved in the 'test_results' folder.

**3. Test customization**

Modify the variables in ``test.py`` as needed: 

'path_model' (the path of the pretrained model), 

'path_img1' (the path of the under-exposure images), 

'path_img2' (the path of the over-exposure images), 

'path_result' (the path to save the fusion result). 

Then run
```
python test.py
``` 
and the fusion results will be saved in the 'path_result'.
