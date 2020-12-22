# CRNN_v1

* The data is from https://www.music-ir.org/mirex/wiki/2017:Structural_Segmentation a corpus of 177 Beatles songs  
<img src="https://github.com/sungbohsun/CRNN_v1/blob/main/png/labels.png" width="600" />  

* You can also get the data from https://github.com/sungbohsun/music_segmentation **audio** & **annotations/Labels**  the data triansform to the seven diffirent label

  * 1、引子 （intro）
  * 2、主歌 （verse）
  * 3、連接段 （Bridge）
  * 4、結語 （outro）
  * 5、休息 (break)
  * 6、副歌  (Refrain)
  * 7、沉默  (silece)

<img src="https://github.com/sungbohsun/CRNN_v1/blob/main/png/model.png" width="600" />  

```bash
├── audio
├── Labels
├── model
├── README.md
├── train.py
└── utils.py
```
