# yolov7-autolabel

基于知乎用户**滴水穿石​**的自动标注工具AutoLableImg改进的yolov7+cpu版本，不需要启动GPU就可以跑啦。

[大佬的知乎原文](https://zhuanlan.zhihu.com/p/467730793)

[AutoLableImg原版（基于Yolov5）](https://zhuanlan.zhihu.com/p/467730793)

[yolov7 的权重下载链接](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

虽然文件名还有pytorch_yolov5，但是可以选择yolov7权重，只是懒得改文件名了...

仅支持目前（24年4月）yolov7的最新版本v1.13.1，向下和向上兼容性不保证。

**（！重要！）请将权重置于pytorch_yolov5\weights\中。**

使用方法
```
python labelImg.py
```

你可以通过拷贝我的conda来配置环境，请修改文件夹下的**environment.yml**文件的最后一行：**prefix: C:\Users\adamin\.conda\envs\labelme**为你的conda地址。如果你不知道conda的位置，请通过where conda(Windows)或whereis conda(Linux)来查看conda位置。之后，通过指令

```
conda env create -f environment.yml
```

来安装对应的环境，并通过

```
conda activate labelme
```

来启动它。

# Citation

```
{   AutoLabelImg,
    author = {Wu Fan},
    year = {2020},
    url = {\url{https://https://github.com/wufan-tb/AutoLabelImg}}
}
```
