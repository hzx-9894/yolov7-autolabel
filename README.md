# yolov7-autolabel
基于知乎用户**滴水穿石​**的自动标注工具AutoLableImg改进的yolov7+cpu版本，不需要启动GPU就可以跑啦。
[大佬的知乎原文](https://zhuanlan.zhihu.com/p/467730793)
[AutoLableImg原版（基于Yolov5）](https://zhuanlan.zhihu.com/p/467730793)
[yolov7 的权重下载链接](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)
虽然文件名还有pytorch_yolov5，但是可以选择yolov7权重，只是懒得改文件名了...
仅支持目前（24年4月）yolov7的最新最版本v1.13.1，向下和向上兼容性不保证。
请将权重置于pytorch_yolov5\weights\中。
使用方法：
```
python labelImg.py
``` 
