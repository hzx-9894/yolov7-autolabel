import codecs,os,sys,platform,subprocess,random,natsort,cv2,easygui
import xml.etree.ElementTree as ET
from copy import deepcopy
from strsimpy.jaro_winkler import JaroWinkler
import numpy as np
from skimage import exposure
from functools import partial

sys.path.append('pytorch_yolov5/')
from models.experimental import *
from utils.datasets import *
from utils.utils import *

from libs.combobox import ComboBox
from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem

def yolov5_auto_labeling():
    try:
        tree = ET.ElementTree(file='./data/origin.xml')
        root = tree.getroot()
        for child in root.findall('object'):
            template_obj = child  # 保存一个物体的样板
            root.remove(child)
        tree.write('./data/template.xml')

        # =====def some function=====
        def change_obj_property(detect_result, template_obj):
            temp_obj = template_obj
            for child in temp_obj:
                key = child.tag
                if key in detect_result.keys():
                    child.text = detect_result[key]
                if key == 'bndbox':
                    for gchild in child:
                        gkey = gchild.tag
                        gchild.text = str(detect_result[gkey])
            return temp_obj

        def change_result_type_yolov5(boxes, scores, labels):
            result = []
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.3:
                    try:
                        new_obj = {}
                        new_obj['name'] = label
                        new_obj['xmin'] = int(box[0])
                        new_obj['ymin'] = int(box[1])
                        new_obj['xmax'] = int(box[2])
                        new_obj['ymax'] = int(box[3])
                        result.append(new_obj)
                    except:
                        print('labels_info have no label: ' + str(label))
                        pass
            return result

        print('请输入图片路径。')
        filePath = input()
        source = os.path.dirname(filePath)
        xml_path = filePath

        print('请输入权重路径。')
        weights = input()


        '''        
        weight_list = []
        for item in sorted(os.listdir(weight_path)):
            if item.endswith('.h5') or item.endswith('.pt') or item.endswith('.pth'):
                weight_list.append(item)
        items = tuple(weight_list)
        if len(weight_list) > 0:
            weights, ok = QInputDialog.getItem(self, "Select",
                                               "Model weights file(weights file should under 'pytorch_yolov5/weights'):",
                                               items, 0, False)
            if not ok:
                return
            else:
                weights = os.path.join(weight_path, weights)
        else:
            weights, _ = QFileDialog.getOpenFileName(self,
                                                     "'pytorch_yolov5/weights' is empty, choose model weights file:")
            if not (weights.endswith('.pt') or weights.endswith('.pth')):
                QMessageBox.information(self, u'Wrong!', u'weights file must endswith .h5 or .pt or .pth')
                return
        '''
        conf_thres = 0.5
        iou_thres = 0.5
        # Initialize
        device = torch.device('cpu')
        half = False  # 不使用半精度
        # Load model and label name.
        model = attempt_load(weights, map_location=device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names
        print('该权重总共有',len(names),'个种类，为',names,'。请输入要自动标注的种类。')
        needed_labels=input()
        if needed_labels not in names:
            print('没有该种类！')
            return
        # set imsize
        imgsz = 1280
        imgsz = check_img_size(imgsz, s=model.stride.max())
        if half:
            model.half()  # to FP16

        # load img and run inference
        dataset = LoadImages(source, img_size=imgsz)
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        index = -1


        for path, img, im0s, vid_cap in dataset:
            index += 1
            img = torch.from_numpy(img).to(device)
            #print('1ok')
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            #print('2ok')
            # Inference
            pred = model(img, augment=False)[0]
            #print('3ok')
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
            t2 = torch_utils.time_synchronized()
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                labels = []
                scores = []
                boxes = []
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # Write results
                    for *xyxy, conf, cls in det:
                        if names[int(cls)] not in needed_labels:
                            continue
                        labels.append(names[int(cls)])
                        scores.append(conf.item())
                        boxes.append(
                            [int(xyxy[0].item()), int(xyxy[1].item()), int(xyxy[2].item()), int(xyxy[3].item())])
                tree = ET.ElementTree(file='./data/template.xml')
                root = tree.getroot()
                common_property = {'filename': path.split('\\')[-1], 'path': source, 'folder': 'JPEGImages'}
                for child in root:
                    key = child.tag
                    if key in common_property.keys():
                        child.text = common_property[key]
                result = change_result_type_yolov5(boxes, scores, labels)
                if len(result) > 0:
                    for j in range(len(result)):
                        new_obj = change_obj_property(result[j], template_obj)
                        root.append(deepcopy(new_obj))  # 深度复制
                        # !!!这块没直接append(new_obj)是因为当增加多个节点的话，new_obj会进行覆盖，必须要用深度复制以进行区分
                tree.write(os.path.join(xml_path, path[len(os.path.dirname(path)) + 1:-4] + '.xml'))
        print(u'Done!', u'auto labeling done, please reload img folder')
    except Exception as e:
        print(u'Sorry!', u'something is wrong. ({})'.format(e))


if __name__ == '__main__':
    with torch.no_grad():
        yolov5_auto_labeling()