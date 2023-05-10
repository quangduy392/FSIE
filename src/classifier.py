import cv2
import copy
import numpy as np
import math
import time
import traceback
import utility as utility
from utility import get_image_file_list, check_and_read

class ClsPostProcess(object):
    def __init__(self, label_list=None, key=None, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list
        self.key = key

    def __call__(self, preds, label=None, *args, **kwargs):
        if self.key is not None:
            preds = preds[self.key]

        label_list = self.label_list
        if label_list is None:
            label_list = {idx: idx for idx in range(preds.shape[-1])}

        pred_idxs = preds.argmax(axis=1)
        decode_out = [(label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        label = [(label_list[idx], 1.0) for idx in label]
        return decode_out, label

class TextClassifier(object):
    def __init__(self, args):
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.cls_batch_num
        self.cls_thresh = args.cls_thresh
        self.cls_labels = args.label_list
        self.cls_postprocess = ClsPostProcess()
        postprocess_params = {
            'name': 'ClsPostProcess',
            "label_list": args.label_list,
        }
        # self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, _ = \
            utility.create_predictor(args, 'cls')

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list1 = copy.deepcopy(img_list)
        img_num = len(img_list)
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):

            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            starttime = time.time()
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list1[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list1[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            outputs = self.predictor.run(self.output_tensors, input_dict)
            prob_out = outputs[0]
            cls_result = self.cls_postprocess(prob_out)
            elapse += time.time() - starttime
            
        return img_list, cls_res, elapse



def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_classifier = TextClassifier(args)
    valid_image_file_list = []
    img_list = []
    for i in range(len(image_file_list)):
        img, flag, _ = check_and_read(image_file_list[i])
        if not flag:
            img = cv2.imread(image_file_list[i])
            # cv2.namedWindow("input", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
            # # im = cv2.imread("earth.jpg")                    # Read image
            # imS = cv2.resize(img, (3840, 2160)) 
            # cv2.imshow('input',imS)
            # cv2.waitKey(0)
        if img is None:
            print("error in loading image:{}".format(image_file_list[i]))
            continue
        valid_image_file_list.append(image_file_list[i])
        img_list.append(img)
    try:
        img_out, cls_res, predict_time = text_classifier(img_list)
    #     cv2.namedWindow("input", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    #         # im = cv2.imread("earth.jpg")                    # Read image
    #     imO = cv2.resize(img_out, (900, 1500))
    #     cv2.imshow('output',imO)
    #     cv2.waitKey(0)
    except Exception as E:
        print(traceback.format_exc())
        print(E)
        exit()
    for ino in range(len(img_list)):
        print("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               cls_res[ino]))


if __name__ == "__main__":
    main(utility.parse_args())
