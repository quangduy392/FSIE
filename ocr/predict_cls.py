import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import numpy as np
import time
from ocr.postprocess import config
import utils.utility as utility
from utils.utility import get_image_file_list, check_and_read
from ocr.postprocess.cls_pre_postprocess import *
from utils.logging import get_logger
logger = get_logger()


class TextClassifier(object):
    def __init__(self, args):
        configs = config.get_config(args.cls_config, show=True)
        self.preprocess_ops = []
        self.postprocess = None
        if "PreProcess" in configs:
            if "transform_ops" in configs["PreProcess"]:
                self.preprocess_ops = create_operators(configs["PreProcess"][
                    "transform_ops"])
        if "PostProcess" in configs:
            self.postprocess = build_postprocess(configs["PostProcess"])
        self.predictor, self.input_tensor, self.output_tensors, _ = \
            utility.create_predictor(args, 'cls', logger)

    def __call__(self, images):
        img_ori = images[0].copy()
        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])

        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_handle(output_names[0])
        elapse = 0
        starttime = time.time()
        if not isinstance(images, (list, )):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)

        input_tensor.copy_from_cpu(image)
        self.predictor.run()
        batch_output = output_tensor.copy_to_cpu()

        if self.postprocess is not None:
            batch_output = self.postprocess(batch_output)
        elapse += time.time() - starttime
        cv_rotate_code = {
                    '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
                    '180': cv2.ROTATE_180,
                    '270': cv2.ROTATE_90_CLOCKWISE
                }
        print('\n===========================')
        print(batch_output[0]['label_names'][0])
        if batch_output[0]['label_names'][0] in cv_rotate_code:
            img_ori = cv2.rotate(img_ori, cv_rotate_code[batch_output[0]['label_names'][0]])
        return img_ori, batch_output, elapse


def main(config):
    cls_predictor = TextClassifier(config)
    image_list = get_image_file_list(config["Global"]["infer_imgs"])

    batch_imgs = []
    batch_names = []
    cnt = 0
    for idx, img_path in enumerate(image_list):
        img, flag, _ = check_and_read(img_path)
        if img is None:
            print(
                "Image file failed to read and has been skipped. The path: {}".
                format(img_path))
        else:
            img = img[:, :, ::-1]
            batch_imgs.append(img)
            img_name = os.path.basename(img_path)
            batch_names.append(img_name)
            cnt += 1

        if cnt % config["Global"]["batch_size"] == 0 or (idx + 1
                                                         ) == len(image_list):
            if len(batch_imgs) == 0:
                continue
            batch_results = cls_predictor(batch_imgs)
            for number, result_dict in enumerate(batch_results):
                if "PersonAttribute" in config[
                        "PostProcess"] or "VehicleAttribute" in config[
                            "PostProcess"]:
                    filename = batch_names[number]
                    print("{}:\t {}".format(filename, result_dict))
                else:
                    filename = batch_names[number]
                    clas_ids = result_dict["class_ids"]
                    scores_str = "[{}]".format(", ".join("{:.2f}".format(
                        r) for r in result_dict["scores"]))
                    label_names = result_dict["label_names"]
                    print(
                        "{}:\tclass id(s): {}, score(s): {}, label_name(s): {}".
                        format(filename, clas_ids, scores_str, label_names))
            batch_imgs = []
            batch_names = []
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    main(config)
