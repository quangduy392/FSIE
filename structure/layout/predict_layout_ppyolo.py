# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import numpy as np
import paddle
import time

import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
from preprocess import preprocess, Resize, NormalizeImage, Permute
import utils.utility as utility
from utils.utility import Timer
from utils.logging import get_logger

logger = get_logger()


class PPYoloDetector(object):

    def __init__(self, args):
        model_dir = args.layout_ppyolo_model_dir
        self.labels = self.load_layout_dict(args.layout_dict_path)
        self.pred_config = self.set_config(model_dir)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'layout_ppyolo', logger)
        
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.threshold = args.layout_score_threshold

    def load_layout_dict(self, layout_dict_path):
        with open(layout_dict_path, 'r', encoding='utf-8') as fp:
            labels = fp.readlines()
        return [label.strip('\n') for label in labels]
    
    def set_config(self, model_dir):
        return PredictConfig(model_dir)

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        im, im_info = preprocess(image_list, preprocess_ops)
        inputs = create_inputs(im, im_info)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']
        assert isinstance(np_boxes_num, np.ndarray), \
            '`np_boxes_num` should be a `numpy.ndarray`'

        result = {k: v for k, v in result.items() if v is not None}
        return result

    def filter_box(self, result, threshold):
        np_boxes_num = result['boxes_num']
        boxes = result['boxes']
        start_idx = 0
        filter_boxes = []
        filter_num = []
        for i in range(len(np_boxes_num)):
            boxes_num = np_boxes_num[i]
            boxes_i = boxes[start_idx:start_idx + boxes_num, :]
            idx = boxes_i[:, 1] > threshold
            filter_boxes_i = boxes_i[idx, :]
            filter_boxes.append(filter_boxes_i)
            filter_num.append(filter_boxes_i.shape[0])
            start_idx += boxes_num
        boxes = np.concatenate(filter_boxes)
        filter_num = np.array(filter_num)
        filter_res = {'boxes': boxes, 'boxes_num': filter_num}
        return filter_res

    def predict(self, repeats=1, run_benchmark=False):
        # model prediction
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None

        if run_benchmark:
            for i in range(repeats):
                self.predictor.run()
                paddle.device.cuda.synchronize()
            result = dict(
                boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
            return result

        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if len(output_names) == 1:
                # some exported model can not get tensor 'bbox_num' 
                np_boxes_num = np.array([len(np_boxes)])
            else:
                boxes_num = self.predictor.get_output_handle(output_names[1])
                np_boxes_num = boxes_num.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def __call__(self, image_list):
        resultouts = []
        starttime = time.time()
        # preprocess
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image_list)
        self.det_times.preprocess_time_s.end()

        # model prediction
        self.det_times.inference_time_s.start()
        result = self.predict()
        self.det_times.inference_time_s.end()

        # postprocess
        self.det_times.postprocess_time_s.start()
        result = self.postprocess(inputs, result)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += len(image_list)

        elapse = time.time() - starttime
        np_boxes = result['boxes']
        
        expect_boxes = (np_boxes[:, 1] > self.threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        for dt in np_boxes:
            clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
            label = self.labels[clsid]
            resultout = {'bbox': bbox, 'label': label}
            resultouts.append(resultout)
        return resultouts, elapse

def create_inputs(imgs, im_info):
    inputs = {}

    inputs['image'] = np.array((imgs, )).astype('float32')
    inputs['im_shape'] = np.array(
        (im_info['im_shape'], )).astype('float32')
    inputs['scale_factor'] = np.array(
        (im_info['scale_factor'], )).astype('float32')
    return inputs

class PredictConfig():
    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']

        