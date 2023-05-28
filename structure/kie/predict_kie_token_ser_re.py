# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import numpy as np
import time
import paddle
import utils.utility as utility

from ocr.postprocess import build_post_process
from utils.logging import get_logger
from utils.utility import draw_ser_results, draw_re_results
from utils.utility import get_image_file_list, check_and_read
from utils.utility import parse_args
from kie.predict_kie_token_ser import SerPredictor

logger = get_logger()

def make_input(ser_inputs, ser_results):
    entities_labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2}
    batch_size, max_seq_len = ser_inputs[0].shape[:2]
    entities = ser_inputs[8][0]
    ser_results = ser_results[0]
    assert len(entities) == len(ser_results)

    # entities
    start = []
    end = []
    label = []
    entity_idx_dict = {}
    for i, (res, entity) in enumerate(zip(ser_results, entities)):
        if res['pred'] == 'O':
            continue
        entity_idx_dict[len(start)] = i
        start.append(entity['start'])
        end.append(entity['end'])
        label.append(entities_labels[res['pred']])

    entities = np.full([max_seq_len + 1, 3], fill_value=-1, dtype=np.int64)
    entities[0, 0] = len(start)
    entities[1:len(start) + 1, 0] = start
    entities[0, 1] = len(end)
    entities[1:len(end) + 1, 1] = end
    entities[0, 2] = len(label)
    entities[1:len(label) + 1, 2] = label

    # relations
    head = []
    tail = []
    for i in range(len(label)):
        for j in range(len(label)):
            if label[i] == 1 and label[j] == 2:
                head.append(i)
                tail.append(j)

    relations = np.full([len(head) + 1, 2], fill_value=-1, dtype=np.int64)
    relations[0, 0] = len(head)
    relations[1:len(head) + 1, 0] = head
    relations[0, 1] = len(tail)
    relations[1:len(tail) + 1, 1] = tail

    entities = np.expand_dims(entities, axis=0)
    entities = np.repeat(entities, batch_size, axis=0)
    relations = np.expand_dims(relations, axis=0)
    relations = np.repeat(relations, batch_size, axis=0)

    # remove ocr_info segment_offset_id and label in ser input
    if isinstance(ser_inputs[0], paddle.Tensor):
        entities = paddle.to_tensor(entities)
        relations = paddle.to_tensor(relations)
    ser_inputs = ser_inputs[:5] + [entities, relations]

    entity_idx_dict_batch = []
    for b in range(batch_size):
        entity_idx_dict_batch.append(entity_idx_dict)
    return ser_inputs, entity_idx_dict_batch

def to_tensor(data):
    import numbers
    from collections import defaultdict
    data_dict = defaultdict(list)
    to_tensor_idxs = []

    for idx, v in enumerate(data):
        if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
            if idx not in to_tensor_idxs:
                to_tensor_idxs.append(idx)
        data_dict[idx].append(v)
    for idx in to_tensor_idxs:
        data_dict[idx] = paddle.to_tensor(data_dict[idx])
    return list(data_dict.values())

    
class SerRePredictor(object):
    def __init__(self, args):
        self.use_visual_backbone = args.use_visual_backbone
        self.ser_engine = SerPredictor(args)
        if args.re_model_dir is not None:
            postprocess_params = {'name': 'VQAReTokenLayoutLMPostProcess'}
            self.postprocess_op = build_post_process(postprocess_params)
            self.predictor, self.input_tensor, self.output_tensors, self.config = \
                utility.create_predictor(args, 're', logger)
        else:
            self.predictor = None

    def __call__(self, img):
        starttime = time.time()
        ser_results, ser_inputs, ser_elapse = self.ser_engine(img)
        if self.predictor is None:
            return ser_results, ser_elapse

        re_input, entity_idx_dict_batch = make_input(ser_inputs, ser_results)
        if self.use_visual_backbone == False:
            re_input.pop(4)
        for idx in range(len(self.input_tensor)):
            self.input_tensor[idx].copy_from_cpu(re_input[idx])

        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        preds = dict(
            loss=outputs[1],
            pred_relations=outputs[2],
            hidden_states=outputs[0], )

        post_result = self.postprocess_op(
            preds,
            ser_results=ser_results,
            entity_idx_dict_batch=entity_idx_dict_batch)

        elapse = time.time() - starttime
        return post_result, elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    ser_re_predictor = SerRePredictor(args)
    count = 0
    total_time = 0

    os.makedirs(args.output, exist_ok=True)
    with open(
            os.path.join(args.output, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
                img = img[:, :, ::-1]
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            re_res, elapse = ser_re_predictor(img)
            re_res = re_res[0]

            res_str = '{}\t{}\n'.format(
                image_file,
                json.dumps(
                    {
                        "ocr_info": re_res,
                    }, ensure_ascii=False))
            f_w.write(res_str)
            if ser_re_predictor.predictor is not None:
                img_res = draw_re_results(
                    image_file, re_res, font_path=args.vis_font_path)
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] +
                    "_ser_re.jpg")
            else:
                img_res = draw_ser_results(
                    image_file, re_res, font_path=args.vis_font_path)
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] +
                    "_ser.jpg")

            cv2.imwrite(img_save_path, img_res)
            logger.info("save vis result to {}".format(img_save_path))
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    main(parse_args())
