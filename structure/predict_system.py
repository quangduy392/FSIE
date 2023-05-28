import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import numpy as np
import time
import logging
from copy import deepcopy

from utils.utility import get_image_file_list, check_and_read, parse_args, draw_structure_result, draw_ser_results, draw_re_results
from utils.logging import get_logger
import ocr.predict_cls as classifier
from ocr.predict_system import TextSystem
from structure.layout.predict_layout import LayoutPredictor
from structure.layout.predict_layout_ppyolo import PPYoloDetector
from structure.table.predict_table import TableSystem, to_excel
logger = get_logger()


class StructureSystem(object):
    def __init__(self, args):
        self.mode = args.mode
        self.recovery = args.recovery
        self.text_classifier = classifier.TextClassifier(args)
        if self.mode == 'structure':
            if not args.show_log:
                logger.setLevel(logging.INFO)
            if args.layout == False and args.ocr == True:
                args.ocr = False
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            args.drop_score = 0
            # init model
            self.layout_predictor = None
            self.text_system = None
            self.table_system = None
            if args.layout:
                if args.arch == "PicoDet":
                    self.layout_predictor = LayoutPredictor(args)
                elif args.arch == "PPYOLO":
                    self.layout_predictor = PPYoloDetector(args)
                if args.ocr:
                    self.text_system = TextSystem(args)
            if args.table:
                if self.text_system is not None:
                    self.table_system = TableSystem(
                        args, self.text_system.text_detector,
                        self.text_system.text_recognizer)
                else:
                    self.table_system = TableSystem(args)

        elif self.mode == 'kie':
            from structure.kie.predict_kie_token_ser_re import SerRePredictor
            self.kie_predictor = SerRePredictor(args)

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        time_dict = {
            'image_orientation': 0,
            'layout': 0,
            'table': 0,
            'table_match': 0,
            'det': 0,
            'rec': 0,
            'kie': 0,
            'all': 0
        }
        start = time.time()
        if self.text_classifier is not None:
            tic = time.time()
            img, angle_list, elapse = self.text_classifier([img])
            toc = time.time()
            time_dict['image_orientation'] = toc - tic
        if self.mode == 'structure':
            ori_im = img.copy()
            if self.layout_predictor is not None:
                layout_res, elapse = self.layout_predictor(img)
                time_dict['layout'] += elapse
            else:
                h, w = ori_im.shape[:2]
                layout_res = [dict(bbox=None, label='table')]
            res_list = []
            for region in layout_res:
                res = ''
                if region['bbox'] is not None:
                    x1, y1, x2, y2 = region['bbox']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                if region['label'] == 'table':
                    if self.table_system is not None:
                        res, table_time_dict = self.table_system(
                            roi_img, return_ocr_result_in_table)
                        time_dict['table'] += table_time_dict['table']
                        time_dict['table_match'] += table_time_dict['match']
                        time_dict['det'] += table_time_dict['det']
                        time_dict['rec'] += table_time_dict['rec']
                else:
                    if self.text_system is not None:
                        if self.recovery:
                            wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype)
                            wht_im[y1:y2, x1:x2, :] = roi_img
                            filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                                wht_im)
                        else:
                            filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                                roi_img)
                        time_dict['det'] += ocr_time_dict['det']
                        time_dict['rec'] += ocr_time_dict['rec']

                        # remove style char,
                        # when using the recognition model trained on the PubtabNet dataset,
                        # it will recognize the text format in the table, such as <b>
                        style_token = [
                            '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                            '</b>', '<sub>', '</sup>', '<overline>',
                            '</overline>', '<underline>', '</underline>', '<i>',
                            '</i>'
                        ]
                        res = []
                        for box, rec_res in zip(filter_boxes, filter_rec_res):
                            rec_str, rec_conf = rec_res
                            for token in style_token:
                                if token in rec_str:
                                    rec_str = rec_str.replace(token, '')
                            if not self.recovery:
                                box += [x1, y1]
                            res.append({
                                'text': rec_str,
                                'confidence': float(rec_conf),
                                'text_region': box.tolist()
                            })
                res_list.append({
                    'type': region['label'].lower(),
                    'bbox': [x1, y1, x2, y2],
                    'img': roi_img,
                    'res': res,
                    'img_idx': img_idx
                })
            end = time.time()
            time_dict['all'] = end - start
            return res_list, time_dict
        if self.mode == 'kie':
            re_res, elapse = self.kie_predictor(img)
            time_dict['kie'] = elapse
            time_dict['all'] = elapse
            return re_res[0], time_dict
        return None, None


def save_structure_res(res, save_folder, img_name, img_idx=0):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)
    # save res
    with open(
            os.path.join(excel_save_folder, 'res_{}.txt'.format(img_idx)),
            'w',
            encoding='utf8') as f:
        for region in res_cp:
            roi_img = region.pop('img')
            f.write('{}\n'.format(json.dumps(region)))

            if region['type'].lower() == 'table' and len(region[
                    'res']) > 0 and 'html' in region['res']:
                excel_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.xlsx'.format(region['bbox'], img_idx))
                to_excel(region['res']['html'], excel_path)
            elif region['type'].lower() == 'figure':
                img_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.jpg'.format(region['bbox'], img_idx))
                cv2.imwrite(img_path, roi_img)


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    structure_sys = StructureSystem(args)
    save_folder = os.path.join(args.output, structure_sys.mode)
    os.makedirs(save_folder, exist_ok=True)
    img_num = len(image_file_list)

    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag_gif, flag_pdf = check_and_read(image_file)
        img_name = os.path.basename(image_file).split('.')[0]

        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)

        if not flag_pdf:
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            imgs = img

        all_res = []
        for index, img in enumerate(imgs):
            res, time_dict = structure_sys(img, img_idx=index)
            img_save_path = os.path.join(save_folder, img_name,
                                         'show_{}.jpg'.format(index))
            os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
            if structure_sys.mode == 'structure' and res != []:
                draw_img = draw_structure_result(img, res, args.vis_font_path)
                save_structure_res(res, save_folder, img_name, index)
            elif structure_sys.mode == 'kie':
                if structure_sys.kie_predictor.predictor is not None:
                    draw_img = draw_re_results(
                        img, res, font_path=args.vis_font_path)
                else:
                    draw_img = draw_ser_results(
                        img, res, font_path=args.vis_font_path)

                with open(
                        os.path.join(save_folder, img_name,
                                     'res_{}_kie.txt'.format(index)),
                        'w',
                        encoding='utf8') as f:
                    res_str = '{}\t{}\n'.format(
                        image_file,
                        json.dumps(
                            {
                                "ocr_info": res
                            }, ensure_ascii=False))
                    f.write(res_str)
            if res != []:
                cv2.imwrite(img_save_path, draw_img)
                logger.info('result save to {}'.format(img_save_path))
            if args.recovery and res != []:
                from structure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
                h, w, _ = img.shape
                res = sorted_layout_boxes(res, w)
                all_res += res

        if args.recovery and all_res != []:
            try:
                convert_info_docx(img, all_res, save_folder, img_name)
            except Exception as ex:
                logger.error("error in layout recovery image:{}, err msg: {}".
                             format(image_file, ex))
                continue
        logger.info("Predict time : {:.3f}s".format(time_dict['all']))


if __name__ == "__main__":
    args = parse_args()
    main(args)
