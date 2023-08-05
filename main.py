import cv2
import os
import shutil
from structure.predict_system import StructureSystem, save_structure_res
from utils.utility import parse_args, get_image_file_list, check_and_read, parse_args, draw_structure_result, draw_ser_results, draw_re_results, concat_docx2html
from flask import *  
from utils.logging import get_logger

logger = get_logger()

app = Flask(__name__)

def run(image_dir):
    image_file_list = get_image_file_list(image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[process_id::total_process_num]

    img_num = len(image_file_list)

    for i, image_file in enumerate(image_file_list):
        table_info= {
        "pre" : 0,
        "now" : 0,
        "pre_name" : ''
        }
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
        pre_page ={
            'index_pre' : 10000,
            'header_pre' : '',
            'table_index' : 0
        }
        for index, img in enumerate(imgs):
            res, time_dict, table_index_now  = structure_sys(img, pre_page, img_idx=index)
            table_info['now'] = table_index_now
            img_save_path = os.path.join(save_folder, img_name,
                                         'show_{}.jpg'.format(index))
            os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
            if structure_sys.mode == 'structure' and res != []:
                draw_img = draw_structure_result(img, res, vis_font_path)
                save_structure_res(res, save_folder, img_name, table_info, index)
                table_info['pre'] = table_index_now
            elif structure_sys.mode == 'kie':
                if structure_sys.kie_predictor.predictor is not None:
                    draw_img = draw_re_results(
                        img, res, font_path = vis_font_path)
                else:
                    draw_img = draw_ser_results(
                        img, res, font_path = vis_font_path)

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
            if recovery and res != []:
                from structure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
                h, w, _ = img.shape
                res = sorted_layout_boxes(res, w)
                all_res += res

        if recovery and all_res != []:
            try:
                docx_path = convert_info_docx(img, all_res, save_folder, img_name)
                # concat_docx2html(docx_path)
            except Exception as ex:
                logger.error("error in layout recovery image:{}, err msg: {}".
                             format(image_file, ex))
                continue
        logger.info("Predict time : {:.3f}s".format(time_dict['all']))
    return img_name

@app.route('/')  
def main():  
    return render_template("index.html")

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_file(filename)

@app.route('/success', methods=['POST'])  
def success(): 
    if request.method == 'POST': 
        for filename in os.listdir(save_folder):
            file_path = os.path.join(save_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        f = request.files['file']
        f.save('input/' + f.filename)
        child_save_folder = run('input/' + f.filename)
        os.remove('input/' + f.filename)
        shutil.make_archive(save_folder + '/' + child_save_folder, 'zip', save_folder + '/' + child_save_folder)
        # try:
        #     return send_file(save_folder + '/' + child_save_folder + '.zip')
        # except Exception as e:
        #     return str(e)
        for filename in os.listdir(zip_dir):
            file_path = os.path.join(zip_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        shutil.copy2(save_folder + '/' + child_save_folder + '.zip', zip_dir)
    return render_template("output.html", filename= zip_dir + child_save_folder + '.zip')
    

if __name__ == "__main__":
    args = parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    process_id = args.process_id
    total_process_num = args.total_process_num
    vis_font_path = args.vis_font_path
    recovery = args.recovery
    structure_sys = StructureSystem(args)
    save_folder = args.output
    os.makedirs(save_folder, exist_ok=True)
    zip_dir = 'static/output/zip/' 
    os.makedirs(os.path.join(zip_dir), exist_ok=True)
    figure_dir = 'static/output/figure/'
    os.makedirs(os.path.join(figure_dir), exist_ok=True)
    for filename in os.listdir(save_folder):
        file_path = os.path.join(save_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    img_num = len(image_file_list)
    app.run(host = '127.0.0.1', port = '8686', debug = False)


    run(args)

