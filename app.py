'''Pipeline's demo'''
import os
import random
import shutil
import time
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect
from flask_caching import Cache
from requests import Response
from pipeline.models_pipeline import LabelLocalizationModel, ObjectDetectionModel, \
    AddrIdentificationModel, AddrClassificationModel, easyOCR
from pipeline.utils_.functions import get_masks_from, crop_mask, rotate_image
from pipeline.utils_.scanner import Extractor

app = Flask(__name__, template_folder='templates/', static_url_path='/static')

LL_CLS_NAME = ['Label', 'Text', 'Logo', 'Other marks']

config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIME": 300,
    "UPLOAD_FOLDER": '.temp'
}
app.config.from_mapping(config)
cache = Cache(app)

ll_model = LabelLocalizationModel("./pipeline/weights/ll_model_final.pth","./pipeline/weights/IS5_cfg.pickle", mask=True)
yolo = ObjectDetectionModel("./pipeline/weights/best-yolov5x6.pt", "pipeline/yolov5/",0.1)
ocr = easyOCR()
idt_model = AddrIdentificationModel(
    token_model_path="./pipeline/weights/00_addr_identification_token.h5",
    tokenizer_path="./pipeline/weights/tokenizer_ident.pickle",
    vector_model_path="./pipeline/weights/glove_embedding_identification.h5",
    vectorizer_path="./pipeline/weights/model.glove"
    )
clf_model = AddrClassificationModel("./pipeline/weights/00_addr_clf.h5","./pipeline/weights/tokenizer_clf.pickle")
ext = Extractor()


@app.route('/')
def home():
    '''home route'''
    return render_template('link-form.html')

@app.route('/submit-form', methods=['POST'])
def submit_file() -> Response:
    '''route for saving a file'''
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],'uploaded_file.jpg'))
        return redirect('/demo-processing')

@app.route('/demo-processing')
def demo_processing():
    if os.path.isdir('static/models_outputs/'):
        shutil.rmtree('static/models_outputs/')
    os.mkdir('static/models_outputs/')

    img = cv2.imread('.temp/uploaded_file.jpg')

    times = []
    all_txt = []

    start = time.time()
    out_ll = ll_model.predict(img, file_name=f'./static/models_outputs/ll_result.jpg', save_result=True)
    times.append(time.time()-start)

    masks, ll_scores, ll_cls = get_masks_from(out_ll, [0,1,2,3])

    cropped_bboxes = [crop_mask(mask, '.temp/uploaded_file.jpg', margin=30) for mask in masks]

    img_lst = []

    os.mkdir('./static/models_outputs/ll')
    os.mkdir('./static/models_outputs/redress')
    
    start = time.time()
    for idx, img in enumerate(cropped_bboxes):
        try:
            cv2.imwrite(f'./static/models_outputs/ll/cropped{idx}.jpg', img)
            out = ext(img)[-1]

            cv2.imwrite(f'./static/models_outputs/redress/{idx}.jpg',out)

            img_lst.append(out)
        except:
            pass
    times.append(time.time()-start)

    ll_dir = os.listdir('./static/models_outputs/ll')
    redress_dir = os.listdir('./static/models_outputs/redress')
    ll_cls=list(ll_cls)

    ll_dir = sorted(ll_dir)
    if len(img_lst) > 0:
        start = time.time()
        yolo_bboxes = yolo.predict(img_lst)
        times.append(time.time()-start)
    else :
        return render_template(
            'demo.html',
            times=times*1000,
            ll_dir = ll_dir,
            ll_metrics={
                'scores': ll_scores*100,
                'cls': [ll_cls.count(i) for i in range(4)]
            },
            redress_dir= redress_dir
            )

    # yolo_bboxes.save(save_dir='static/yolo')

    if not os.path.exists('./static/models_outputs/ocr'):
        os.mkdir('./static/models_outputs/ocr')

    # bboxes = yolo_bboxes.crop(save=False, )
    # bboxes = [np.array(bboxes[i]["im"]) for i in range(len(bboxes))]
    start = time.time()   
    for idx, od_bbox in enumerate(yolo_bboxes.crop(save=True, save_dir='./static/models_outputs/yolo/')):
        img = od_bbox['im']

        if img.shape[0] > img.shape[1]:
            img = rotate_image(img, 90)

        out_ocr = ocr.predict(
            image=img,
            low_text=0.5,
            threshold=0.5,
            min_size=5,
            mag_ratio=3,
            paragraph=True,
            detail=1,
            bbox_min_size=1,
            contrast_ths=0.3,
            adjust_contrast=0.5,
            rotation_info=[180]
        )

        if len(out_ocr) > 12:
            all_txt.append(out_ocr)

        cv2.imwrite(f'./static/models_outputs/ocr/{random.randint(0,99999)}.jpg',img)
    times.append(time.time()-start)

    yolo_dir = os.listdir('./static/models_outputs/ocr')

    if len(all_txt) < 1:
        return render_template(
            'demo.html',
            times=times*1000,
            ll_dir = ll_dir,
            ll_metrics={
                'scores': ll_scores*100,
                'cls': [ll_cls.count(i) for i in range(4)]
            },
            redress_dir= redress_dir,
            yolo_dir=yolo_dir
        )


    start = time.time()
    out = idt_model.predict(all_txt)
    times.append(time.time()-start)

    out = np.array(out)

    addr_idx = [ (idx, conf) for idx, conf in sorted(enumerate(out[:,0]), key=lambda x: x[1])]    

    addr_lst = []
    addr_lst_sorted = []
    idt_conf_lst = []
    for (i, conf) in addr_idx:
        addr_lst_sorted.append(all_txt[i])
        idt_conf_lst.append( conf)  
    all_txt = addr_lst_sorted
    addr_lst = addr_lst_sorted[-2:] 

    idt_conf_lst = np.array(idt_conf_lst)
    if len(addr_lst) > 1:
        start = time.time()
        out_lst = clf_model.predict([addr_lst[0]], [addr_lst[1]])
        times.append(time.time()-start)
    elif len(addr_lst) > 0:
        start = time.time()
        out_lst = clf_model.predict([addr_lst[0]], [''])
        times.append(time.time()-start)
        addr_lst.append('')

    out = list(zip(addr_lst, out_lst))

    times = np.array(times)
    times = times*1000

    yolo_dir = os.listdir('./static/models_outputs/ocr')
    ocr_dir = os.listdir('./static/models_outputs/ocr')


    return render_template(
        'demo.html', 
        yolo_dir=yolo_dir, 
        all_txt=all_txt,
        addr_lst=addr_lst, 
        out=out,
        ocr_dir=ocr_dir, 
        ll_dir=ll_dir,
        ll_metrics={
            'scores': ll_scores*100,
            'cls': [ll_cls.count(i) for i in range(4)]
        },
        idt_metrics={
            'scores': idt_conf_lst*100
        },
        redress_dir=redress_dir,
        times=times
    )