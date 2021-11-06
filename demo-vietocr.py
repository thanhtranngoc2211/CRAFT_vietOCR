from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os 
from PIL import Image

def main() :
    img_files = []
    config = Cfg.load_config_from_name('vgg_transformer')

    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    config['predictor']['beamsearch']=False

    detector = Predictor(config)
    for (dirpath, dirnames, filenames) in os.walk(r'D:\CODE\Python\NAVER\final\result_final_test'):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
    for x in img_files:
        img = Image.open(x)

        result = detector.predict(img, return_prob=True)

        print(result)

if __name__ == '__main__':
    main()

