import argparse
import cv2
import warnings
from app.trainer import Trainer
from app.detector import Detector

warnings.filterwarnings('ignore')

MODEL_PATH = 'pretrained/best.pt'

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description="Car plate detection with YOLO"
        )

        parser.add_argument(
            '-t', '--train',
            type=bool,
            default=False,
            help="Train model with the local dataset"
        )

        parser.add_argument(
            '-i', '--image',
            default=0,
            help="Input image path"
        )

        args = parser.parse_args()

        if args.train:
            trainer = Trainer()
            trainer.train(task='detect', mode='train', model='yolo11m.pt', imgsz=640, data='dataset/plate_detection/data.yaml',
                        device=0, epochs=50, batch=16, learning_rate=0.001, optimizer='Adam', weight_decay=0.01,
                        name='plate-detection', exist_ok=True)

            print('Pretreined model weights saved in "runs/detect/plate-detection/weights/best.pt"')
            print('If you wish to use that please set it in "MODEL_PATH" variable')
        else:
            detector = Detector(model=MODEL_PATH, mode='predict', device=0, exist_ok=True)
            if args.image != 0:
                img = cv2.imread(args.image)
                results = detector.predict(img)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)


                cv2.imshow('Car plate', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print('Image not found. Please use --image [IMAGE_PATH] to detect a car plate')

    except Exception as e:
        print(f'Unexpected application error. {e}')
img