from ultralytics import YOLO

class Detector:
    def __init__(self,  mode='predict', model='', source='', device=None, exist_ok=True):
        self.mode = mode
        self.model = model
        self.source = source
        self.device = device
        self.exist_ok = exist_ok

    def load_model(self):
        model = YOLO(self.model)
        model.to('cpu')
        return model

    def predict(self, img):
        model = self.load_model()
        return model.predict(source=img, save=False)