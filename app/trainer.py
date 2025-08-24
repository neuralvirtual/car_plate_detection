from ultralytics import YOLO

class Trainer:
    def __init__(self, task='', mode=None, model='', imgsz=640, data='',
                 device=0, epochs=None, batch=None, learning_rate=None, optimizer=None, weight_decay=None, name='', exist_ok=True):
        self.task = task
        self.mode = mode
        self.model = model
        self.imgsz = imgsz
        self.data = data
        self.device = device
        self.epochs = epochs
        self.batch = batch
        self.name = name
        self.exist_ok = exist_ok
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    def load_model(self):
        model = YOLO(self.model)
        return model

    def train(self):
        model = self.load_model()
        model.train(task=self.task, mode=self.mode, data=self.data, device=self.device,
                epochs=self.epochs, batch=self.batch, imgsz=self.imgsz,
                name=self.name, exist_ok=self.exist_ok, lr0=self.learning_rate,
                optimizer=self.optimizer, weight_decay=self.weight_decay)