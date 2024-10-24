from ultralytics import YOLO
import time


def train(regime_name, epochs=100):
    seed = int(time.time())
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
    model.train(data="./data/" + regime_name, epochs=epochs, imgsz=512, erasing=0, batch=8, seed=seed)

if __name__ == '__main__':

    dataset = "tracks_indeed_actually"

    # regime_name = f"{dataset} (snow track train)"
    # regime_name = f"{dataset} (other track train)"
    regime_name =  f"{dataset} (mix train)"

    train(regime_name, 500)
