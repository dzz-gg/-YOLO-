from ultralytics import YOLO
if __name__ == '__main__':

    model = YOLO('../ultralytics/cfg/models/11/yolo11.yaml')  # build a new model from YAML
    model.train()

