from roboflow import Roboflow

rf = Roboflow(api_key="krCxkHYiurHqmKqHjNeu")  # free at roboflow.com
project = rf.workspace("material-identification").project("garbage-classification-3")
dataset = project.version(2).download("yolov8")