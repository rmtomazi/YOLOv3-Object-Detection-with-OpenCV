# ORIGINAL -- YOLOv3-Object-Detection-with-OpenCV

https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV

# ADAPTED -- YOLOv3-Object-Detection-with-OpenCV

## How to use?

1) Clone the repository

```
git clone https://github.com/rmtomazi/YOLOv3-Object-Detection-with-OpenCV.git
```

2) Move to the directory
```
cd YOLOv3-Object-Detection-with-OpenCV
```

3) Download the https://pjreddie.com/media/files/yolov3.weights file in the yolov3-coco folder

4) To infer on a video that is stored on your local machine
```
python3 yolo.py --video-path='/path/to/video/'
```
5) To infer real-time on webcam with showing the image
```
python3 yolo.py --camera-show=True
```
6) To infer real-time on webcam without showing the image
```
python3 yolo.py
```
7) To infer real-time on a online video
```
python3 yolo.py --url-video="url"
```
8) To set the camera number (default=0)
```
python3 yolo.py --camera-number=(int)
```
9) To define the object to be detected (default="bus")
```
python3 yolo.py --object="object-name"
```
10) To set the confidence value of infer (default=0.8)
```
python3 yolo.py --confidence=(float)
```
For more details
```
yolo.py --help
```
City Traffic.mp4 - Video by Burak K from Pexels
