import numpy as np
import argparse
import cv2 as cv
import time
from yolo_utils import infer_image, show_image

FLAGS = []

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.8,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.8')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion.')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file.')

	parser.add_argument('-url', '--url-video',
		type=str,
		help='The url to the video online.')

	parser.add_argument('-cn', '--camera-number',
		type=int,
		default=0,
		help='The camera number to be connected.')

	parser.add_argument('-cs', '--camera-show',
		type=bool,
		help='Show or not the camera image.')

	parser.add_argument('-o', '--object',
		type=str,
		default='bus',
		help='Object you want to detect.')

	FLAGS, unparsed = parser.parse_known_args()


	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		
	objDetect = 0
	if FLAGS.video_path:
		# Read the video
		try:
			vid = cv.VideoCapture(FLAGS.video_path)
			height, width = None, None
			writer = None
		except:
			raise 'Video cannot be loaded!\n\
							Please check the path provided!'

		finally:
			start = 0
			end = 0
			grabbed, frame = vid.read()
			while True:
				for i in range(int((end-start) * vid.get(cv.CAP_PROP_FPS))):
					grabbed, frame = vid.read()
				start = time.time()
				# Checking if the complete video is read
				if not grabbed:
					break

				if width is None or height is None:
					height, width = frame.shape[:2]

				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

				if objDetect < len(classids):
					for i in range(len(classids) - objDetect):
						print(time.ctime(), "-", FLAGS.object, "detect")
				objDetect = len(classids)

				cv.imshow('video', frame)

				if cv.waitKey(1) & 0xFF == ord('q'):
					break

				end = time.time()
			vid.release()
			cv.destroyAllWindows()

	elif FLAGS.url_video:
		print ('Starting Inference on Webcam')
		print ('For close, press ctrl+c on terminal or q on window')
		# Infer real-time on webcam
		count = 0

		vid = cv.VideoCapture(FLAGS.url_video)
		while True:
			_, frame = vid.read()
			height, width = frame.shape[:2]

			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
									height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
									height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6
			
			if objDetect < len(classids):
					for i in range(len(classids) - objDetect):
						print(time.ctime(), "-", FLAGS.object, "detect")
			objDetect = len(classids)

			cv.imshow('online', frame)

			if cv.waitKey(1) & 0xFF == ord('q'):
				break
		vid.release()
		cv.destroyAllWindows()

	elif FLAGS.camera_show:
		print ('Starting Inference on Webcam')
		print ('For close, press ctrl+c on terminal or q on window')
		# Infer real-time on webcam
		count = 0

		vid = cv.VideoCapture(FLAGS.camera_number)
		objDetect = False;
		while True:
			_, frame = vid.read()
			height, width = frame.shape[:2]

			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
									height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
									height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6
			
			if objDetect < len(classids):
					for i in range(len(classids) - objDetect):
						print(time.ctime(), "-", FLAGS.object, "detect")
			objDetect = len(classids)

			cv.imshow('webcam', frame)

			if cv.waitKey(1) & 0xFF == ord('q'):
				break
		vid.release()
		cv.destroyAllWindows()

	else:
		print ('Starting Inference on Webcam')
		print ('For close, press ctrl+c')
		# Infer real-time on webcam
		count = 0

		vid = cv.VideoCapture(FLAGS.camera_number)
		objDetect = False;
		while True:
			_, frame = vid.read()
			height, width = frame.shape[:2]

			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
									height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
									height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6
			
			if objDetect < len(classids):
					for i in range(len(classids) - objDetect):
						print(time.ctime(), "-", FLAGS.object, "detect")
			objDetect = len(classids)


		vid.release()