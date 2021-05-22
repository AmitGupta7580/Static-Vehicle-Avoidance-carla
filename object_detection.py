#!/usr/bin/env python

from __future__ import print_function

import cv2
import time
import numpy as np

# ==============================================================================
# -- Path of Yolov4 weights and cfg file  --------------------------------------
# ==============================================================================

WEIGHT_PATH = '.\\Windows\\CARLA_0.9.11\\PythonAPI\\Models\\yolov4-tiny.weights'
CFG_PATH = '.\\Windows\\CARLA_0.9.11\\PythonAPI\\Models\\yolov4-tiny.cfg'
CLASSES_PATH = '.\\Windows\\CARLA_0.9.11\\PythonAPI\\Models\\yoloclasses.txt'

# ==============================================================================
# -- Diffrent thread for object-detection  -------------------------------------
# ==============================================================================

class ObjectDetection:
	def __init__(self, world):
		self.world = world
		self.shutdown = False

		self.net = cv2.dnn.readNet(WEIGHT_PATH, CFG_PATH)
		self.classes = []
		with open(CLASSES_PATH, "r") as f:
			self.classes = [line.strip() for line in f.readlines()]
		self.layer_names = self.net.getLayerNames()
		self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		self.coordinates = []

		print("[+] Object Detection model Loaded ... \n")

	def detect(self):
		while True:
			if self.shutdown:
				break
			if self.world.camera_manager.cam_image is not None and self.world.camera_manager.dep_image is not None:
				Z = self.world.camera_manager.dep_image*1000
				X = self.world.camera_manager.x
				img = cv2.resize(self.world.camera_manager.cam_image, None, fx=1, fy=1)
				height, width, channels = img.shape
				blob = cv2.dnn.blobFromImage(img, 0.00392, (224, 224), (0, 0, 0), True, crop=False)
				start_time = time.time()
				self.net.setInput(blob)
				outs = self.net.forward(self.output_layers)
				# print("[INFO] Prediction Time = " + str(time.time() - start_time))
				class_ids = []
				confidences = []
				boxes = []
				for out in outs:
					for detection in out:
						scores = detection[5:]
						class_id = np.argmax(scores)
						confidence = scores[class_id]
						if confidence > 0.5:
							# Object detected
							center_x = int(detection[0] * width)
							center_y = int(detection[1] * height)
							w = int(detection[2] * width)
							h = int(detection[3] * height)
							# Rectangle coordinates
							x = int(center_x - w / 2)
							y = int(center_y - h / 2)
							boxes.append([x, y, w, h, center_x, center_y])
							confidences.append(float(confidence))
							class_ids.append(class_id)

				indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
				font = cv2.FONT_HERSHEY_SIMPLEX
				object_in_frame = 0
				obj_points = []
				for i in range(len(boxes)):
					if i in indexes:
						x, y, w, h, cx, cy = boxes[i]
						label = str(self.classes[class_ids[i]])
						if label == 'car' or label == 'truck' or label == 'bus':
							# convert into real x-y plane
							obj_points.append([X[cy][cx], Z[cy][cx]])
							color = (0, 0, 255)# self.colors[i]
							cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
							cv2.putText(img, label, (x, y), font, 1, color, 2)
							object_in_frame += 1
				self.coordinates = obj_points
				# print("[INFO] Object in frame : {}".format(object_in_frame))
				cv2.imshow('object-detection', img)
			else:
				print("No image for object-detection")

			cv2.waitKey(1)