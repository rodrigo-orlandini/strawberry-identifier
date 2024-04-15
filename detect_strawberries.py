import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from load_model import detect_fn

IMAGES_NAMES = ["true-01", "true-02", 'true-03', 'true-04', 'true-05', "false-01", "false-02", 'false-03', 'false-04', 'false-05']

for IMAGE_NAME in IMAGES_NAMES: 
	IMAGE_PATH = os.path.join("images", "evaluation", f"{IMAGE_NAME}.jpg")

	category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt")

	image = cv2.imread(IMAGE_PATH)
	image_np = np.array(image)

	input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
	detections = detect_fn(input_tensor)

	num_detections = int(detections.pop('num_detections'))
	detections = {
		key: value[0, :num_detections].numpy() for key, value in detections.items()
	}

	detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

	strawberry_indices = np.where(detections['detection_classes'] == 0)[0]
	strawberry_boxes = detections['detection_boxes'][strawberry_indices]
	strawberry_scores = detections['detection_scores'][strawberry_indices]

	label_id_offset = 1
	image_np_with_detections = image_np.copy()

	viz_utils.visualize_boxes_and_labels_on_image_array(
		image_np_with_detections,
		strawberry_boxes,
		detections['detection_classes'][strawberry_indices] + label_id_offset,
		strawberry_scores,
		category_index,
		use_normalized_coordinates=True,
		max_boxes_to_draw=5,
		min_score_thresh=0.5,
		agnostic_mode=False
	)

	# final_image = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
	cv2.imwrite(os.path.join("images", "evaluation", f"{IMAGE_NAME}-detected.png"), image_np_with_detections)
