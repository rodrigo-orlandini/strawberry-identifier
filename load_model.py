import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

# Load settings from pipeline and build a model
settings = config_util.get_configs_from_pipeline_file("models/mobilenet/pipeline.config")
detection_model = model_builder.build(model_config=settings['model'], is_training=False)

# Get latest generated checkpoint
checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
checkpoint.restore(os.path.join("models", "mobilenet", "ckpt-3")).expect_partial()

@tf.function
def detect_fn(image):
	image, shapes = detection_model.preprocess(image)
	prediction_dict = detection_model.predict(image, shapes)
	detections = detection_model.postprocess(prediction_dict, shapes)

	return detections