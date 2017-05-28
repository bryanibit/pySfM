from opensfm import dataset
from opensfm import exif
import os.path,sys
import argparse
import logging
import time

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
start=time.time()

parse=argparse.ArgumentParser(description="Extract image exif Info")
parse.add_argument('dataset')
args=parse.parse_args()
data=dataset.DataSet(args.dataset)

camera_models={}
camera_models_overrides={}

for images in data.images():
  logging.info('Extract focal camera for image {}'.format(images))
  d=exif.extract_exif_from_file(data.load_image(images))
  if not data.config['use_exif_size']:
    d['height'], d['width']=data.image_as_array(images).shape[:2]
  data.save_exif(images,d)
  if not d['camera'] in camera_models:
    calib=(exif.hard_coded_calibration(d) or exif.focal_ratio_calibration(d) or exif.default_calibration(data))
    camera_models[d['camera']]={'width':d['width'],'height':d["height"],'focal':calib['focal'],'k1':calib['k1'],'k2':calib['k2'],'projection_type': d.get('projection_type', 'perspective')}
  if data.camera_models_overrides_exists():
    camera_models_overrides=data.load_camera_models_overrides()
  for key, value in camera_models_overrides.items():
    camera_models[key]=value
data.save_camera_models(camera_models)

end=time.time()
with open(data.profile_log(),'a') as fin:
  fin.write('focal_from_exif:{0}\n'.format(end-start))
  

