# add dataset_store to python load path
# you don't need this if you're using brewery
import sys, os
sys.path.insert(0, os.path.abspath(__file__ + '/../../sdk'))
from dataset_store import Dataset
from dataset_store.annotation_task import create_task
dataset = Dataset('2017-12-28-19-02-26')
#dataset = Dataset.open('2017-12-22-09-23-46/')
file_name = './image/cam1/'
frames = []
if os.path.exists(file_name):
    files = os.listdir(file_name)
    timestamp = 0
    for image in files:
        image_name = os.path.join(file_name,image)
        print image_name
        for ts, _, _ in dataset.fetch('/camera1/image_color/compressed', raw=True):
            timestamp = ts
            break
        frames.append({
            'dataset': dataset.name,
            'sensor': '/camera1/image_color/compressed',
            'ts':timestamp,
            'reserved': False,
            'loc': image_name})
print len(frames)


create_task(
    name='segmentation_caofeigang',
    type='segmentation',
    created_by='Hengchen Dai',
    description='caofeigang segmentation task ',
    frames=frames,
    tags=['segmentation' ],
    #notify='hengchen.dai@tusimple.com',
    #level=3,
    #reserved=0,
)
