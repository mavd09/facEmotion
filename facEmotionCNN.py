print ( "Loading libraries for face recognition")
import math
import numpy
import threading
import hungarian
import time
import keras.models
import PIL
import os
print ( "Finished loading libraries for face recognition" )

CERTAINTY_THREASHOLD = 0.45

def synchronized(func):
    func.__lock__ = threading.Lock()
    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)
    return synced_func


class FaceBBox:
	def center(self):
		return numpy.array([self.sx+self.bx, self.sy+self.by])/2
	def __repr__(self):
		return ("{ (%d,%d)->(%d,%d) id=%d emotion=%d}" % (self.sx, self.sy, self.bx, self.by, self.id, self.emotion))	
	def __init__(self, coords):
		self.sx = min(coords[0],coords[2])
		self.bx = max(coords[0],coords[2])
		self.sy = min(coords[1],coords[3])
		self.by = max(coords[1],coords[3])
		self.id = -1
		self.emotion = 7

from keras.preprocessing.image import img_to_array as preprocess_image
from keras.applications.vgg16 import preprocess_input

def get_features(image): 
    imageBW = PIL.Image.new("L", image.size)
    imageBW.paste(image)
    imageRGB = PIL.Image.new("RGB", image.size)
    imageRGB.paste(imageBW)
    image = imageRGB.resize( (224, 224), PIL.Image.ANTIALIAS)
    features = preprocess_input(preprocess_image(image))
    return features

def feedforward(face_bboxes, images):
	global CERTAINTY_THREASHOLD
	feedforward.running = True
	if feedforward.model == None:
		print("Loading model")
		feedforward.model = keras.models.load_model('model_v3.h5')
		print("Model loaded :D")
	model_output = feedforward.model.predict(numpy.array([get_features(img) for img in images]))
	predictions = numpy.argmax(model_output, axis=1)
	for i in range(len(predictions)):
		if model_output[i][predictions[i]] < CERTAINTY_THREASHOLD:
			predictions[i] = 8
	feedforward.result = [ (face_bboxes[i].id,predictions[i]) for i in range(len(face_bboxes)) ]
	feedforward.running = False
feedforward.model = None
feedforward.running = False
feedforward.result = None

@synchronized
def get_emotions(bboxes, images):

	cur = get_emotions.current_faces
	m, n = len(cur), len(bboxes)
	new = [ FaceBBox(bboxes[i]) for i in range(n) ]
	dist_matrix = numpy.ones ( (m,n) )*1e10

	for i in range(m):
		for j in range(n):
			dist_matrix[i,j] = numpy.linalg.norm(cur[i].center()-new[j].center())

	solver = hungarian.Hungarian()
	solver.calculate(dist_matrix)
	
	cur_prime = [None for i in range(n)]
	for edge in solver.get_results():
		cur_prime[edge[1]] = cur[edge[0]]
	cur = get_emotions.current_faces = cur_prime
	for j in range(n):
		if cur[j] == None:
			new[j].id = get_emotions.next_face_id
			get_emotions.next_face_id += 1
			cur[j] = new[j]

	assert n  == len(cur), "Some bounding box was lost"

	if not feedforward.running:
		if feedforward.result != None:
			for result in feedforward.result:
				for face in cur:
					if face.id == result[0]:
						face.emotion = result[1]
			feedforward.result = None
		if len(images) != 0:
			thread = threading.Thread(target=feedforward, args=(cur,images))
			thread.start()

	return [ face.emotion for face in cur ]
get_emotions.next_face_id = 0
get_emotions.current_faces = []


def test1():
	def check_current_faces(wanted):
		cur = get_emotions.current_faces
		assert len(cur) == len(wanted), "Wanted len is %d but got %d" % ( len(wanted), len(cur) )
		for i in range(len(cur)):
			assert cur[i].id == wanted[i], "Faces are not what was expected: %s" % ( str(wanted) )

	imagesNames = [ "1.jpg", "2.jpg", "3.jpeg", "4.jpeg", "5.jpeg", "6.jpeg" ]
	images = [ PIL.Image.open(os.path.join("test_imgs",imagesNames[i])) for i in range(len(imagesNames))]
	get_emotions([(0,0,10,10), (40,40,50,50)], [images[0], images[1]])
	check_current_faces([0,1])
	get_emotions([(0,2,10,12), (42,40,52,50), (0,100,10,110)], [images[0],images[1],images[2]])
	check_current_faces([0,1,2])
	get_emotions([(0,90,10,100), (0,2,10,12)], [images[2], images[0]] )
	check_current_faces([2,0])
	get_emotions([(0,0,10,10), (2,2,12,12), (50,50,60,60), (2,90,12,102)], [images[0],images[3],images[4],images[2]])
	check_current_faces([0,3,4,2])
	get_emotions([(2,90,12,105), (0,0,10,10)], [images[2], images[0]])
	check_current_faces([2,0])
	print("Passed test1 :D")


test1()


