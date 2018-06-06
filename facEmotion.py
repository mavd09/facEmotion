import math, sys, io, numpy, time
import face_recognition, cv2
import datetime, os

import facEmotionCNN

from threading import Thread
import threading
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QBrush, QPen, QFont
from PyQt5.QtCore import QTimer, QBuffer, Qt
from PyQt5.uic import loadUi

emotionCategories = [ 'Angry', 'Neutral', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'CNN working', "CNN not sure" ]

class FacEmotion_App(QDialog):

  APP_NAME = 'facEmotion'
  WIDTH_CAM = 640
  HEIGHT_CAM = 480

  IS_GET_FACES_RUNNING = False

  PERSONAL_PHOTOS_FOLDER = 'personal_photos'

  def __init__(self):
    super(FacEmotion_App,self).__init__()
    loadUi('facEmotion.ui',self)

    self.bboxes_lock = threading.Lock()
    self.image = None
    self.photo = None
    self.bboxes = []
    self.bboxes_labels = []
    self.takePhotoButton.clicked.connect(self.take_photo)
    self.savePhotoButton.clicked.connect(self.save_photo)
    self.start_cam()

  def start_cam(self):
    self.setWindowTitle(self.APP_NAME)
    self.capture = cv2.VideoCapture(0)
    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT_CAM)
    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH_CAM)
    # create a timer to update the screen every 5 ms 
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.update_frame)
    self.timer.start(10)

  def update_frame(self):
    _, cur_image = self.capture.read()
    cur_image = cv2.flip(cur_image, 1)
    cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)
    self.image = QImage(cur_image.data, cur_image.shape[1], cur_image.shape[0], QImage.Format_RGB888)
    painter = QPainter(self.image)
    painter.setPen(Qt.red)
    painter.setFont(QFont('Arial',20))
    if(self.IS_GET_FACES_RUNNING == False):
      thread = Thread(target=self.get_faces, args=(self.image,))
      thread.start()
    self.bboxes_lock.acquire()
    for idx, (x0, y0, x1, y1) in enumerate(self.bboxes):
      painter.drawRect(min(x0,x1), min(y0,y1), abs(x0-x1), abs(y0-y1))
      painter.drawText(min(x0,x1)+5, max(y0,y1)-5, emotionCategories[self.bboxes_labels[idx]])
    painter.end()
    self.bboxes_lock.release()
    self.camLabel.setPixmap(QPixmap.fromImage(self.image))
    self.camLabel.setScaledContents(True)

  def take_photo(self):
    self.photo = self.image
    self.photoLabel.setPixmap(QPixmap.fromImage(self.photo))
    self.photoLabel.setScaledContents(True)

  def save_photo(self):
    if(self.photo == None): return 
    if not os.path.exists(self.PERSONAL_PHOTOS_FOLDER):
      os.makedirs(self.PERSONAL_PHOTOS_FOLDER)
    self.photo = self.q_image_to_pil_image(self.photo)
    filename = str(datetime.datetime.now().timestamp())
    self.photo.save("personal_photos/"+filename+".png")

  def q_image_to_pil_image(self, image):
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    image.save(buffer, "PNG")
    pil_image = Image.open(io.BytesIO(buffer.data()))
    return pil_image

  def get_faces(self, image):
    self.IS_GET_FACES_RUNNING = True
    pil_image = self.q_image_to_pil_image(image)
    face_locations = face_recognition.face_locations(numpy.asarray(pil_image))
    bboxes = []
    images = []
    for idx, face_location in enumerate(face_locations):  
      y1, x1, y0, x0 = face_location
      offset_x = abs(x1-x0) * 0.20
      offset_y = abs(y1-y0) * 0.25
      x0 = max(1,x0-offset_x)
      y0 = min(self.HEIGHT_CAM,y0+offset_y)
      x1 = min(self.WIDTH_CAM,x1+offset_x)
      y1 = max(1,y1-offset_y)
      bboxes.append((x0,y1,x1,y0))
      images.append(pil_image.crop((x0,y1,x1,y0)))
    self.bboxes_lock.acquire()
    self.bboxes = bboxes
    self.bboxes_labels = facEmotionCNN.get_emotions(bboxes, images)
    self.bboxes_lock.release()
    self.IS_GET_FACES_RUNNING = False

def main():
  app = QApplication(sys.argv)
  window = FacEmotion_App()
  window.show()
  sys.exit(app.exec_())

main()