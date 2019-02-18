import cv2
import os
import glob
img_dir = "/home/nikita/.virtualenvs/nikk/projects/face_recognition/dataset/pikachu"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
#data = []
for f1 in files:
    img = cv2.imread(f1)
    #data.append(img)
    if img is None:
        print("[INFO] deleting:")
        os.remove(f1)
        continue
