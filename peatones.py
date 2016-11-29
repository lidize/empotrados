from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import datetime
import time
import imutils
import cv2
 
#contruccion de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i","--images", required=True, help="directorio de imagenes")
args = vars(ap.parse_args())
 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
 
for imagePath in paths.list_images(args["images"]):
    start = datetime.datetime.now()
    image = cv2.imread(imagePath)
    image =imutils.resize(image,width=min(400, image.shape[1]))
    orig = image.copy()
     
    (rects, weights) = hog.detectMultiScale(image, winStride=(1, 1),
        padding=(16, 16), scale=1.05)
  
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
  
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
  
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
  
    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))
    print("peatones encontrados: ", len(pick))
    # show the output images
    #cv2.imshow("Before NMS", orig)
    cv2.imshow("Deteccion de peatones", image)
    print("[tiempo] : {}s".format(
        (datetime.datetime.now()- start).total_seconds()))
    eli = datetime.datetime.now()
    hora = time.strftime("%X")
    print ("Fecha: %s / %s / %s "%(eli.day, eli.month, eli.year))
    print("Hora: "+ time.strftime("%X"))
    
    #inicio de escritura de archivos
    archivo = open("info.txt", 'a')
    archivo.write("Imagen:    " + filename + "   ")
    archivo.write("Peatones:  " + str(len(pick)) + "   ")
    archivo.write("Fecha:     " + str(eli.day)+" |"+ str(eli.month)+ " |" + str(eli.year)+"  ")
    archivo.write("Hora:  " + str(hora)+"\n\n")
    cv2.waitKey(0)
