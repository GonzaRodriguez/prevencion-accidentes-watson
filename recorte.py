import json
import cv2
import os
import random
from ibm_watson import VisualRecognitionV3


# autenticación mediante API_KEY
visual_recognition = VisualRecognitionV3(
        '2018-03-19',
        iam_apikey='5BrAiurAP_B0MGDLLdDxepa0slfF6GmSpYjvvY1m5m_G')

        
 
path = 'test'
for archivo in os.listdir(path):

    #Utilizo el servicio detectfaces para identificar las caras en la imagen
    with open('./test/' + archivo, 'rb') as images_file:
        faces = visual_recognition.detect_faces(images_file).get_result()

    elementos = faces['images'][0]['faces']
    face_location = []
    for item in elementos:
        face_location.append(item['face_location'])

    #carpeta donde se guardan los recortes
    path = 'cut_test'
    img = cv2.imread('./test/' + archivo)
            
    if (len(face_location) >= 1):
                
        for i in range(len(face_location)):
            x = face_location[i]['left']
            y = face_location[i]['top']
            w = face_location[i]['width']
            h = face_location[i]['height']
            crop_img = img[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(path , str(random.randint(0,1000)) + '.png'), crop_img)