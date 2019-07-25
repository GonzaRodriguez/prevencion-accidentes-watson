import json
import cv2
import os
from ibm_watson import VisualRecognitionV3
import shutil
import zipfile
from os import remove
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

CONFIDENCE = 0.6
id_class= "{MODEL_ID}"

# autenticación mediante API_KEY
visual_recognition = VisualRecognitionV3(
        '2018-03-19',
        iam_apikey='{SERVICE_API_KEY}')

        
image_test_path = '{PATH_TO_THE_TEST_IMAGE}}'
image_test_name = '{TEST_IMAGE_NAME}'

#Utilizo el servicio detectfaces para identificar las caras en la imagen
with open(image_test_path, 'rb') as images_file:
    faces = visual_recognition.detect_faces(images_file).get_result()

elementos = faces['images'][0]['faces']
face_location = []
for item in elementos:
    face_location.append(item['face_location'])

#carpeta donde se guardan los recortes
path = 'recortes'
img = cv2.imread(image_test_path)
        
if (len(face_location) > 1):
            
    for i in range(len(face_location)):
        x = face_location[i]['left']
        y = face_location[i]['top']
        w = face_location[i]['width']
        h = face_location[i]['height']
        crop_img = img[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(path , str(i) + '.png'), crop_img)
                
        #llamada al clasificador con el recorte
        with open('recortes/' + str(i) + '.png', 'rb') as images_file:
            classes = visual_recognition.classify(
                images_file,
                threshold='0.1',
                classifier_ids=[id_class]).get_result()
                #print(json.dumps(classes, indent=2))

        #Obtengo la clase que mejor clasifico a el ejemplo
        clases = classes['images'][0]['classifiers'][0]['classes']
        if (len(clases) > 0):     
            class_name = []
            class_socre = []
            for item in clases:
                class_name.append(item['class'])
                class_socre.append(item['score'])
                
            max_score = 0
            indice = -1
            for idx,item in enumerate(class_socre):
                if item > max_score:
                    max_score = item
                    indice = idx
                
            if max_score > CONFIDENCE:
                nombre_zip = 'more_' + class_name[indice] + '.zip'
                    
                #genero el zip con la nueva imagen para actualizar el modelo
                zf = zipfile.ZipFile(nombre_zip, mode="w")
                try:
                    zf.write(image_test_path, compress_type=compression)
                finally:
                    zf.close()
                    
                #actualizo el modelo
                nombre = class_name[indice]
                with open('./' + nombre_zip, 'rb') as nombre:
                    updated_model = visual_recognition.update_classifier(
                        classifier_id=id_class,
                        positive_examples={'more_positive_examples': nombre}).get_result()
                    #print(json.dumps(updated_model, indent=2))
                remove(nombre_zip)
else: 
    if (len(face_location) > 0):
        cv2.imwrite(os.path.join(path , '1.png'), img)  
        with open('recortes/' + '1.png', 'rb') as images_file:
            classes = visual_recognition.classify(
                images_file,
                threshold='0.1',
                classifier_ids=[id_class]).get_result()
            #print(json.dumps(classes, indent=2))
            
        clases = classes['images'][0]['classifiers'][0]['classes']
        if (len(clases) > 0):    
            class_name = []
            class_socre = []
            for item in clases:
                class_name.append(item['class'])
                class_socre.append(item['score'])
                    
            max_score = 0
            indice = -1
            for idx,item in enumerate(class_socre):
                if item > max_score:
                    max_score = item
                    indice = idx
                    
            if max_score > CONFIDENCE:
                nombre_zip = 'more_' + class_name[indice] + '.zip'
                    
                #genero el zip con la nueva imagen para actualizar el modelo
                zf = zipfile.ZipFile(nombre_zip, mode="w")
                try:
                    zf.write(image_test_path, compress_type=compression)
                finally:
                    zf.close()
                        
                #actualizo el modelo
                nombre = class_name[indice]
                with open('./' + nombre_zip, 'rb') as nombre:
                    updated_model = visual_recognition.update_classifier(
                        classifier_id=id_class,
                        positive_examples={'more_positive_examples': nombre}).get_result()
                    #print(json.dumps(updated_model, indent=2))
                remove(nombre_zip) 
                

