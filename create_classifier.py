import json
from ibm_watson import VisualRecognitionV3

# autenticación mediante API_KEY
visual_recognition = VisualRecognitionV3(
        '2018-03-19',
        iam_apikey='{SERVICE_API_KEY}')

#Creación del modelo, se debe sustituir {model_name} por el nombre que desee asignarle al modelo, al
#igual que hicimos en la interface, asiganmos a una clase positiva los ejemplos con ojos cerrados y
#como negativa los ejemplos con ojos abiertos

with open('./abiertos.zip', 'rb') as abiertos, open('./cerrados.zip', 'rb') as cerrados:
    model = visual_recognition.create_classifier('{model_name}',
        positive_examples={'cerrados': cerrados},
        negative_examples=abiertos).get_result()  
    #obtener datos del modelo recien creado
    model_id = model['classifier_id']
    model_name = model['name']
    model_status = model['status']
    model_create_date = model['created']
    model_class = model['classes'][0]['class']

print("Datos del Modelo " + model_name + "recien creado.")
print("ID del modelo" + model_id)
print("Fecha de creación" + model_create_date)
print("Clases del modelo" + model_class)


#obtener el clasificador recien creado
classifier = visual_recognition.get_classifier(classifier_id=model_id).get_result()