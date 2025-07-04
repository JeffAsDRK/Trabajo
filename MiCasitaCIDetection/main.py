import cv2
import torch
import boto3
import re
import numpy as np
from ultralytics import YOLO

def create_case_insensitive_exclude_words(words):
    # Automatically generate case-insensitive set from input words
    return {word.lower() for word in words}

def clean_text(text):
    return re.sub(r'\s+', '', re.sub(r'[^\w]', '', text)).strip()

def clean_array(text_array):
    # Limpiar cada elemento del array y eliminar los elementos vacíos
    cleaned_array = [clean_text(item) for item in text_array]
    return [item for item in cleaned_array if item]

EXCLUDE_WORDS = create_case_insensitive_exclude_words([
    'Apellidos', 'Nombres', 'Y', 'El', 'Los', 'Las', 
    'Un', 'Una', 'Unos', 'Unas', 'En', 'Con', 'Para', 'Por', "NOMBRENAME", "NUI",
    "NUMBER", "LICENCIA","no", "LUGARDE", "NACAL", "ACIMIENTO", "LUGAR","LUGARDENA",
    "NACIMIENTO","LUGARDENACIMIENTO","CIUDADANIA", "DANIA", "MIENTO", "IDAD", "NOMBREADS",
    "NOMBR", "NOM", "NOMBRS", "APERIDOS", "APLLIDOS", "APPLLIDOS", "MENSON", "given",
    "NACIONALIDAD", "nacional","lidad", "APELLIDOFAMILY", "NAME", "aaai", "ce", "VOI", "SOCITEMY",
    "CERTIFICADO", "VOTACIÓN", "VOTACION", "DIRECCION", "GENERAL", "REGISTRO", "NACIONALISAD", "adidas"
])

class YOLODocumentProcessor:
    def __init__(self, model_paths, rekognition_client=None):
        self.models = {}
        for name, path in model_paths.items():
            # Use Ultralytics YOLO for loading models
            self.models[name] = YOLO(path)
        
        self.rekognition = rekognition_client or boto3.client('rekognition')
    
    def detect_sections(self, image_path):
        image = cv2.imread(image_path)
        detected_sections = {}
        
        for model_name, model in self.models.items():
            # Perform inference using the loaded model
            results = model(image_path)
            
            # Process detection results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class and coordinates
                    cls = model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Focus on specific sections of interest
                    if cls in ['nombres', 'apellidos', 'numero_cedula']:
                        section = image[y1:y2, x1:x2]
                        
                        # Convert to base64 for Rekognition
                        _, buffer = cv2.imencode('.jpg', section)
                        image_bytes = buffer.tobytes()
                        
                        # Process with Rekognition
                        response = self.rekognition.detect_text(Image={'Bytes': image_bytes})
                        
                        # Extract text
                        texts = [
                            clean_text(text['DetectedText']) for text in response.get('TextDetections', [])
                            if text['Type'] == 'WORD' and clean_text(text['DetectedText']).lower() not in EXCLUDE_WORDS
                        ]
                        texts = clean_array(texts)
                        
                        if cls == 'numero_cedula':
                            texts = [re.sub(r'\D', '', t) for t in texts] 
                            texts = [t for t in texts if t] 
                        
                        detected_sections[cls] = {
                            'coordinates': (x1, y1, x2, y2),
                            'rekognition_text': texts,
                            'model': model_name
                        }
        
        return detected_sections

# Usage example
def main():
    # Initialize processor with multiple models
    processor = YOLODocumentProcessor({
        'yolo11n': 'yolo11n.pt',
        'best': 'best.pt'
    })
    
    # Process document
    results = processor.detect_sections('ImagenCedula_eQLQ1OJ.jpg')
    
    # Print results
    for section, data in results.items():
        print(f"{section} (from {data['model']}): {data}")

if __name__ == '__main__':
    main()