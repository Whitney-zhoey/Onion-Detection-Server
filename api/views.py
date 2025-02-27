# views.py
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
import torch
from PIL import Image
import io
import base64
from ultralytics import YOLO

# Load the YOLO model once when the server starts
model = YOLO('api/yolo_model/yolo11n.pt')


@api_view(['POST'])
def process_image(request):
    if 'image' not in request.FILES:
        return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Get the image from the request
        image_file = request.FILES['image']

        # Read image file
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Process the image with YOLO
        results = model(image)

        # Get the annotated image using Ultralytics' built-in plotting
        result_plotted = results[0].plot()

        # Convert numpy array to PIL Image
        result_image = Image.fromarray(result_plotted)

        # Convert to base64 for response
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Extract detection results - classes and confidence scores
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Extract confidence and class
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                detections.append({
                    'class_name': class_name,
                    'confidence': round(confidence * 100, 2)  # Convert to percentage and round to 2 decimal places
                })

        # Return the processed image and detection information
        return Response({
            'success': True,
            'processed_image': img_str,
            'detections': detections
        })

    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


