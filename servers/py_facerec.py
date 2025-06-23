#!/usr/bin/env python3
"""
A simple Flask server implementing face recognition operations:
- /verify: verify two faces match.
- /identify: identify faces in gallery.
- /enroll: register a face in the gallery.
- /clear: clear registered faces.
- /pad: presentation attack detection.
- /info: get system information.
"""

from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import uuid
import time
import base64
import io
from PIL import Image

app = Flask(__name__)

known_face_encodings = []
known_face_ids = []
DEFAULT_TOLERANCE = 0.6

def decode_base64_image(base64_string):
    """Decode base64 string to image array for face_recognition."""
    try:
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"ERROR decoding base64 image: {str(e)}")
        return None

def get_face_encoding_from_base64(base64_string):
    """Get face encoding from base64 image string."""
    try:
        image_array = decode_base64_image(base64_string)
        if image_array is None:
            return None
        
        encodings = face_recognition.face_encodings(image_array)
        if not encodings:
            print("No face encodings found in image")
            return None
        return encodings[0]
    except Exception as e:
        print(f"ERROR getting face encoding: {str(e)}")
        return None

@app.route('/verify', methods=['POST'])
def verify():
    start_time = time.time()
    
    try:
        data = request.json or {}
        
        if 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Missing image1 or image2 in request'}), 400
        
        print(f"DEBUG: Received image1 length: {len(data['image1']) if data['image1'] else 0}")
        print(f"DEBUG: Received image2 length: {len(data['image2']) if data['image2'] else 0}")
        
        enc1 = get_face_encoding_from_base64(data['image1'])
        enc2 = get_face_encoding_from_base64(data['image2'])
        
        if enc1 is None:
            return jsonify({'error': 'No face detected in image1'}), 400
        if enc2 is None:
            return jsonify({'error': 'No face detected in image2'}), 400
        
        distance = face_recognition.face_distance([enc1], enc2)[0]
        score = 1.0 - float(distance)  # Convert distance to similarity score
        decision = bool(distance <= DEFAULT_TOLERANCE)  # Convert numpy bool to Python bool
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'score': score,
            'decision': decision,
            'processing_time_ms': processing_time
        })
        
    except Exception as e:
        print(f"ERROR in verify: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/identify', methods=['POST'])
def identify():
    start_time = time.time()
    data = request.json or {}
    
    if 'image' not in data:
        return jsonify({'error': 'Missing image in request'}), 400
    
    top_k = data.get('top_k', 100)
    
    # If gallery is empty, return empty matches
    if len(known_face_encodings) == 0:
        return jsonify({
            "matches": [], 
            "processing_time_ms": int((time.time() - start_time) * 1000)
        })
    
    enc = get_face_encoding_from_base64(data['image'])
    if enc is None:
        return jsonify({'error': 'No face detected in image'}), 400
    
    # Calculate distances to all known faces
    distances = face_recognition.face_distance(known_face_encodings, enc)
    
    # Create matches with scores and template_ids
    matches = []
    for i, distance in enumerate(distances):
        score = 1.0 - distance  # Convert distance to similarity score
        matches.append({
            'template_id': known_face_ids[i],
            'score': float(score)
        })
    
    # Sort by score (highest first) and limit to top_k
    matches.sort(key=lambda x: x['score'], reverse=True)
    matches = matches[:top_k]
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return jsonify({
        'matches': matches,
        'processing_time_ms': processing_time
    })

@app.route('/enroll', methods=['POST'])
def enroll():
    start_time = time.time()
    data = request.json or {}
    
    if 'image' not in data:
        return jsonify({'error': 'Missing image in request'}), 400
    
    # Generate new template ID
    template_id = str(uuid.uuid4())
    
    enc = get_face_encoding_from_base64(data['image'])
    if enc is None:
        return jsonify({'error': 'No face detected in image'}), 400
    
    # Add to gallery
    known_face_encodings.append(enc)
    known_face_ids.append(template_id)
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return jsonify({
        'template_id': template_id,
        'processing_time_ms': processing_time
    })

@app.route('/clear', methods=['POST'])
def clear():
    start_time = time.time()
    known_face_encodings.clear()
    known_face_ids.clear()
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return jsonify({
        'message': 'Template gallery cleared',
        'processing_time_ms': processing_time
    })

@app.route('/pad', methods=['POST'])
def pad():
    """Presentation Attack Detection - mock implementation."""
    start_time = time.time()
    data = request.json or {}
    
    if 'image' not in data:
        return jsonify({'error': 'Missing image in request'}), 400
    
    # Mock PAD - always return live for this demo
    # In a real implementation, this would analyze the image for liveness
    processing_time = int((time.time() - start_time) * 1000)
    
    return jsonify({
        'is_live': True,
        'reason': 'Face appears to be live',
        'processing_time_ms': processing_time
    })

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'company': 'Face Recognition Inc.',
        'version': '1.0.0',
        'product_name': 'Face Recognition API',
        'description': 'A simple API for face recognition operations.',
        'thresholds': {
            'verification': DEFAULT_TOLERANCE,
            'identification': DEFAULT_TOLERANCE
        }
    })

@app.route('/quit', methods=['GET'])
def quit_server():
    """Endpoint to gracefully shut down the server."""
    import os
    import signal
    def shutdown_server():
        print("Shutting down server...")
        os.kill(os.getpid(), signal.SIGINT)
    # Schedule shutdown
    # wait for a moment to ensure the response is sent
    import threading
    threading.Timer(0.1, shutdown_server).start()
    # Return a response immediately
    print("Server shutdown initiated.")
    return jsonify({
        'message': 'Server is shutting down...'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)