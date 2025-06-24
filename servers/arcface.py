#!/usr/bin/env python3
"""
A Flask server implementing ArcFace face recognition operations:
- /verify: verify two faces match.
- /identify: identify faces in gallery.
- /enroll: register a face in the gallery.
- /clear: clear registered faces.
- /pad: presentation attack detection.
- /info: get system information.
"""

from flask import Flask, request, jsonify
import numpy as np
import uuid
import time
import base64
import io
from PIL import Image
import cv2
import onnxruntime as ort
import sklearn.metrics.pairwise
import os

app = Flask(__name__)

# Global variables for face gallery
known_face_encodings = []
known_face_ids = []
DEFAULT_TOLERANCE = 0.4  # ArcFace typically uses lower threshold due to higher accuracy

# ArcFace model session (will be initialized on first use)
arcface_session = None
MODEL_PATH = "arcfaceresnet100-11-int8.onnx"  # Path to ArcFace ONNX model

def initialize_arcface_model():
    """Initialize ArcFace ONNX model session."""
    global arcface_session
    if arcface_session is None:
        try:
            if not os.path.exists(MODEL_PATH):
                print(f"WARNING: ArcFace model not found at {MODEL_PATH}")
                print("Please download the ArcFace ONNX model and place it in the same directory")
                print("You can download it from: https://github.com/onnx/models/blob/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-11-int8.onnx")
                return False
            
            # Initialize ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                # Use GPU if available
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            arcface_session = ort.InferenceSession(MODEL_PATH, providers=providers)
            print(f"ArcFace model loaded successfully with providers: {providers}")
            
            # Print model input/output info for debugging
            print("Model inputs:")
            for inp in arcface_session.get_inputs():
                print(f"  - {inp.name}: {inp.shape} ({inp.type})")
            print("Model outputs:")
            for out in arcface_session.get_outputs():
                print(f"  - {out.name}: {out.shape} ({out.type})")
            
            return True
        except Exception as e:
            print(f"ERROR loading ArcFace model: {str(e)}")
            return False
    return True

def decode_base64_image(base64_string):
    """Decode base64 string to image array."""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array (RGB format)
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"ERROR decoding base64 image: {str(e)}")
        return None

def detect_and_align_face(image_array):
    """Detect and align face using OpenCV's DNN face detector."""
    try:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        
        # Check if face detector files exist
        model_file = 'opencv_face_detector_uint8.pb'
        config_file = 'opencv_face_detector.pbtxt'
        
        if not (os.path.exists(model_file) and os.path.exists(config_file)):
            print(f"WARNING: Face detector files not found. Using entire image.")
            print(f"Download from: https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830")
            return image_bgr
        
        # Load face detector
        net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), [104, 117, 123])
        net.setInput(blob)
        detections = net.forward()
        
        # Find the face with highest confidence
        best_face = None
        best_confidence = 0
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7 and confidence > best_confidence:  # Increased threshold
                best_confidence = confidence
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Ensure valid coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:  # Valid face region
                    best_face = (x1, y1, x2, y2)
        
        if best_face is None:
            print("No face detected with high confidence, using center crop")
            # Use center crop as fallback
            size = min(h, w)
            y_start = (h - size) // 2
            x_start = (w - size) // 2
            face_img = image_bgr[y_start:y_start+size, x_start:x_start+size]
        else:
            # Extract face region with some padding
            x1, y1, x2, y2 = best_face
            # Add 20% padding
            face_w, face_h = x2 - x1, y2 - y1
            pad_w, pad_h = int(face_w * 0.2), int(face_h * 0.2)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            face_img = image_bgr[y1:y2, x1:x2]
            print(f"Face detected with confidence {best_confidence:.3f}")
        
        return face_img
        
    except Exception as e:
        print(f"ERROR in face detection: {str(e)}")
        # Fallback: return center crop
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        return image_bgr[y_start:y_start+size, x_start:x_start+size]

def preprocess_face(image_array, target_size=(112, 112)):
    """Preprocess face image for ArcFace model."""
    try:
        # Detect and extract face
        face_img = detect_and_align_face(image_array)
        
        # Resize to target size
        face_resized = cv2.resize(face_img, target_size, interpolation=cv2.INTER_LINEAR)
        print(f"Face resized to: {face_resized.shape}")
        
        # Convert to float32 and normalize to [-1, 1]
        face_normalized = face_resized.astype(np.float32)
        face_normalized = (face_normalized - 127.5) / 127.5
        
        # Change from HWC to CHW format (channels first)
        face_transposed = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        face_batch = np.expand_dims(face_transposed, axis=0)
        
        print(f"Preprocessed face shape: {face_batch.shape}")
        print(f"Preprocessed face range: [{face_batch.min():.3f}, {face_batch.max():.3f}]")
        
        return face_batch
        
    except Exception as e:
        print(f"ERROR preprocessing face: {str(e)}")
        return None

def get_face_embedding(base64_string):
    """Get ArcFace embedding from base64 image string."""
    try:
        if not initialize_arcface_model():
            return None
        
        image_array = decode_base64_image(base64_string)
        if image_array is None:
            return None
        
        print(f"Input image shape: {image_array.shape}")
        
        # Preprocess the image
        preprocessed_face = preprocess_face(image_array)
        if preprocessed_face is None:
            return None
        
        # Run inference
        input_name = arcface_session.get_inputs()[0].name
        output = arcface_session.run(None, {input_name: preprocessed_face})
        
        # Get the embedding (usually the first output)
        embedding = output[0][0]  # Remove batch dimension
        print(f"Raw embedding shape: {embedding.shape}")
        print(f"Raw embedding norm: {np.linalg.norm(embedding):.6f}")
        
        # Check for NaN or infinity values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print("WARNING: NaN or infinity values in embedding!")
            return None
        
        # L2 normalize the embedding
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm == 0:
            print("WARNING: Zero norm embedding!")
            return None
            
        embedding = embedding / embedding_norm
        print(f"Normalized embedding norm: {np.linalg.norm(embedding):.6f}")
        print(f"Embedding stats: mean={embedding.mean():.6f}, std={embedding.std():.6f}")
        
        return embedding
        
    except Exception as e:
        print(f"ERROR getting face embedding: {str(e)}")
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    try:
        # Ensure embeddings are properly normalized
        emb1_norm = np.linalg.norm(embedding1)
        emb2_norm = np.linalg.norm(embedding2)
        
        print(f"Embedding norms: {emb1_norm:.6f}, {emb2_norm:.6f}")
        
        if emb1_norm == 0 or emb2_norm == 0:
            print("WARNING: Zero norm embedding in similarity calculation!")
            return 0.0
        
        # Calculate dot product (since embeddings should be normalized)
        dot_product = np.dot(embedding1, embedding2)
        
        # Alternative: use sklearn cosine similarity
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        sklearn_similarity = sklearn.metrics.pairwise.cosine_similarity(emb1, emb2)[0][0]
        
        print(f"Dot product similarity: {dot_product:.6f}")
        print(f"Sklearn similarity: {sklearn_similarity:.6f}")
        
        # Use dot product for normalized embeddings
        return float(dot_product)
        
    except Exception as e:
        print(f"ERROR calculating similarity: {str(e)}")
        return 0.0

@app.route('/verify', methods=['POST'])
def verify():
    start_time = time.time()
    
    try:
        data = request.json or {}
        
        if 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Missing image1 or image2 in request'}), 400
        
        print(f"DEBUG: Received image1 length: {len(data['image1']) if data['image1'] else 0}")
        print(f"DEBUG: Received image2 length: {len(data['image2']) if data['image2'] else 0}")
        
        print("=== Processing Image 1 ===")
        emb1 = get_face_embedding(data['image1'])
        print("=== Processing Image 2 ===")
        emb2 = get_face_embedding(data['image2'])
        
        if emb1 is None:
            return jsonify({'error': 'No face detected in image1 or model not loaded'}), 400
        if emb2 is None:
            return jsonify({'error': 'No face detected in image2 or model not loaded'}), 400
        
        # Calculate similarity score
        print("=== Calculating Similarity ===")
        score = calculate_similarity(emb1, emb2)
        decision = bool(score >= DEFAULT_TOLERANCE)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        print(f"=== FINAL RESULT ===")
        print(f"Similarity score: {score:.6f}")
        print(f"Threshold: {DEFAULT_TOLERANCE}")
        print(f"Decision: {decision}")
        print(f"Processing time: {processing_time}ms")
        
        return jsonify({
            'score': score,
            'decision': decision,
            'processing_time_ms': processing_time,
            'threshold': DEFAULT_TOLERANCE,
            'debug_info': {
                'embedding1_norm': float(np.linalg.norm(emb1)),
                'embedding2_norm': float(np.linalg.norm(emb2)),
                'embedding_dimension': len(emb1)
            }
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
    
    query_embedding = get_face_embedding(data['image'])
    if query_embedding is None:
        return jsonify({'error': 'No face detected in image or model not loaded'}), 400
    
    # Calculate similarities to all known faces
    matches = []
    for i, known_embedding in enumerate(known_face_encodings):
        score = calculate_similarity(query_embedding, known_embedding)
        matches.append({
            'template_id': known_face_ids[i],
            'score': score
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
    
    embedding = get_face_embedding(data['image'])
    if embedding is None:
        return jsonify({'error': 'No face detected in image or model not loaded'}), 400
    
    # Add to gallery
    known_face_encodings.append(embedding)
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

@app.route('/info', methods=['GET'])
def info():
    model_loaded = arcface_session is not None
    
    return jsonify({
        'company': 'ArcFace ONNX Implementation',
        'version': '1.0.0',
        'product_name': 'ArcFace Face Recognition Server',
        'description': 'A Flask API for ArcFace-based face recognition operations.',
        'model_loaded': model_loaded,
        'model_path': MODEL_PATH,
        'thresholds': {
            'verification': DEFAULT_TOLERANCE,
            'identification': DEFAULT_TOLERANCE
        },
        'gallery_size': len(known_face_encodings),
        'end_points': {
            'verify': '/verify',
            'identify': '/identify',
            'enroll': '/enroll',
            'clear': '/clear',
            'info': '/info',
            'quit': '/quit'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    model_status = "loaded" if arcface_session is not None else "not_loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'gallery_size': len(known_face_encodings),
        'timestamp': int(time.time())
    })

@app.route('/quit', methods=['GET'])
def quit_server():
    """Endpoint to gracefully shut down the server."""
    import os
    import signal
    
    def shutdown_server():
        print("Shutting down ArcFace server...")
        os.kill(os.getpid(), signal.SIGINT)
    
    # Schedule shutdown
    import threading
    threading.Timer(0.1, shutdown_server).start()
    
    print("ArcFace server shutdown initiated.")
    return jsonify({
        'message': 'ArcFace server is shutting down...'
    })

if __name__ == '__main__':
    print("Starting ArcFace Face Recognition Server...")
    print(f"Looking for ArcFace model at: {MODEL_PATH}")
    
    # Try to initialize the model on startup
    if initialize_arcface_model():
        print("ArcFace model loaded successfully!")
    else:
        print("WARNING: ArcFace model not loaded. Server will start but face recognition will not work.")
        print("Please ensure the ArcFace ONNX model is available.")
    
    app.run(host='0.0.0.0', port=5001, debug=True)