#!/usr/bin/env python3
"""
A Flask server implementing face recognition operations using DeepFace:
- /verify: verify two faces match.
- /identify: identify faces in gallery.
- /enroll: register a face in the gallery.
- /clear: clear registered faces.
- /pad: presentation attack detection.
- /info: get system information.
"""

from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import uuid
import time
import base64
import io
import os
import tempfile
import shutil
from PIL import Image
import cv2


# DeepFaceHandler class encapsulates DeepFace configuration and calls
class DeepFaceHandler:
    def __init__(self, model_name="ArcFace", detector_backend="opencv", distance_metric="cosine"):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.default_threshold = 0.68

    def get_deepface_threshold(self):
        """Get the appropriate threshold for the current model and distance metric."""
        thresholds = {
            'VGG-Face': {'cosine': 0.68, 'euclidean': 0.60, 'euclidean_l2': 0.86},
            'Facenet': {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
            'OpenFace': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
            'DeepFace': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
            'DeepID': {'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17},
            'ArcFace': {'cosine': 0.68, 'euclidean': 4.15, 'euclidean_l2': 1.13},
            'Dlib': {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6},
            'SFace': {'cosine': 0.593, 'euclidean': 10.734, 'euclidean_l2': 1.055}
        }
        return thresholds.get(self.model_name, {}).get(self.distance_metric, self.default_threshold)

    def verify(self, img1_path, img2_path, enforce_detection=True):
        return DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            distance_metric=self.distance_metric,
            enforce_detection=enforce_detection
        )

    def represent(self, img_path, enforce_detection=True):
        return DeepFace.represent(
            img_path=img_path,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=enforce_detection
        )

    def extract_faces(self, img_path, enforce_detection=True, anti_spoofing=False):
        #print(help(DeepFace.extract_faces))  # Debugging line to show available parameters
        return DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=self.detector_backend,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing
        )

    def find(self, img_path, db_path, top_k=100, enforce_detection=False):
        # Returns a pandas DataFrame of matches
        return DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            distance_metric=self.distance_metric,
            enforce_detection=enforce_detection,
            silent=True,
            # top_k supported in some DeepFace versions, fallback handled below
            #top_k=top_k
        )

app = Flask(__name__)

# Instantiate DeepFaceHandler with desired settings
deepface_handler = DeepFaceHandler(
    model_name="ArcFace",  # Change to desired model
    detector_backend="opencv",
    distance_metric="cosine"
)

# Gallery storage
gallery_dir = None
known_face_ids = []
known_face_paths = []

def setup_gallery():
    """Setup temporary directory for gallery images."""
    global gallery_dir
    if gallery_dir is None:
        gallery_dir = tempfile.mkdtemp(prefix="deepface_gallery_")
        print(f"Gallery directory created: {gallery_dir}")

def cleanup_gallery():
    """Clean up temporary gallery directory."""
    global gallery_dir
    if gallery_dir and os.path.exists(gallery_dir):
        shutil.rmtree(gallery_dir)
        gallery_dir = None
        print("Gallery directory cleaned up")

def decode_base64_image(base64_string):
    """Decode base64 string to image and save to temporary file."""
    try:
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image.save(temp_file.name, 'JPEG')
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        print(f"ERROR decoding base64 image: {str(e)}")
        return None

def safe_remove_file(filepath):
    """Safely remove temporary file."""
    try:
        if filepath and os.path.exists(filepath):
            os.unlink(filepath)
    except Exception as e:
        print(f"Warning: Could not remove temporary file {filepath}: {str(e)}")


@app.route('/verify', methods=['POST'])
def verify():
    start_time = time.time()
    img1_path = None
    img2_path = None
    
    try:
        data = request.json or {}

        if 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Missing image1 or image2 in request'}), 400

        print(f"DEBUG: Received image1 length: {len(data['image1']) if data['image1'] else 0}")
        print(f"DEBUG: Received image2 length: {len(data['image2']) if data['image2'] else 0}")

        # Decode images to temporary files
        img1_path = decode_base64_image(data['image1'])
        img2_path = decode_base64_image(data['image2'])

        if img1_path is None:
            return jsonify({'error': 'Failed to decode image1'}), 400
        if img2_path is None:
            return jsonify({'error': 'Failed to decode image2'}), 400

        # Perform verification using DeepFaceHandler
        result = deepface_handler.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            enforce_detection=True
        )

        # Extract results
        distance = result['distance']
        threshold = result['threshold']
        decision = result['verified']

        # Convert distance to similarity score (1 - normalized distance)
        if deepface_handler.distance_metric == 'cosine':
            score = 1.0 - distance
        else:
            # For euclidean distances, normalize by threshold
            #score = max(0.0, 1.0 - (distance / threshold))
            score = distance


        processing_time = int((time.time() - start_time) * 1000)

        return jsonify({
            'score': float(score),
            'decision': bool(decision),
            'processing_time_ms': processing_time,
            'distance': float(distance),
            'threshold': float(threshold)
        })

    except Exception as e:
        print(f"ERROR in verify: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    finally:
        # Clean up temporary files
        safe_remove_file(img1_path)
        safe_remove_file(img2_path)

@app.route('/identify', methods=['POST'])
def identify():
    start_time = time.time()
    query_img_path = None
    
    try:
        data = request.json or {}

        if 'image' not in data:
            return jsonify({'error': 'Missing image in request'}), 400

        top_k = data.get('top_k', 100)

        # If gallery is empty, return empty matches
        if len(known_face_ids) == 0 or gallery_dir is None:
            return jsonify({
                "matches": [],
                "processing_time_ms": int((time.time() - start_time) * 1000)
            })

        # Decode query image
        query_img_path = decode_base64_image(data['image'])
        if query_img_path is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Use DeepFaceHandler.find for identification
        try:
            df_list = deepface_handler.find(
                img_path=query_img_path,
                db_path=gallery_dir,
                top_k=top_k,
                enforce_detection=False
            )
            # DeepFace.find returns a list of DataFrames (one for each model, but we use only one)
            if isinstance(df_list, list):
                df = df_list[0] if df_list else None
            else:
                df = df_list
            matches = []
            if df is not None and hasattr(df, "iterrows"):
                # The identity is the filename (path) of the match, so we map back to template_id
                for _, row in df.iterrows():
                    face_path = row.get("identity") or row.get("Identity") or row.get("img_path")
                    # Map face_path to template_id
                    template_id = None
                    for tid, path in zip(known_face_ids, known_face_paths):
                        if os.path.abspath(face_path) == os.path.abspath(path):
                            template_id = tid
                            break
                    # If not found, use filename without extension as fallback
                    if template_id is None and face_path:
                        template_id = os.path.splitext(os.path.basename(face_path))[0]
                    # Distance and score
                    distance = row.get("distance", row.get("Distance", 0.0))
                    threshold = deepface_handler.get_deepface_threshold()
                    if deepface_handler.distance_metric == 'cosine':
                        score = 1.0 - distance
                    else:
                        score = max(0.0, 1.0 - (distance / threshold))
                    matches.append({
                        'template_id': template_id,
                        'score': float(score),
                        'distance': float(distance)
                    })
            # Sort by score (highest first) and limit to top_k
            matches.sort(key=lambda x: x['score'], reverse=True)
            matches = matches[:top_k]
        except Exception as e:
            print(f"ERROR in DeepFaceHandler.find: {str(e)}")
            matches = []

        processing_time = int((time.time() - start_time) * 1000)

        return jsonify({
            'matches': matches,
            'processing_time_ms': processing_time
        })

    except Exception as e:
        print(f"ERROR in identify: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    finally:
        # Clean up temporary query image
        safe_remove_file(query_img_path)

@app.route('/enroll', methods=['POST'])
def enroll():
    start_time = time.time()
    temp_img_path = None
    
    try:
        setup_gallery()

        data = request.json or {}

        if 'image' not in data:
            return jsonify({'error': 'Missing image in request'}), 400

        # Generate new template ID
        template_id = str(uuid.uuid4())

        # Decode image to temporary file
        temp_img_path = decode_base64_image(data['image'])
        if temp_img_path is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Verify face can be detected
        try:
            # Try to extract face representation to validate the image
            deepface_handler.represent(
                img_path=temp_img_path,
                enforce_detection=True
            )
        except Exception as e:
            return jsonify({'error': f'No face detected in image: {str(e)}'}), 400

        # Save to gallery
        gallery_img_path = os.path.join(gallery_dir, f"{template_id}.jpg")
        shutil.copy2(temp_img_path, gallery_img_path)

        # Add to tracking lists
        known_face_ids.append(template_id)
        known_face_paths.append(gallery_img_path)

        processing_time = int((time.time() - start_time) * 1000)

        return jsonify({
            'template_id': template_id,
            'processing_time_ms': processing_time
        })

    except Exception as e:
        print(f"ERROR in enroll: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    finally:
        # Clean up temporary file
        safe_remove_file(temp_img_path)

@app.route('/clear', methods=['POST'])
def clear():
    start_time = time.time()
    
    try:
        # Clear tracking lists
        known_face_ids.clear()
        known_face_paths.clear()
        
        # Clean up gallery directory
        cleanup_gallery()
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'message': 'Template gallery cleared',
            'processing_time_ms': processing_time
        })
        
    except Exception as e:
        print(f"ERROR in clear: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/pad', methods=['POST'])
def pad():
    """Presentation Attack Detection - Basic implementation using DeepFace's anti-spoofing."""
    start_time = time.time()
    img_path = None
    
    try:
        data = request.json or {}

        if 'image' not in data:
            return jsonify({'error': 'Missing image in request'}), 400

        # Decode image
        img_path = decode_base64_image(data['image'])
        if img_path is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Liveness check using DeepFace anti_spoofing
        try:
            # Try to detect face(s) with anti_spoofing
            face_objs = deepface_handler.extract_faces(
                img_path=img_path,
                enforce_detection=True,
                anti_spoofing=True
            )
            if not isinstance(face_objs, list) or len(face_objs) == 0:
                return jsonify({'error': 'No face detected in image'}), 400
            # Check if all detected faces are real
            all_real = all(face_obj.get("is_real", False) for face_obj in face_objs)
            # Confidence: average of all faces' "confidence" field if present, else 1.0 if all real else 0.0
            confidences = [face_obj.get("confidence", 1.0) for face_obj in face_objs]
            confidence = float(np.mean(confidences)) if confidences else (1.0 if all_real else 0.0)

            processing_time = int((time.time() - start_time) * 1000)

            return jsonify({
                'is_live': bool(all_real),
                'confidence': float(confidence),
                'processing_time_ms': processing_time,
                'reason': 'DeepFace anti_spoofing result. For production use, consider specialized PAD models.'
            })
        except Exception as e:
            return jsonify({'error': f'PAD processing failed: {str(e)}'}), 400

    except Exception as e:
        print(f"ERROR in pad: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    finally:
        safe_remove_file(img_path)

@app.route('/info', methods=['GET'])
def info():
    try:
        # Get DeepFace version
        import deepface
        version = getattr(deepface, '__version__', 'unknown')

        return jsonify({
            'company': 'DeepFace by serengil',
            'version': version,
            'product_name': 'DeepFace',
            'description': 'A lightweight face recognition and facial attribute analysis framework',
            'model': deepface_handler.model_name,
            'detector_backend': deepface_handler.detector_backend,
            'distance_metric': deepface_handler.distance_metric,
            'thresholds': {
                'verify': deepface_handler.get_deepface_threshold(),
                'identify': deepface_handler.get_deepface_threshold()
            },
            'gallery_size': len(known_face_ids),
            'end_points': {
                'verify': '/verify',
                'identify': '/identify',
                'enroll': '/enroll',
                'clear': '/clear',
                'pad': '/pad',
                'info': '/info',
                'quit': '/quit'
            }
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get system info: {str(e)}'}), 500

@app.route('/quit', methods=['GET'])
def quit_server():
    """Endpoint to gracefully shut down the server."""
    try:
        # Clean up gallery before shutdown
        cleanup_gallery()
        
        import os
        import signal
        import threading
        
        def shutdown_server():
            print("Shutting down server...")
            os.kill(os.getpid(), signal.SIGINT)
        
        # Schedule shutdown
        threading.Timer(0.1, shutdown_server).start()
        
        print("Server shutdown initiated.")
        return jsonify({
            'message': 'Server is shutting down...'
        })
    except Exception as e:
        return jsonify({'error': f'Shutdown failed: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        setup_gallery()
        print(f"Starting DeepFace server with model: {deepface_handler.model_name}")
        print(f"Detector backend: {deepface_handler.detector_backend}")
        print(f"Distance metric: {deepface_handler.distance_metric}")
        app.run(host='0.0.0.0', port=5001, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup_gallery()
