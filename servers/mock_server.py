import jinja2
import markupsafe
# Monkey-patch jinja2 to provide escape and Markup for Flask compatibility
jinja2.escape = markupsafe.escape
jinja2.Markup = markupsafe.Markup
from flask import Flask, request, jsonify
# Disable Jinja2 default extensions to avoid compatibility issues with Jinja2 v3
Flask.jinja_options['extensions'] = []
try:
    from flasgger import Swagger
except ImportError:
    Swagger = None
import base64
import uuid
import random
import time

# Algorithm information for /info endpoint
ALGO_INFO = {
    "company": "BixeLab",
    "product_name": "Mock Biometric Test Server",
    "version": "1.0.0",
    # Optional thresholds for decision points
    "thresholds": {
        "identify": 0.5,
        "verify": 0.75
    },
    # Optional description of the algorithm
    "description": "A mock biometric API for PoC and testing, no real biometric algorithm used."
}

app = Flask(__name__)
if Swagger:
    Swagger(app)

TEMPLATE_DB = {}

@app.route('/enroll', methods=['POST'])
def enroll():
    """
    Enroll a biometric image.
    ---
    tags:
      - Biometric API
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - image
            properties:
              image:
                type: string
                format: byte
                description: Base64-encoded image
    responses:
      200:
        description: Enrollment successful
        content:
          application/json:
            schema:
              type: object
              properties:
                template_id:
                  type: string
                  description: Unique ID for enrolled template
                processing_time_ms:
                  type: number
      400:
        description: Enrollment failed
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                processing_time_ms:
                  type: number
    """
    start_time = time.time()
    data = request.json
    image = data.get('image')

    if not image:
        return jsonify({
            "error": "FTE: Missing or invalid image input",
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }), 400

    try:
        base64.b64decode(image)
    except Exception:
        return jsonify({
            "error": "FTE: Unable to decode image",
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }), 400

    template_id = str(uuid.uuid4())
    TEMPLATE_DB[template_id] = image

    return jsonify({
        "template_id": template_id,
        "processing_time_ms": int((time.time() - start_time) * 1000)
    })


@app.route('/identify', methods=['POST'])
def identify():
    """
    Identify a biometric subject.
    ---
    tags:
      - Biometric API
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - image
            properties:
              image:
                type: string
                format: byte
                description: Base64-encoded image
    responses:
      200:
        description: Identification results
        content:
          application/json:
            schema:
              type: object
              properties:
                matches:
                  type: array
                  items:
                    type: object
                    properties:
                      template_id:
                        type: string
                      score:
                        type: number
                        format: float
                processing_time_ms:
                  type: number
      400:
        description: Identification failed
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                processing_time_ms:
                  type: number
    """
    start_time = time.time()
    data = request.json or {}
    image = data.get('image')

    # Validate image input
    if not image:
        return jsonify({
            "error": "FTE: Missing or invalid image input",
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }), 400
    try:
        base64.b64decode(image)
    except Exception:
        return jsonify({
            "error": "FTE: Unable to decode image",
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }), 400

    # Respect optional top_k parameter (min 1, max 100)
    top_k = data.get('top_k', 100)
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 100
    top_k = max(1, min(top_k, 100))

    # Build matches up to top_k templates
    keys = list(TEMPLATE_DB.keys())
    matches = [
        {
            "template_id": tid,
            "score": round(random.uniform(0.5, 1.0), 4)
        }
        for tid in keys[:top_k]
    ]
    return jsonify({
        "matches": matches,
        "processing_time_ms": int((time.time() - start_time) * 1000)
    })


@app.route('/clear', methods=['POST'])
def clear_gallery():
    """
    Clear the template gallery.
    ---
    tags:
      - Biometric API
    responses:
      200:
        description: Gallery cleared
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                processing_time_ms:
                  type: number
    """
    start_time = time.time()
    TEMPLATE_DB.clear()
    return jsonify({
        "message": "Template gallery cleared",
        "processing_time_ms": int((time.time() - start_time) * 1000)
    })
@app.route('/pad', methods=['POST'])
def pad():
    """
    Presentation Attack Detection (PAD).
    ---
    tags:
      - Biometric API
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - image
            properties:
              image:
                type: string
                format: byte
                description: Base64-encoded image
    responses:
      200:
        description: PAD result
        content:
          application/json:
            schema:
              type: object
              properties:
                is_live:
                  type: boolean
                reason:
                  type: string
                processing_time_ms:
                  type: number
      400:
        description: PAD failed
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                processing_time_ms:
                  type: number
    """
    start_time = time.time()
    data = request.json or {}
    image = data.get('image')

    if not image:
        return jsonify({
            "error": "FTE: Missing or invalid image input",
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }), 400
    try:
        base64.b64decode(image)
    except Exception:
        return jsonify({
            "error": "FTE: Unable to decode image",
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }), 400

    is_live = random.choice([True, False])
    reason = "liveness detected" if is_live else "spoof detected"
    return jsonify({
        "is_live": is_live,
        "reason": reason,
        "processing_time_ms": int((time.time() - start_time) * 1000)
    })
@app.route('/info', methods=['GET'])
def info():
    """
    Get information about the tested algorithm.
    ---
    tags:
      - Biometric API
    responses:
      200:
        description: Algorithm information
        content:
          application/json:
            schema:
              type: object
              properties:
                company:
                  type: string
                  description: Algorithm provider/company
                product_name:
                  type: string
                  description: Product name of the algorithm
                version:
                  type: string
                  description: Algorithm version identifier
                thresholds:
                  type: object
                  description: Optional decision thresholds
                  additionalProperties:
                    type: number
                description:
                  type: string
                  description: Optional algorithm description
    """
    return jsonify(ALGO_INFO)

@app.route('/verify', methods=['POST'])
def verify():
    """
    Verify two biometric images (one-to-one).
    ---
    tags:
      - Biometric API
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - image1
              - image2
            properties:
              image1:
                type: string
                format: byte
                description: Base64-encoded first image
              image2:
                type: string
                format: byte
                description: Base64-encoded second image
    responses:
      200:
        description: Verification result
        content:
          application/json:
            schema:
              type: object
              properties:
                score:
                  type: number
                  format: float
                  description: Similarity score between images
                decision:
                  type: boolean
                  description: True if score meets or exceeds the verify threshold
                processing_time_ms:
                  type: number
      400:
        description: Verification failed
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                processing_time_ms:
                  type: number
    """
    start_time = time.time()
    data = request.json or {}
    image1 = data.get('image1')
    image2 = data.get('image2')
    if not image1 or not image2:
        return jsonify({
            "error": "FTE: Missing or invalid image input",
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }), 400

    try:
        base64.b64decode(image1)
        base64.b64decode(image2)
    except Exception:
        return jsonify({
            "error": "FTE: Unable to decode image",
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }), 400

    # Simulate a similarity score between 0.0 and 1.0
    score = round(random.uniform(0.0, 1.0), 4)
    threshold = ALGO_INFO.get('thresholds', {}).get('verify')
    decision = score >= threshold if threshold is not None else None

    result = {
        "score": score,
        "processing_time_ms": int((time.time() - start_time) * 1000)
    }
    if decision is not None:
        result['decision'] = decision
    return jsonify(result)

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
    # Run server without debug mode to avoid Jinja2 extension reload issues
    app.run(debug=False, port=5001)
