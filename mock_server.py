from flask import Flask, request, jsonify
from flasgger import Swagger
import base64
import uuid
import random
import time

app = Flask(__name__)
swagger = Swagger(app)

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

    matches = [
        {
            "template_id": tid,
            "score": round(random.uniform(0.5, 1.0), 4)
        }
        for tid in list(TEMPLATE_DB.keys())[:100]
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


if __name__ == '__main__':
    app.run(debug=True, port=5001)
