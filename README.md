# Mock Biometric Test Server

This is a mock biometric API server designed for use in biometric Proof-of-Concept (PoC) environments. It simulates biometric **enrolment**, **identification**, and **gallery management**, exposing a simple REST API with interactive Swagger documentation.

## Features

- REST API endpoints: **GET /info**, **POST /enroll**, **POST /identify**, **POST /verify**, **POST /clear**, **POST /pad**
- Input: Base64-encoded images
- Output: simulated template IDs, match scores, liveness decisions, and processing time (ms)
- Interactive Swagger UI (if `flasgger` is installed) at `/apidocs/`

> [!IMPORTANT]  
> Note this currently only designed be run completely inside a secure trusted testing network, and so there is currently no encryption or authentication on REST calls.

---

## Requirements

- Python 3.8+
- pip
- supply your own face image (default `face1.png`)


### Install dependencies

```bash
# Core dependencies
pip install flask

# Optional Swagger UI (interactive docs)
pip install flasgger
```

---

## Running the Server

```bash
python mock_server.py
```
The server will start on `http://localhost:5001`. If the optional `flasgger` package is installed,
interactive API documentation is available at:

```
http://localhost:5001/apidocs/
```

---


---

## API Endpoints

### 🔹 `POST /enroll`

Enrolls a biometric template from a Base64-encoded image.

#### Request Body:

```json
{
  "image": "<base64-encoded-image>"
}
```

#### Success Response:

```json
{
  "template_id": "uuid-string",
  "processing_time_ms": 3
}
```

#### Error Response:
e.g.
```json
{
   "error": "FTE: Unable to decode image"
}
```
---

### 🔹 `POST /identify`

Identify enrolled templates matching the provided image.

#### Request Body:

```json
{
  "image": "<base64-encoded-image>",
  "top_k": 50            # Optional: number of top matches to return (1–100, default=100)
}
```

#### Success Response:

```json
{
  "matches": [
    {"template_id": "uuid-1", "score": 0.982},
    ...
  ],
  "processing_time_ms": 5
}
```

#### Error Response:
e.g.
```json
{
   "error": "FTA: Unable to decode image"
}
```
---

### 🔹 `POST /clear`

Clears the in-memory biometric gallery.

#### Success Response:

```json
{
  "message": "Template gallery cleared",
  "processing_time_ms": 1
}
```

---

### 🔹 `GET /info`

Get information about the tested algorithm.

#### Success Response:

```json
{
  "company": "BixeLab",
  "product_name": "Mock Biometric Test Server",
  "version": "1.0.0",
  "thresholds": { "identify": 0.5, "verify": 0.75 },
  "description": "A mock biometric API for PoC and testing, no real biometric algorithm used."
}
```

---

### 🔹 `POST /verify`

Verify two biometric images (one-to-one).

#### Request Body:

```json
{
  "image1": "<base64-encoded-image>",
  "image2": "<base64-encoded-image>"
}
```

#### Success Response:

```json
{
  "score": 0.8321,
  "decision": true,
  "processing_time_ms": 4
}
```

#### Error Response:

```json
{
  "error": "FTE: Missing or invalid image input",
  "processing_time_ms": 2
}
```
---

## 🔍 Passive PAD (Presentation Attack Detection)

### 🔹 `POST /pad`

Simulates a passive PAD check to determine whether a biometric image represents a live person or a spoof (e.g. printed photo or screen replay).

#### Request Body:
```json
{
  "image": "<base64-encoded-image>"
}
```

#### Response:
```json
{
  "is_live": true,
  "reason": "Face structure consistent with live subject",
  "processing_time_ms": 3
}
```
Response includes a human-readable explanation and processing time.

Intended for testing integration and fallback logic in client applications.

## Testing the API

A sample client (`mock_client.py`) is provided to show how interaction with the server work. You can also use Swagger UI or tools like Postman.

In practice this will can be used to test the interfaces of your server.

---

> [!NOTE] 
> * This is a **mock application only** for integration testing. No actual biometric algorithms are used in this repo.
> * The `TEMPLATE_DB` for the mock is stored in memory and cleared on restart or via `/clear`.
> * Intended for PoC and testing integrations in secure environments (e.g. behind VPN).

---

## License

MIT – Use freely with attribution.

---

## Author

Dr. Ted Dunstone
CEO, BixeLab
[https://bixelab.com](https://bixelab.com)
