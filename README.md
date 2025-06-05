# Mock Biometric Test Server

This is a mock biometric API server designed for use in biometric Proof-of-Concept (PoC) environments, such as the Brisbane Airport evaluation. It simulates biometric **enrolment**, **identification**, and **gallery management**, exposing a simple REST API with interactive Swagger documentation.

## Features

- REST API for **/enroll**, **/identify**, and **/clear**
- Input: Base64-encoded images (optional top_k for identify)
- Output: Simulated template IDs and match scores
- In-memory gallery (no persistent storage)
- Swagger UI for interactive API testing
- Processing time (in milliseconds) included in every response

---

## Requirements

- Python 3.8+
- pip

### Install dependencies

```bash
pip install flask flasgger
````

---

## Running the Server

```bash
python mock_server.py
```

The server will start on:

```
http://localhost:5001
```

---

## Swagger UI

Interactive API documentation and testing is available at:

```
http://localhost:5001/apidocs/
```

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

---

### 🔹 `POST /identify`

Performs identification against the enrolled gallery.

(optional top_k:

* type: integer
* minimum: 1, maximum: 100
* default: 100
* description: Number of top matches to return (max 100)))

#### Request Body:

```json
{
  "image": "<base64-encoded-image>"
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

## Testing the API

A sample client (`mock_client.py`) can be created to automate image uploads and interaction with the server. You can also use Swagger UI or tools like Postman.

---

## Notes

* This is a **simulation only**. No actual biometric algorithms are used.
* The `TEMPLATE_DB` is stored in memory and cleared on restart or via `/clear`.
* Intended for PoC and testing integrations in secure environments (e.g. behind VPN).

---

## License

MIT – Use freely with attribution.

---

## Author

Dr. Ted Dunstone
CEO, BixeLab
[https://bixelab.com](https://bixelab.com)
