import requests
import base64

SERVER_URL = "http://localhost:5001"

# Load image and convert to base64
def encode_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def enroll(image_path):
    image_b64 = encode_image(image_path)
    response = requests.post(f"{SERVER_URL}/enroll", json={"image": image_b64})

    print("Status Code:", response.status_code)
    print("Raw Response Text:", response.text)

    try:
        result = response.json()
        print("Enroll Response:", result)
        return result.get("template_id")
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse response as JSON")
        return None


def identify(image_path):
    image_b64 = encode_image(image_path)
    response = requests.post(f"{SERVER_URL}/identify", json={"image": image_b64})
    print("Identify Response:", response.json())

def clear():
    response = requests.post(f"{SERVER_URL}/clear")
    print("Clear Response:", response.json())

if __name__ == "__main__":
    # Use your own image path here (PNG or JPEG)
    img_path = "face1.jpeg"
    
    print("\n--- Enrolling image ---")
    template_id = enroll(img_path)
    template_id = enroll(img_path)
    template_id = enroll(img_path)
    template_id = enroll(img_path)
    template_id = enroll(img_path)

    print("\n--- Identifying image ---")
    identify(img_path)

    clear()
    identify(img_path)


    template_id = enroll(img_path)
    template_id = enroll(img_path)

    print("\n--- Identifying image ---")
    identify(img_path)
