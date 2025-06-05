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

    assert response.status_code == 200, f"Enroll failed with status {response.status_code}"
    result = response.json()
    assert "template_id" in result, "Enroll response missing 'template_id'"
    assert "processing_time_ms" in result, "Enroll response missing 'processing_time_ms'"

    print("✅ Enroll success:", result)
    return result["template_id"]


def identify(image_path, top_k=100):
    image_b64 = encode_image(image_path)
    payload = {"image": image_b64, "top_k": top_k}
    response = requests.post(f"{SERVER_URL}/identify", json=payload)

    assert response.status_code == 200, f"Identify failed with status {response.status_code}"
    result = response.json()
    assert "matches" in result, "Identify response missing 'matches'"
    assert "processing_time_ms" in result, "Identify response missing 'processing_time_ms'"
    assert isinstance(result["matches"], list), "'matches' should be a list"
    assert len(result["matches"]) <= top_k, f"More matches returned than top_k ({top_k})"

    print(f"✅ Identify success: {len(result['matches'])} match(es)")
    return result["matches"]


def clear():
    response = requests.post(f"{SERVER_URL}/clear")
    assert response.status_code == 200, f"Clear failed with status {response.status_code}"
    result = response.json()
    assert result.get("message") == "Template gallery cleared", "Unexpected clear message"
    assert "processing_time_ms" in result, "Clear response missing 'processing_time_ms'"

    print("✅ Clear success:", result)


if __name__ == "__main__":
    img_path = "face1.png"

    print("\n🔄 Clear gallery...")
    clear()

    print("\n🧪 Enrolling 5 images...")
    ids = [enroll(img_path) for _ in range(5)]

    print("\n🧪 Identifying with top_k=3...")
    matches = identify(img_path, top_k=3)
    assert len(matches) <= 3, "Returned more matches than expected"

    print("\n🧪 Clearing gallery and testing identification on empty gallery...")
    clear()
    matches = identify(img_path, top_k=5)
    assert len(matches) == 0, "Expected zero matches after clearing"

    print("\n🧪 Re-enroll 2 images and test identify...")
    ids = [enroll(img_path) for _ in range(2)]
    matches = identify(img_path, top_k=10)
    assert len(matches) == 2, "Expected 2 matches after re-enrolling"

    print("\n✅ All tests passed successfully.")
