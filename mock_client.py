import requests
import base64

PORT = 5001
SERVER_URL = f"http://localhost:{PORT}"

TEST_IMAGE_PATH = "testdata/face1.png"

api_key = "this_is_a_secret_key"

end_points = {
            'verify': '/verify',
            'identify': '/identify',
            'enroll': '/enroll',
            'clear': '/clear',
            'pad': '/pad',
            'info': '/info',
            'quit': '/quit'
        }

# Load image and convert to base64
def encode_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def response_error_str(response):
    try:
        json_r = response.json()
    except:
        print(response)
        json_r={"error":"json_decode error"}
    if "error" in json_r:
        return json_r["error"]
    else:
        return response.status_code

def set_api_key(api_key):
    api_key = api_key.strip()

def headers():
    return {
        'X-API-Key': api_key
    }

def enroll(image_path):
    if 'enroll' not in end_points:
        print("⚠️ Warning: 'enroll' endpoint not defined in end_points")
        return None
    image_b64 = encode_image(image_path)
    response = requests.post(f"{SERVER_URL}/{end_points['enroll']}", json={"image": image_b64}, headers=headers())

    assert response.status_code == 200, f"Enroll failed with status {response_error_str(response)}"
    result = response.json()
    assert "template_id" in result, "Enroll response missing 'template_id'"
    assert "processing_time_ms" in result, "Enroll response missing 'processing_time_ms'"

    print("✅ Enroll success:", result)
    return result["template_id"]


def identify(image_path, top_k=100):
    if 'identify' not in end_points:
        print("⚠️ Warning: 'identify' endpoint not defined in end_points")
        return []
    image_b64 = encode_image(image_path)
    payload = {"image": image_b64, "top_k": top_k}
    response = requests.post(f"{SERVER_URL}/{end_points['identify']}", json=payload, headers=headers())
    assert response.status_code == 200, f"Identify failed with error {response_error_str(response)}"
    result = response.json()
    assert "matches" in result, "Identify response missing 'matches'"
    assert "processing_time_ms" in result, "Identify response missing 'processing_time_ms'"
    assert isinstance(result["matches"], list), "'matches' should be a list"
    assert len(result["matches"]) <= top_k, f"More matches returned than top_k ({top_k})"

    print(f"✅ Identify success: {len(result['matches'])} match(es)")
    return result["matches"]


def clear():
    if 'clear' not in end_points:
        print("⚠️ Warning: 'clear' endpoint not defined in end_points")
        return
    response = requests.post(f"{SERVER_URL}/{end_points['clear']}",headers=headers())
    assert response.status_code == 200, f"Clear failed with status {response_error_str(response)}"
    result = response.json()
    #assert result.get("message") == "Template gallery cleared", "Unexpected clear message"
    assert "processing_time_ms" in result, "Clear response missing 'processing_time_ms'"

    print("✅ Clear success:", result)

def test_pad(image_path):
    if 'pad' not in end_points:
        print("⚠️ Warning: 'pad' endpoint not defined in end_points")
        return
    image_b64 = encode_image(image_path)
    response = requests.post(f"{SERVER_URL}/pad", json={"image": image_b64},headers=headers())

    assert response.status_code == 200, f"PAD failed with status {response.json()['error']}"
    result = response.json()

    assert "is_live" in result, "PAD response missing 'is_live'"
    assert isinstance(result["is_live"], bool), "'is_live' must be a boolean"
    assert "reason" in result, "PAD response missing 'reason'"
    assert isinstance(result["reason"], str), "'reason' must be a string"
    assert "processing_time_ms" in result, "PAD response missing 'processing_time_ms'"

    print("✅ PAD check success:", result)

def info():
    if 'info' not in end_points:
        print("⚠️ Warning: 'info' endpoint not defined in end_points")
        return None
    response = requests.get(f"{SERVER_URL}/{end_points['info']}")
    assert response.status_code == 200, f"Info failed with status {response_error_str(response)}"
    result = response.json()
    print(result)
    assert "company" in result, "Info response missing 'company'"
    assert "product_name" in result, "Info response missing 'product_name'"
    assert "version" in result, "Info response missing 'version'"
    if "thresholds" not in result:
        print("⚠️ Warning: Info response missing 'thresholds'")
    elif not isinstance(result["thresholds"], dict):
        print("⚠️ Warning: 'thresholds' should be a dict")
    if "description" not in result:
        print("⚠️ Warning: Info response missing 'description'")
    if "gallery_size" not in result:
        print("⚠️ Warning: Info response missing 'gallery_size'")
    elif not isinstance(result["gallery_size"], int):
        print("⚠️ Warning: 'gallery_size' should be an integer")
    if "end_points" not in result:
        print("⚠️ Warning: Info response missing 'end_points'")
    print("✅ Info success:", result)
    return result

def verify(image1_path, image2_path):
    if 'verify' not in end_points:
        print("⚠️ Warning: 'verify' endpoint not defined in end_points")
        return None
    img1_b64 = encode_image(image1_path)
    img2_b64 = encode_image(image2_path)
    payload = {"image1": img1_b64, "image2": img2_b64}
    response = requests.post(f"{SERVER_URL}/{end_points['verify']}", json=payload)

    assert response.status_code == 200, f"Verify failed with status {response_error_str(response)}"
    result = response.json()
    assert "score" in result, "Verify response missing 'score'"
    assert isinstance(result["score"], (int, float)), "'score' should be a number"
    assert "processing_time_ms" in result, "Verify response missing 'processing_time_ms'"
    assert isinstance(result["processing_time_ms"], (int, float)), "'processing_time_ms' should be a number"
    assert "decision" in result, "Verify response missing 'decision'"
    assert isinstance(result["decision"], bool), "'decision' should be a boolean"
    print("✅ Verify success:", result)
    return result

def quit_server():
    if 'quit' not in end_points:
        print("⚠️ Warning: 'quit' endpoint not defined in end_points")
        return
    response = requests.get(f"{SERVER_URL}/{end_points['quit']}")
    assert response.status_code == 200, f"Quit failed with status {response_error_str(response)}"
    result = response.json()
    print("✅ Quit success:", result)
    return result

if __name__ == "__main__":
    img_path = TEST_IMAGE_PATH

    # Test info endpoint
    print("\nℹ️ Testing /info endpoint...")
    try:
        info_dict = info()
    except Exception as e:
        print(f"❌ Error connection to /info test: {e}")
        print(f"\n👋 Server can't be found at {SERVER_URL}")
        exit(1)
    if 'end_points' in info_dict:
        print("Available endpoints:", info_dict['end_points'])
        end_points = info_dict['end_points']

    # Ensure gallery is empty
    print("\n🔄 Clearing gallery...")
    clear()

    # Test identify on empty gallery
    print("\n🔎 Testing /identify on empty gallery...")
    matches = identify(img_path, top_k=5)
    assert len(matches) == 0, f"Expected 0 matches on empty gallery, got {len(matches)}"

    # Enroll multiple templates
    N = 5
    print(f"\n📝 Enrolling {N} images...")
    ids = [enroll(img_path) for _ in range(N)]
    assert len(ids) == N, f"Enroll returned {len(ids)} IDs, expected {N}"

    # Test identify with top_k smaller than gallery size
    top_k = 3
    print(f"\n🔎 Testing /identify with top_k={top_k}...")
    matches = identify(img_path, top_k=top_k)
    assert len(matches) == top_k, f"Expected {top_k} matches, got {len(matches)}"

    # Test identify with top_k larger than gallery size
    top_k_large = N + 2
    print(f"\n🔎 Testing /identify with top_k={top_k_large} (larger than gallery)...")
    matches = identify(img_path, top_k=top_k_large)
    assert len(matches) == N, f"Expected {N} matches, got {len(matches)}"

    # Test gallery of one
    print("\n🔄 Clearing gallery for single-item test...")
    clear()
    print("\n📝 Enrolling 1 image...")
    one_id = enroll(img_path)
    print("\n🔎 Testing /identify for gallery of one...")
    matches = identify(img_path)
    assert len(matches) == 1, f"Expected 1 match for gallery of one, got {len(matches)}"

    # Test verify endpoint
    print("\n🧪 Testing /verify endpoint with same image pair...")
    verify(img_path, img_path)

    # Test PAD endpoint
    print("\n🎃 Testing /pad endpoint...")
    test_pad(img_path)

    print("\n✅ All endpoint tests passed successfully.")

    quit_server()
    print("\n👋 Server shutdown initiated.")
