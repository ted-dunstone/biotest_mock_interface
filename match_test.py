import requests
import base64
import os
import csv
import argparse
from pathlib import Path
import time

SERVER_URL = "http://localhost:5001"

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
        json_r = {"error": "json_decode error"}
    if "error" in json_r:
        return json_r["error"]
    else:
        return response.status_code

def get_info():
    """Get server info including algorithm name"""
    if 'info' not in end_points:
        print("⚠️ Warning: 'info' endpoint not defined in end_points")
        return None
    
    response = requests.get(f"{SERVER_URL}{end_points['info']}")
    if response.status_code != 200:
        print(f"❌ Info failed with status {response_error_str(response)}")
        return None
    
    result = response.json()
    print("✅ Server info retrieved:", result)
    return result

def clear_gallery():
    """Clear the gallery"""
    if 'clear' not in end_points:
        print("⚠️ Warning: 'clear' endpoint not defined in end_points")
        return False
    
    response = requests.post(f"{SERVER_URL}{end_points['clear']}")
    if response.status_code != 200:
        print(f"❌ Clear failed with status {response_error_str(response)}")
        return False
    
    result = response.json()
    print("✅ Gallery cleared:", result)
    return True

def enroll_image(image_path, image_name):
    """Enroll a single image"""
    if 'enroll' not in end_points:
        print("⚠️ Warning: 'enroll' endpoint not defined in end_points")
        return None
    
    try:
        image_b64 = encode_image(image_path)
        response = requests.post(f"{SERVER_URL}{end_points['enroll']}", json={"image": image_b64})
        
        if response.status_code != 200:
            print(f"❌ Enroll failed for {image_name}: {response_error_str(response)}")
            return None
        
        result = response.json()
        if "template_id" not in result:
            print(f"❌ Enroll response missing 'template_id' for {image_name}")
            return None
        
        print(f"✅ Enrolled {image_name}: template_id={result['template_id']}")
        return result["template_id"]
    
    except Exception as e:
        print(f"❌ Error enrolling {image_name}: {str(e)}")
        return None

def identify_image(image_path, image_name, top_k=100):
    """Identify a single image against the gallery"""
    if 'identify' not in end_points:
        print("⚠️ Warning: 'identify' endpoint not defined in end_points")
        return []
    
    try:
        image_b64 = encode_image(image_path)
        payload = {"image": image_b64, "top_k": top_k}
        response = requests.post(f"{SERVER_URL}{end_points['identify']}", json=payload)
        
        if response.status_code != 200:
            print(f"❌ Identify failed for {image_name}: {response_error_str(response)}")
            return []
        
        result = response.json()
        if "matches" not in result:
            print(f"❌ Identify response missing 'matches' for {image_name}")
            return []
        
        matches = result["matches"]
        print(f"✅ Identified {image_name}: {len(matches)} match(es)")
        return matches
    
    except Exception as e:
        print(f"❌ Error identifying {image_name}: {str(e)}")
        return []

def get_image_files(directory):
    """Get all image files from a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory does not exist: {directory}")
        return []
    
    image_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)

import matplotlib.pyplot as plt

def plot_score_histogram(scores, labels, bins=50, match_label=1, nonmatch_label=0, title="Score Histogram"):
    """
    Plot a histogram of match vs non-match scores.
    
    Parameters:
        scores (list or np.array): Similarity or distance scores.
        labels (list or np.array): Binary labels (1 for match, 0 for non-match).
        bins (int): Number of histogram bins.
        match_label (int): Value representing matches.
        nonmatch_label (int): Value representing non-matches.
        title (str): Title for the plot.
    """
    import numpy as np

    scores = np.array(scores)
    labels = np.array(labels)

    match_scores = scores[labels == match_label]
    nonmatch_scores = scores[labels == nonmatch_label]

    plt.figure(figsize=(10, 6))
    plt.hist(nonmatch_scores, bins=bins, alpha=0.6, label='Non-match', density=True)
    plt.hist(match_scores, bins=bins, alpha=0.6, label='Match', density=True)
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_histogram_from_csv_data(csv_data):
    scores = [entry['score'] for entry in csv_data]
    labels = [entry['is_same_person'] for entry in csv_data]
    
    plot_score_histogram(scores, labels, title="Match vs Non-Match Score Distribution")

import matplotlib.pyplot as plt
import numpy as np

def compute_cmc(csv_data, max_rank=10):
    """
    Compute and plot the CMC curve from csv_data.

    Args:
        csv_data (list of dict): Each entry must include 'probe_image', 'rank', and 'is_same_person'.
        max_rank (int): The maximum rank to compute CMC up to.
    """
    from collections import defaultdict

    # Group entries by probe
    probe_to_ranks = defaultdict(list)
    for entry in csv_data:
        probe = entry['probe_image']
        is_match = entry['is_same_person']
        rank = entry['rank']
        if is_match:
            probe_to_ranks[probe].append(rank)

    # Compute rank of first correct match for each probe
    correct_ranks = []
    for probe, ranks in probe_to_ranks.items():
        correct_ranks.append(min(ranks))  # Best (lowest) rank for correct match

    correct_ranks = np.array(correct_ranks)

    # Compute CMC values
    cmc_curve = []
    num_probes = len(correct_ranks)
    for r in range(1, max_rank + 1):
        cmc_value = np.mean(correct_ranks <= r)
        cmc_curve.append(cmc_value)

    # Plot CMC curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_rank + 1), cmc_curve, marker='o')
    plt.title("CMC Curve")
    plt.xlabel("Rank")
    plt.ylabel("Identification Rate")
    plt.xticks(range(1, max_rank + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Face recognition gallery identification")
    parser.add_argument("gallery_dir", help="Directory containing gallery images to enroll")
    parser.add_argument("probe_dir", help="Directory containing probe images to identify")
    parser.add_argument("--top_k", type=int, default=50, help="Number of top matches to return (default: 10)")
    
    args = parser.parse_args()
    
    # Get server info
    print("📡 Getting server information...")
    info = get_info()
    if not info:
        print("❌ Failed to get server info. Exiting.")
        return
    
    # Extract algorithm name for CSV filename
    algorithm_name = info.get('product_name', 'unknown_algorithm')
    algorithm_name = algorithm_name.replace(' ', '_').replace('/', '_')
    csv_filename = f"{algorithm_name}.csv"
    
    # Get image files
    print(f"📁 Scanning gallery directory: {args.gallery_dir}")
    gallery_files = get_image_files(args.gallery_dir)
    if not gallery_files:
        print("❌ No image files found in gallery directory. Exiting.")
        return
    
    print(f"📁 Scanning probe directory: {args.probe_dir}")
    probe_files = get_image_files(args.probe_dir)
    if not probe_files:
        print("❌ No image files found in probe directory. Exiting.")
        return
    
    print(f"📊 Found {len(gallery_files)} gallery images and {len(probe_files)} probe images")
    
    # Clear gallery
    print("\n🔄 Clearing gallery...")
    if not clear_gallery():
        print("❌ Failed to clear gallery. Exiting.")
        return
    
    # Enroll gallery images
    print(f"\n📝 Enrolling {len(gallery_files)} gallery images...")
    gallery_enrollment = {}  # filename -> template_id
    
    for gallery_file in gallery_files:
        template_id = enroll_image(gallery_file, gallery_file.name)
        if template_id:
            gallery_enrollment[gallery_file.name] = template_id
        else:
            print(f"⚠️ Failed to enroll {gallery_file.name}")
    
    print(f"✅ Successfully enrolled {len(gallery_enrollment)} out of {len(gallery_files)} gallery images")
    
    if not gallery_enrollment:
        print("❌ No images were successfully enrolled. Exiting.")
        return
    
    # Create reverse mapping: template_id -> filename
    template_to_filename = {v: k for k, v in gallery_enrollment.items()}
    
    # Prepare CSV data
    csv_data = []
    
    # Identify probe images
    print(f"\n🔎 Identifying {len(probe_files)} probe images...")
    
    for i, probe_file in enumerate(probe_files, 1):
        print(f"\nProcessing probe {i}/{len(probe_files)}: {probe_file.name}")
        
        matches = identify_image(probe_file, probe_file.name, args.top_k)
        
        if matches:
            for rank, match in enumerate(matches, 1):
                template_id = match.get('template_id', 'unknown')
                score = match.get('score', 0.0)
                is_same_person = False
                gallery_filename = template_to_filename.get(template_id, 'unknown')
                if probe_file.name.split('_')[0] == gallery_filename.split('_')[0]:
                    is_same_person = True
                
                csv_data.append({
                    'probe_image': probe_file.name,
                    'rank': rank,
                    'gallery_match': gallery_filename,
                    'template_id': template_id,
                    'score': score,
                    'is_same_person': is_same_person
                })
        else:
            # No matches found
            csv_data.append({
                'probe_image': probe_file.name,
                'rank': 0,
                'gallery_match': 'no_match',
                'template_id': 'none',
                'score': 0.0,
                'is_same_person': False
            })
    
    # Write CSV file
    print(f"\n💾 Writing results to {csv_filename}...")
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['probe_image', 'rank', 'gallery_match', 'template_id', 'score','is_same_person']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"✅ Results saved to {csv_filename}")
    print(f"📊 Total records: {len(csv_data)}")
    
    # Summary statistics
    unique_probes = len(set(row['probe_image'] for row in csv_data))
    probes_with_matches = len(set(row['probe_image'] for row in csv_data if row['rank'] > 0))
    
    print(f"\n📈 Summary:")
    print(f"   - Gallery images enrolled: {len(gallery_enrollment)}")
    print(f"   - probe images processed: {unique_probes}")
    print(f"   - probes with matches: {probes_with_matches}")
    print(f"   - probes with no matches: {unique_probes - probes_with_matches}")
    print(f"   - Algorithm: {algorithm_name}")

    # Plot histogram of scores
    print("\n📊 Plotting score histogram...")
    plot_histogram_from_csv_data(csv_data)
    compute_cmc(csv_data, max_rank=10)

if __name__ == "__main__":
    main()