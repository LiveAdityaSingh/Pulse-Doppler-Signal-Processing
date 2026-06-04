import requests
import os
import sys
import time

url = "http://localhost:8000/analyze"
wav_file = os.path.join("data", "sample_1.wav")

def test():
    if not os.path.exists(wav_file):
        print("Data file not found for testing.")
        return
        
    print(f"Testing with file: {wav_file}")
    with open(wav_file, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        data = {
            "fmin": 1500.0,
            "fmax": 10000.0,
            "noise_factor": 1.5,
            "tmin": 0.0,
            "tmax": 10.0 # Just testing 10 seconds for speed
        }
        
        try:
            print("Calling FastAPI /analyze endpoint...")
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            res_json = response.json()
            print("\n================ REPORT ================")
            report_text = res_json.get("report", "No report field in response")
            print(report_text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
            print("---------------- METRICS ---------------")
            print(f"Overall Diagnostic Confidence : {res_json.get('confidence', 0):.1f}%")
            print(f"System Accuracy Estimation    : {res_json.get('accuracy', 0):.1f}%")
            print("========================================\n")
            
            if "ML Analysis" in res_json.get("report", ""):
                print("SUCCESS: Model evaluation found in the report.")
            else:
                print("WARNING: Model evaluation not found in the report.")
        except Exception as e:
            print(f"Error calling API: {e}")

if __name__ == "__main__":
    # Add a small delay for safety in case server is just booting
    test()
