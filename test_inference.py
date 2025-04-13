import requests

def csv_file_to_string(filepath: str) -> str:
    with open(filepath, 'r') as file:
        return file.read()

def test_eeg_inference(csv_fname):
    url = "http://127.0.0.1:8000/inference"
    csv_string = csv_file_to_string(csv_fname)

    payload = {
        "csv_data": csv_string
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Failed with status code:", response.status_code)
        print(response.text)

if __name__ == "__main__":
    test_eeg_inference("data/raw/yoyo_1_1.csv")
    test_eeg_inference("data/raw/yoyo_5_1.csv")

