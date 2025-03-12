import requests

# API URL
url = "http://127.0.0.1:8000/predict/"

# Test cases
test_emails = [
    {
        "text": "Congratulations! You have won a lottery of $1,000,000. Click here to claim your prize.",
        "model": "all"
    },
    {
        "text": "Dear customer, your bank account requires verification. Please update your details at the given link.",
        "model": "all"
    },
    {
        "text": "Hello John, let's catch up over coffee tomorrow at Starbucks.",
        "model": "all"
    }
]

# Loop through test cases
for i, email in enumerate(test_emails, start=1):
    response = requests.post(url, json=email)
    
    # Ensure the request was successful
    if response.status_code == 200:
        result = response.json()  # Extract JSON response
        print(f"Test {i}: {email['text']}")
        print("Response:", result)
    else:
        print(f"Test {i}: Request failed with status code {response.status_code}")
        print("Error:", response.text)

    print("-" * 50)