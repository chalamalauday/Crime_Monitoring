import requests
import os

url = "http://127.0.0.1:5000/digest"
pdf_path = r"c:\Users\Uday Chalamala\OneDrive\Desktop\Crime_Monitoring\test\Articles-13_06_2025.pdf"

if not os.path.exists(pdf_path):
    print("PDF does not exist:", pdf_path)
    exit(1)

files = [
    ('files', open(pdf_path, 'rb'))
]
data = {
    'keywords': '',
    'districts': 'Vijayawada, Guntur',
    'date': ''
}

print("Uploading to", url)
try:
    response = requests.post(url, files=files, data=data, allow_redirects=True)
    if response.status_code == 200:
        print("Success! Status code:", response.status_code)
        if "PDF Digest Generated Successfully" in response.text or "alert-success" in response.text:
            print("Digest processed successfully.")
            # Let's save the HTML to see the results
            with open("test_out.html", "w", encoding="utf-8") as f:
                f.write(response.text)
            print("Response saved to test_out.html.")
        elif "No Intelligence Found" in response.text:
            print("No Intelligence Found.")
        else:
            print("Uploaded, but output was unexpected. Saved to test_out.html.")
            with open("test_out.html", "w", encoding="utf-8") as f:
                f.write(response.text)
    else:
        print("Error:", response.status_code)
        print(response.text[:500])
except Exception as e:
    print("Exception", e)
