# import os
# import requests
# from dotenv import load_dotenv
# from requests.auth import HTTPBasicAuth
# from bs4 import BeautifulSoup
#
# load_dotenv()
# confluence_domain = os.getenv("CONFLUENCE_DOMAIN")
# username = os.getenv("USERNAME")
# api_token = os.getenv("API_TOKEN")
#
#
# auth = HTTPBasicAuth(username, api_token)
#
#
# def fetch_pages():
#
#     url = f"{confluence_domain}/rest/api/content?expand=title,body.storage"
#     print(url)
#     response = requests.get(url, auth=auth)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"Failed to fetch pages. Status code: {response.status_code}")
#         return None
#
# data = fetch_pages()
# extracted_pages = {}
# if data:
#
#     for page in data.get("results", []):
#         # Check if 'title' exists
#         title = page.get("title", "No title available")
#         # print(f"Page Title: {title}")
#
#         body_storage = page.get("body", {}).get("storage", {}).get("value", "No content available")
#         # print(f"Page Content (HTML): {body_storage[:2500]}...")
#
#         soup = BeautifulSoup(body_storage, "html.parser")
#         plain_text = soup.get_text(separator=" ")  # Extract text with spaces between elements
#
#         extracted_pages[title] = plain_text
#
# # print(extracted_pages)


import os
import requests
import json
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
confluence_domain = os.getenv("CONFLUENCE_DOMAIN")
username = os.getenv("USERNAME")
api_token = os.getenv("API_TOKEN")

# Authentication
auth = HTTPBasicAuth(username, api_token)


# Function to fetch pages
def fetch_pages():
    url = f"{confluence_domain}/rest/api/content?expand=title,body.storage"
    response = requests.get(url, auth=auth)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch pages. Status code: {response.status_code}")
        return None


# Fetch data
data = fetch_pages()
extracted_pages = {}

if data:
    for page in data.get("results", []):
        title = page.get("title", "No title available")
        html_content = page.get("body", {}).get("storage", {}).get("value", "No content available")

        # Convert HTML to plain text
        soup = BeautifulSoup(html_content, "html.parser")
        plain_text = soup.get_text(separator=" ")

        # Store in dictionary
        extracted_pages[title] = plain_text

    # Print extracted content for verification
print("\n✅ Extracted Pages Data:\n")
for title, content in extracted_pages.items():
    print(f"🔹 Title: {title}")
    print(f"📜 Content Preview: {content}...\n")  # Display only the first 500 characters

# Save to JSON file for later use
with open("confluence_data.json", "w", encoding="utf-8") as f:
    json.dump(extracted_pages, f, indent=4)

print("\n✅ Data successfully saved to 'confluence_data.json'!")
