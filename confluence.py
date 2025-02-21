"""This script acts as a API to fetch data->pages from the confluence"""
import os
import json
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup

load_dotenv()

confluence_domain = os.getenv("CONFLUENCE_DOMAIN")
username = os.getenv("USERNAME")
api_token = os.getenv("API_TOKEN")


auth = HTTPBasicAuth(username, api_token)


def fetch_pages():
    """This function sends out request along with the credentials"""
    url = f"{confluence_domain}/rest/api/content?expand=title,body.storage"
    response = requests.get(url, auth=auth)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch pages. Status code: {response.status_code}")
        return None


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

print("Extracted Pages Data")
for title, content in extracted_pages.items():
    print(f" Title: {title}")
    print(f" Content Preview: {content}...\n")

with open("confluence_data.json", "w", encoding="utf-8") as f:
    json.dump(extracted_pages, f, indent=4)

print("Data successfully saved to 'confluence_data.json'!")
