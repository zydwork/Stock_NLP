import requests
import json

def get_snapshots(url):
    base_url = 'http://archive.org/wayback/available'
    params = {'url': url}
    response = requests.get(base_url, params=params)
    data = response.json()
    return data

# Example usage
url = 'http://finance.yahoo.com/news'
snapshots = get_snapshots(url)
print(json.dumps(snapshots, indent=4))