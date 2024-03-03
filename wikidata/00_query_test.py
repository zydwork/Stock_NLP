import requests

# Define the endpoint URL
endpoint_url = "https://query.wikidata.org/sparql"

# Define the SPARQL query
query = """
SELECT DISTINCT ?id ?idLabel ?exchange ?ticker
WHERE {
    ?id wdt:P31/wdt:P279* wd:Q4830453 .
    ?id p:P414 ?exchange . 
    ?exchange ps:P414 wd:Q13677 .
    ?exchange pq:P249 ?ticker . FILTER(LCASE(STR(?ticker)) = 'ibm') .

    ?id rdfs:label ?idLabel 
    FILTER(LANG(?idLabel) = 'en').
}
"""

# Send the request
response = requests.get(endpoint_url, params={'query': query, 'format': 'json'})


# Check if the response was successful
if response.ok:
    # Process the results
    data = response.json()
    print(data)
    for result in data['results']['bindings']:
        company_label = result['companyLabel']['value']
        product_label = result['productLabel']['value']
        print(f"Company: {company_label}, Product: {product_label}")
else:
    print("Error in fetching data")