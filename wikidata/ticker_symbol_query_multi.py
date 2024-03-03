import requests
import pandas as pd
import json
from multiprocessing import Pool, Manager, Lock



def query_wikidata(symbol):
    # Define the endpoint URL
    endpoint_url = "https://query.wikidata.org/sparql"

    # Define the SPARQL query
    query_1 = """
    SELECT ?ticker ?id
        (GROUP_CONCAT(DISTINCT ?idLabel;separator="| | |") AS ?idLabels)
        (GROUP_CONCAT(DISTINCT ?altLabel; separator = "| | |") AS ?aliases)
        (GROUP_CONCAT(DISTINCT ?industryLabel; separator = "| | |") AS ?industries)
        (GROUP_CONCAT(DISTINCT ?productLabel; separator = "| | |") AS ?products)
    WHERE {
        {
            # Find the exchange and its ticker
            ?id wdt:P414 ?exchange .
            ?id p:P414 ?exchangesub .
            ?exchangesub pq:P249 ?ticker . FILTER(UCASE(STR(?ticker)) = '"""+symbol+"""') .
            OPTIONAL { ?id rdfs:label ?idLabel . FILTER (LANG(?idLabel) = "en") }
        }
        OPTIONAL {
            ?id skos:altLabel ?altLabel .
            FILTER (LANG(?altLabel) = "en")
        }
        OPTIONAL {
            ?id wdt:P452 ?industry .
            ?industry rdfs:label ?industryLabel .
            FILTER (LANG(?industryLabel) = "en")
        }
        OPTIONAL {
            ?id wdt:P1056 ?product .
            ?product rdfs:label ?productLabel .
            FILTER (LANG(?productLabel) = "en")
        }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    GROUP BY ?ticker ?id
    """

    query_2 = """
    SELECT ?ticker ?id
        (GROUP_CONCAT(DISTINCT ?idLabel;separator="| | |") AS ?idLabels)
        (GROUP_CONCAT(DISTINCT CONCAT(?subsidiaryLabel, 
            IF(BOUND(?start_time), CONCAT(" (Start: ", STR(?start_time), ")"), ""), 
            IF(BOUND(?end_time), CONCAT(" (End: ", STR(?end_time), ")"), "")
        );separator="| | |") AS ?subsidiaries)
        (GROUP_CONCAT(DISTINCT CONCAT(?ownerOfLabel, 
            IF(BOUND(?start_time_owner), CONCAT(" (Start: ", STR(?start_time_owner), ")"), ""), 
            IF(BOUND(?end_time_owner), CONCAT(" (End: ", STR(?end_time_owner), ")"), "")
        );separator="| | |") AS ?ownedEntities)
    WHERE {
        {
            # Find the exchange and its ticker 
            ?id wdt:P414 ?exchange . 
            ?id p:P414 ?exchangesub .
            ?exchangesub pq:P249 ?ticker . FILTER(UCASE(STR(?ticker)) = '"""+symbol+"""') .
            OPTIONAL { ?id rdfs:label ?idLabel . FILTER (LANG(?idLabel) = "en") }
        }
        OPTIONAL {
            ?id wdt:P355 ?subsidiary .
            ?subsidiary rdfs:label ?subsidiaryLabel .
            FILTER (LANG(?subsidiaryLabel) = "en")
            OPTIONAL { ?id p:P355 [ps:P355 ?subsidiary; pq:P580 ?start_time; pq:P582 ?end_time] }
        }
        OPTIONAL {
            ?id wdt:P1830 ?ownerOf .
            ?ownerOf rdfs:label ?ownerOfLabel .
            FILTER (LANG(?ownerOfLabel) = "en")
            OPTIONAL { ?id p:P1830 [ps:P1830 ?ownerOf; pq:P580 ?start_time_owner; pq:P582 ?end_time_owner] }
        }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    GROUP BY ?ticker ?id
    """

    query_3 = """
    SELECT ?ticker ?id
        (GROUP_CONCAT(DISTINCT ?ceoLabel;separator="| | |") AS ?ceos)
        (GROUP_CONCAT(DISTINCT CONCAT(?ceoLabel, 
            IF(BOUND(?ceoStart), CONCAT(" (Start: ", STR(?ceoStart), ")"), ""), 
            IF(BOUND(?ceoEnd), CONCAT(" (End: ", STR(?ceoEnd), ")"), "")
        );separator="| | |") AS ?ceosWithTerms)
        (GROUP_CONCAT(DISTINCT ?boardMemberLabel;separator="| | |") AS ?boardMembers)
        (GROUP_CONCAT(DISTINCT CONCAT(?boardMemberLabel, 
            IF(BOUND(?boardMemberStart), CONCAT(" (Start: ", STR(?boardMemberStart), ")"), ""), 
            IF(BOUND(?boardMemberEnd), CONCAT(" (End: ", STR(?boardMemberEnd), ")"), "")
        );separator="| | |") AS ?boardMembersWithTerms)
        (GROUP_CONCAT(DISTINCT CONCAT(?legalFormLabel, 
            IF(BOUND(?legalFormStart), CONCAT(" (Start: ", STR(?legalFormStart), ")"), ""), 
            IF(BOUND(?legalFormEnd), CONCAT(" (End: ", STR(?legalFormEnd), ")"), "")
        );separator="| | |") AS ?legalFormsWithDates)
        (SAMPLE(?shortName) AS ?shortNames)
    WHERE {
        {
            # Find the exchange and its ticker 
            ?id wdt:P414 ?exchange . 
            ?id p:P414 ?exchangesub .
            ?exchangesub pq:P249 ?ticker . FILTER(UCASE(STR(?ticker)) = '"""+symbol+"""') .
        }
        OPTIONAL {
            ?id p:P169 ?ceoStatement .
            ?ceoStatement ps:P169 ?ceo .
            ?ceo rdfs:label ?ceoLabel .
            FILTER (LANG(?ceoLabel) = "en")
            OPTIONAL { ?ceoStatement pq:P580 ?ceoStart }
            OPTIONAL { ?ceoStatement pq:P582 ?ceoEnd }
        }
        OPTIONAL {
            ?id p:P3320 ?boardMemberStatement .
            ?boardMemberStatement ps:P3320 ?boardMember .
            ?boardMember rdfs:label ?boardMemberLabel .
            FILTER (LANG(?boardMemberLabel) = "en")
            OPTIONAL { ?boardMemberStatement pq:P580 ?boardMemberStart }
            OPTIONAL { ?boardMemberStatement pq:P582 ?boardMemberEnd }
        }
        OPTIONAL {
            ?id wdt:P1454 ?legalForm .
            ?legalForm rdfs:label ?legalFormLabel .
            FILTER (LANG(?legalFormLabel) = "en")
            OPTIONAL {
                ?id p:P1454 ?legalFormStatement .
                ?legalFormStatement ps:P1454 ?legalForm .
                OPTIONAL { ?legalFormStatement pq:P580 ?legalFormStart }
                OPTIONAL { ?legalFormStatement pq:P582 ?legalFormEnd }
            }
        }
        OPTIONAL {
            ?id wdt:P1813 ?shortName .
            FILTER (LANG(?shortName) = "en")
        }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    GROUP BY ?ticker ?id
    """



    # ... (rest of your code where you define endpoint_url, symbol, and the modified query_2)

    # Initialize a results list
    all_results = []

    # Send the requests and process the responses
    response_1 = requests.get(endpoint_url, params={'query': query_1, 'format': 'json'})
    response_2 = requests.get(endpoint_url, params={'query': query_2, 'format': 'json'})
    response_3 = requests.get(endpoint_url, params={'query': query_3, 'format': 'json'})

    if response_1.ok and response_2.ok and response_3.ok:
        # Process the first query results
        data_1 = response_1.json()
        data_2 = response_2.json()
        data_3 = response_3.json()
        
        # Assuming both queries return the same number of rows for each company
        for i in range(len(data_1['results']['bindings'])):
            result_1 = data_1['results']['bindings'][i]
            result_2 = data_2['results']['bindings'][i]
            result_3 = data_3['results']['bindings'][i]
            
            entry = {
                'id_label': result_1['idLabels']['value'],
                'ticker': result_1['ticker']['value'],
                'industry': result_1['industries']['value'].split('| | |') if 'industries' in result_1 else [],
                'aliases': result_1['aliases']['value'].split('| | |') if 'aliases' in result_1 else [],
                'products': result_1['products']['value'].split('| | |') if 'products' in result_1 else [],
                'subsidiaries': result_2['subsidiaries']['value'].split('| | |') if 'subsidiaries' in result_2 else [],
                'owned_entities': result_2['ownedEntities']['value'].split('| | |') if 'ownedEntities' in result_2 else [],
                'ceos': result_3['ceosWithTerms']['value'].split('| | |') if 'ceosWithTerms' in result_3 else [],
                'board_members': result_3['boardMembersWithTerms']['value'].split('| | |') if 'boardMembersWithTerms' in result_3 else [],
                # 'legal_forms': result_3['legalFormsWithDates']['value'].split('| | |') if 'legalFormsWithDates' in result_3 else [],
            }
            
            all_results.append(entry)

        # Now we save the combined results to a JSON file
        with open(f'info/{symbol}_info.json', 'w') as f:
            json.dump(all_results, f, indent=4)
    else:
        print("Error in fetching data")

        
# Read the list of symbols from a CSV file
ticker_df = pd.read_csv('sp500list.csv')
symbol_list = ticker_df['Symbol'].tolist()



def worker_task(symbols, lock):
    while True:
        lock.acquire()
        try:
            if len(symbols) == 0:
                break  # If the list is empty, exit the loop
            symbol = symbols.pop()
            print(symbol)  # Pop the last URL from the shared list
        finally:
            lock.release()
        
        # Now that we have a URL, we can scrape it
        query_wikidata(symbol)
    

num_process = 2

if __name__ == "__main__":
    manager = Manager()
    symbols = manager.list(symbol_list)  # Shared list of URLs
    lock = manager.Lock()  # A lock to prevent simultaneous access to the list

    with Pool(processes=num_process) as pool:
        # Start the worker task for each process
        for _ in range(num_process):
            pool.apply_async(worker_task, args=(symbols, lock))

        pool.close()  # Close the pool to any new tasks
        pool.join()  # Wait for all worker processes to finish