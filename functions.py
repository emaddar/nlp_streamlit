import pandas as pd


def process_ner_results(ner_results):
    # create dataframe from ner_results
    df = pd.DataFrame(ner_results)

    # add new column with entity after '-'
    df['entity_type'] = df['entity'].apply(lambda x: x.split('-')[1])
    # delete original entity column
    df = df.drop(columns=['entity'])

    # rename entity_type column to entity
    df = df.rename(columns={'entity_type': 'entity'})


    # define desired column order
    new_column_order = ['entity', 'word', 'score', 'index', 'start', 'end']

    # reindex columns in desired order
    df = df.reindex(columns=new_column_order)

    # group by entity and get count
    count_df = df.groupby('entity').count().reset_index()[['entity', 'score']]
    count_df = count_df.rename(columns={'score': 'count'})

    return df, count_df




import urllib.request
import json
import os
import ssl
from dotenv import load_dotenv
import pandas as pd
load_dotenv()

def ar_hugging_face(input_text):
    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

    # Request data goes here
    # The example below assumes JSON formatting which may be updated
    # depending on the format your endpoint expects.
    # More information can be found here:
    # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
    data = {"inputs" : input_text}

    body = str.encode(json.dumps(data))

    url = 'https://arabic-ner-srpel9nm.westeurope.inference.ml.azure.com/score'
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = os.getenv('api_key')
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'main' }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        # print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        # print(error.info())
        # print(error.read().decode("utf8", 'ignore'))

    json_result = json.loads(result.decode('utf-8'))
    df = pd.DataFrame.from_dict(json_result)
    df['entity'] = df['entity'].apply(lambda x: x.split('-')[1])
    groupby_entity_counts = df.groupby('entity').size().reset_index(name='count')
    return df, groupby_entity_counts