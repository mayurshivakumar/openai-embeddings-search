import toml
from flask import Flask, request, render_template
import pandas as pd

from openai_client import EmbeddedClient1
from pinecone_client import PineconeClient

app = Flask(__name__)

# Read local `config.ini` file.
config = toml.load("config.toml")
# Get values from our .ini file
openai_api_key = config.get("openai").get("openai_api_key")
openai_embedded_search_model = config.get("openai").get("openai_embedded_search_model")
pinecone_dimension = config.get("openai").get("pinecone_dimension")
pinecone_api_key = config.get("database").get("pinecone_api_key")
pinecone_index_name = config.get("database").get("pinecone_index")
pinecone_environment = config.get("database").get("pinecone_environment")


@app.route('/load_data')
def load():
    """
    Loads data from a CSV file and upserts embeddings into a Pinecone index.

    Returns:
        str: An done  indicating that the function has completed successfully.
    """
    pine_client = PineconeClient(pinecone_index_name, pinecone_environment, pinecone_api_key, pinecone_dimension)
    openai_embedded_client = EmbeddedClient1(openai_api_key, openai_embedded_search_model)
    file_name = request.args.get('file')
    if file_name is None:
        file_name = 'sample.csv'
    csv = pd.read_csv(file_name)

    count = 1
    for _, row in csv.iterrows():
        embeds = openai_embedded_client.create(row['text'])
        to_upsert = zip([str(count)], [embeds], [{'text': row['text']}])
        pine_client.upsert_vectors(list(to_upsert))
        count = count + 1
    return 'done'


# TODO add a endpoint to upload a csv file

@app.route('/')
def search_form():
    """
    Displays the search form page.

    Returns:
        str: The rendered HTML for the search form page.
    """
    return render_template('search_form.html')


@app.route('/search')
def search():
    """
    Performs a search for embeddings similar to a given query using an OpenAI client and a Pinecone index.

    Returns:
        str: The rendered HTML for the search results page, which includes the query and a list of matching results.
    """
    query = request.args.get('query')

    openai_embedded_client = EmbeddedClient1(openai_api_key, openai_embedded_search_model)
    embeds = openai_embedded_client.create(query)

    pine_client = PineconeClient(pinecone_index_name, pinecone_environment, pinecone_api_key, pinecone_dimension)

    matches = pine_client.query(embeds)
    results = []
    for txt in matches:
        results.append(txt['metadata']['text'])

    return render_template('search_results.html', query=query, results=results)


if __name__ == '__main__':
    app.run()
