import numpy as np
import toml
from flask import Flask, request, render_template
import pandas as pd
from openai.embeddings_utils import cosine_similarity

from csv_loader import CsvLoader
from data_load import DataLoad
from openai_embedded_client import EmbeddedClient
from openai_moderation_client import ModerationClient
from pinecone_client import PineconeClient
from search import Search

app = Flask(__name__)

# Read local `config.ini` file.
config = toml.load("config.toml")
# Get values from our .ini file
openai_api_key = config.get("openai").get("openai_api_key")
openai_embedded_search_model = config.get("openai").get("openai_embedded_search_model")
openai_moderation_model = config.get("openai").get("openai_moderation_model")
pinecone_dimension = config.get("openai").get("pinecone_dimension")
pinecone_api_key = config.get("database").get("pinecone_api_key")
pinecone_index_name = config.get("database").get("pinecone_index")
pinecone_environment = config.get("database").get("pinecone_environment")


@app.route('/load_data')
def load():
    """
    Loads data from a CSV file and upserts embeddings into a Pinecone index.

    Args:
        file (str, optional): The name of the CSV file to load. Defaults to 'sample.csv'.

    Returns:
        str: A message indicating that the function has completed successfully.

    Raises:
        ValueError: If the file does not exist or is not a valid CSV file.

    Example:
        To load data from a file called "mydata.csv":
        ```
        http://localhost:5000/load_data?file=mydata.csv
        ```
    """
    pinecone_client = PineconeClient(pinecone_index_name, pinecone_environment, pinecone_api_key, pinecone_dimension)
    openai_embedded_client = EmbeddedClient(openai_api_key, openai_embedded_search_model)
    csv_loader = CsvLoader('sample.csv')
    success = DataLoad.load_data(csv_loader, openai_embedded_client, pinecone_client, )
    return success




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
    Route to handle search queries from the user.

    Parameters:
        None.

    Returns:
        A rendered HTML template with search results and information about the search query.

    """
    # Get the query string from the URL parameters
    query = request.args.get('query')

    # Create OpenAI moderation and embedding clients
    openai_moderation_client = ModerationClient(openai_api_key, openai_moderation_model)
    openai_embedded_client = EmbeddedClient(openai_api_key, openai_embedded_search_model)

    # Create Pinecone client for similarity search
    pine_cone_client = PineconeClient(pinecone_index_name, pinecone_environment, pinecone_api_key, pinecone_dimension)

    # Search for similar results using the query and clients created above
    is_safe, moderation_classification, results = Search.search(query, openai_moderation_client, openai_embedded_client,
                                                                pine_cone_client)

    # Render the search results using an HTML template
    return render_template('search_results.html', query=query, is_safe=is_safe,
                           moderation_classification=moderation_classification, results=results)


if __name__ == '__main__':
    app.run()
