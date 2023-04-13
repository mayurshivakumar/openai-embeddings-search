import numpy as np
import toml
from flask import Flask, request, render_template
import pandas as pd
from openai.embeddings_utils import cosine_similarity

from openai_embedded_client import EmbeddedClient
from openai_moderation_client import ModerationClient
from pinecone_client import PineconeClient

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
    pine_client = PineconeClient(pinecone_index_name, pinecone_environment, pinecone_api_key, pinecone_dimension)
    openai_embedded_client = EmbeddedClient(openai_api_key, openai_embedded_search_model)

    file_name = request.args.get('file')
    if file_name is None:
        file_name = 'sample.csv'

    # Create embeddings for positive and negative labels
    labels = ['negative', 'positive']
    label_embeddings = [openai_embedded_client.create(i) for i in labels]

    # Load CSV file and upsert embeddings to Pinecone index
    try:
        csv = pd.read_csv(file_name)
    except FileNotFoundError:
        raise ValueError(f"File '{file_name}' does not exist")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File '{file_name}' is empty or not a valid CSV file")

    count = 1
    for _, row in csv.iterrows():
        embeds = openai_embedded_client.create(row['text'])
        sim = [cosine_similarity(embeds, i) for i in label_embeddings]
        prediction = labels[np.argmax(sim)]
        to_upsert = zip([str(count)], [embeds], [{'text': row['text'], 'prediction': prediction}])
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
    openai_moderation_client = ModerationClient(openai_api_key, openai_moderation_model)
    is_safe, moderation_classification = openai_moderation_client.is_safe(query)
    if not is_safe:
        return render_template('search_results.html',
                               query=query,
                               is_safe=is_safe,
                               moderation_classification=moderation_classification
                               )

    openai_embedded_client = EmbeddedClient(openai_api_key, openai_embedded_search_model)
    embeds = openai_embedded_client.create(query)

    pine_client = PineconeClient(pinecone_index_name, pinecone_environment, pinecone_api_key, pinecone_dimension)

    matches = pine_client.query(embeds)
    results = []
    for txt in matches:
        results.append(
            {
                'text': txt['metadata']['text'],
                'prediction': txt['metadata']['prediction']
            }
        )

    return render_template('search_results.html', query=query, is_safe=is_safe, results=results)


if __name__ == '__main__':
    app.run()
