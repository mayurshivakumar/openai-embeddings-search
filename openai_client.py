import openai


class EmbeddedClient1:
    """
    A class representing a client for creating embeddings using OpenAI's API.

    Args:
        api_key (str): The API key to use for the OpenAI client.
        search_model (str): The name of the OpenAI model to use for creating embeddings.

    Attributes:
        search_model (str): The name of the OpenAI model being used for creating embeddings.

    Methods:
        create: Creates an embedding for a given text using the OpenAI API.

    """

    def __init__(self, api_key: str, search_model: str):
        """
        Initializes a new instance of the EmbeddedClient1 class.

        Args:
            api_key (str): The API key to use for the OpenAI client.
            search_model (str): The name of the OpenAI model to use for creating embeddings.
        """
        openai.api_key = api_key
        self.search_model = search_model

    def create(self, text: str):
        """
        Creates an embedding for a given text using the OpenAI API.

        Args:
            text (str): The text to create an embedding for.

        Returns:
            list: A list containing the embedding for the input text.
        """
        response = openai.Embedding.create(
            input=text,
            model=self.search_model
        )
        return response['data'][0]['embedding']
