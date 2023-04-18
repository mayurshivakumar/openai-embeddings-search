import openai


class EmbeddedClient:
    """
    A class representing a client for creating embeddings using OpenAI's API.

    Args:
        api_key (str): The API key to use for the OpenAI client.
        search_model (str): The name of the OpenAI model to use for creating embeddings.

    Attributes:
        model (str): The name of the OpenAI model being used for creating embeddings.
        labels (list): A list of labels to use for prediction.

    Methods:
        create: Creates an embedding for a given text using the OpenAI API.
        get_label_embeddings: Gets the embeddings for the label texts.
        get_labels: Gets the labels for the embeddings.

    """

    def __init__(self, api_key: str, search_model: str):
        """
        Initializes a new instance of the EmbeddedClient class.

        Args:
            api_key (str): The API key to use for the OpenAI client.
            search_model (str): The name of the OpenAI model to use for creating embeddings.
        """
        openai.api_key = api_key
        self.model = search_model
        self.labels = ['negative', 'positive']

    def create(self, text: str) -> list:
        """
        Creates an embedding for a given text using the OpenAI API.

        Args:
            text (str): The text to create an embedding for.

        Returns:
            list: A list containing the embedding for the input text.
        """
        response = openai.Embedding.create(
            input=text,
            model=self.model
        )
        return response['data'][0]['embedding']

    def get_label_embeddings(self) -> list:
        """
        Gets the embeddings for the label texts.

        Returns:
            list: A list of embeddings for the label texts.
        """
        label_embeddings = [self.create(i) for i in self.labels]
        return label_embeddings

    def get_labels(self):
        """
        Gets the labels for the embeddings.

        Returns:
            list: A list of labels for the embeddings.
        """
        return self.labels
