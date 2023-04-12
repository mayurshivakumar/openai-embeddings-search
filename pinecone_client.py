import pinecone


class PineconeClient:
    """
    A class representing a client for interacting with a Pinecone index.

    Args:
        index_name (str): The name of the Pinecone index to use or create.
        env (str): The environment to use for the Pinecone client.
        api_key (str): The API key to use for the Pinecone client.
        dimension (int): The dimensionality of the vectors to be stored in the index.

    Attributes:
        index_name (str): The name of the Pinecone index being used.
        env (str): The environment being used for the Pinecone client.
        api_key (str): The API key being used for the Pinecone client.
        dimension (int): The dimensionality of the vectors being stored in the index.

    Methods:
        create_index_if_not_exists: Creates a new Pinecone index if one with the given name does not already exist.
        upsert_vectors: Upserts vectors into the Pinecone index.
        query: Queries the Pinecone index for the top-k most similar vectors to a given input vector.

    """

    def __init__(self, index_name: str, env: str, api_key: str, dimension: int):
        self.index_name = index_name
        self.env = env
        self.api_key = api_key
        self.dimension = dimension
        self.create_index_if_not_exists()

    def create_index_if_not_exists(self):
        """
        Creates a new Pinecone index if one with the given name does not already exist.
        """
        pinecone.init(
            api_key=self.api_key,
            environment=self.env
        )
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=self.dimension)

    def upsert_vectors(self, vectors):
        """
        Upserts vectors into the Pinecone index.

        Args:
            vectors (list): A list of vectors to upsert into the index.
        """
        pinecone_index = pinecone.Index(self.index_name)
        pinecone_index.upsert(vectors=vectors)

    def query(self, embeds):
        """
        Queries the Pinecone index for the top-k most similar vectors to a given input vector.

        Args:
            embeds (list): A list of vectors to query the index for.

        Returns:
            list: A list of the top-k most similar vectors to the input vector, along with their metadata.
        """
        pinecone_index = pinecone.Index(self.index_name)
        res = pinecone_index.query([embeds], top_k=5, include_metadata=True)
        return res['matches']
