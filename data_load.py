from csv_loader import CsvLoader
from openai_embedded_client import EmbeddedClient
from pinecone_client import PineconeClient
from openai.embeddings_utils import cosine_similarity
import numpy as np


class DataLoad:
    @staticmethod
    def load_data(csv_loader: CsvLoader, embedded: EmbeddedClient, pinecone: PineconeClient) -> str:
        """
        Load data from a CSV file into Pinecone, using OpenAI's embedding service to create embeddings
        for each row of text and predict a label for each row.

        Args:
            csv_loader (CsvLoader): An instance of CsvLoader containing the CSV file to load.
            embedded (EmbeddedClient): An instance of EmbeddedClient to use for creating text embeddings.
            pinecone (PineconeClient): An instance of PineconeClient to use for upserting vectors.

        Returns:
            str: A message indicating the success or failure of the data loading process.
        """
        csv_loader.read_csv()
        label_embeddings = embedded.get_label_embeddings()
        labels = embedded.get_labels()

        count = 1
        for _, row in csv_loader.get_data().iterrows():
            embeds = embedded.create(row['text'])
            sim = [cosine_similarity(embeds, i) for i in label_embeddings]
            prediction = labels[np.argmax(sim)]
            to_upsert = zip([str(count)], [embeds], [{'text': row['text'], 'prediction': prediction}])
            pinecone.upsert_vectors(list(to_upsert))
            count = count + 1
        # TODO return properly
        return 'success'
