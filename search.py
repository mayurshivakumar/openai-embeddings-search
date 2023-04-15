from dataclasses import dataclass
from typing import List, Tuple

from openai_moderation_client import ModerationClient
from openai_embedded_client import EmbeddedClient
from pinecone_client import PineconeClient


@dataclass
class Matches:
    """
    Represents a match found during a search.

    Attributes:
        text (str): The text of the match.
        prediction (str): The prediction associated with the match.
    """
    text: str
    prediction: str


class Search:
    @staticmethod
    def search(query: str,
               moderation: ModerationClient,
               embedded: EmbeddedClient,
               pinecone: PineconeClient) -> Tuple[bool, str, List[Matches]]:
        """
        Search for matches to a query using a moderation client, an embedded client, and a Pinecone client.

        Args:
            query (str): The query string to search for.
            moderation (ModerationClient): The moderation client to use for checking whether the query is safe.
            embedded (EmbeddedClient): The embedded client to use for creating embeddings of the query.
            pinecone (PineconeClient): The Pinecone client to use for searching for matches.

        Returns:
            A tuple containing a boolean value indicating whether the query is safe, a moderation classification string,
            and a list of Matches objects representing the matches found.
        """
        is_safe, moderation_classification = moderation.is_safe(query)
        matches = []
        if not is_safe:
            return is_safe, moderation_classification, matches

        embeds = embedded.create(query)
        result = pinecone.query(embeds)

        for txt in result:
            text = str(txt['metadata']['text'])
            prediction = str(txt['metadata']['prediction'])

            m = Matches(text, prediction)
            matches.append(m)

        return is_safe, moderation_classification, matches
