from dataclasses import dataclass

from openai_moderation_client import ModerationClient
from openai_embedded_client import EmbeddedClient
from pinecone_client import PineconeClient


@dataclass
class Matches:
    text: str
    prediction: str


class Search:
    @staticmethod
    def search(
               query: str,
               moderation: ModerationClient,
               embedded: EmbeddedClient,
               pinecone: PineconeClient) -> (str, str, Matches):
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
            matches.append(
                m
            )
        return is_safe, moderation_classification, matches
