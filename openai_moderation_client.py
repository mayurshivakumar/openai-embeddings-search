import openai

class ModerationClient:
    """
    A client for the OpenAI Moderation API, which detects potentially harmful content in text.

    Attributes:
        api_key (str): The API key for the OpenAI API.
        model (str): The name of the OpenAI model to use for moderation.
    """

    def __init__(self, api_key: str, model: str):
        """
        Initializes a new ModerationClient object.

        Args:
            api_key (str): The API key for the OpenAI API.
            model (str): The name of the OpenAI model to use for moderation.
        """
        openai.api_key = api_key
        self.model = model

    def create(self, text: str):
        """
        Sends a request to the OpenAI Moderation API to moderate a given text.

        Args:
            text (str): The text to moderate.

        Returns:
            A dictionary containing the moderation results.
        """
        response = openai.Moderation.create(input=text)
        return response['results'][0]

    def is_safe(self, text: str):
        """
        Checks whether a given text is safe or potentially harmful.

        Args:
            text (str): The text to check.

        Returns:
            A tuple (is_safe, category), where is_safe is a boolean indicating whether the text is safe,
            and category is a string indicating the category of the harmful content, if any.
            If the text is safe, category is an empty string.
        """
        response = self.create(text)
        categories = response.get('categories', {})
        for category in categories:
            if categories[category]:
                return False, category
        return True, ''
