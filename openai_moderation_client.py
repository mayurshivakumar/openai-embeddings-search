import openai


class ModerationClient:
    def __init__(self, api_key: str, model: str):
        openai.api_key = api_key
        self.model = model

    def create(self, text: str):
        response = openai.Moderation.create(input=text)
        return response['results'][0]

    def is_safe(self, text: str):
        response = self.create(text)
        if response.get('flagged'):
            if response.get('categories').get('hate'):
                return False, 'hate'
            elif response.get('categories').get('hate/threatening'):
                return False, 'hate/threatening'
            elif response.get('categories').get('self-harm'):
                return False, 'self-harm'
            elif response.get('categories').get('sexual'):
                return False, 'sexual'
            elif response.get('categories').get('sexual/minors'):
                return False, 'sexual/minors'
            elif response.get('categories').get('violence'):
                return False, 'violence'
            elif response.get('categories').get('violence/graphic'):
                return False, 'violence/graphic'
        return True, ''
