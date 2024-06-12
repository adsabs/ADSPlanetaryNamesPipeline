import requests

from adsputils import setup_logging

logger = setup_logging('utils')

from adsplanetnamepipe.utils.common import EntityArgs


class LocalLLM():

    def __init__(self, args: EntityArgs):
        """

        :param args:
        """
        self.args = args

    def forward(self, title, abstract, excerpt):
        """

        :param title:
        :param abstract:
        :param excerpt:
        :return:
        """
        if abstract:
            url = 'https://playground.adsabs.harvard.edu/brain/v1/chat'
            headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer %s'%'edge!'}
            title = ' '.join(title)
            content = f'Consider the following scientific article:\n\n' \
                      f'Title: {title}\n\n' \
                      f'Abstract: {abstract}\n\n' \
                      f'Task: Please analyze the following excerpt: {excerpt}\n\n' \
                      f'Based on the context provided, answer what is the probablity (just give a value between 0 and 1), with the limited information, '\
                      f'that the term "{self.args.feature_name}" refers to a "{self.args.feature_type}" on the {self.args.target}?\n\n '
            json_data = {'system': 'This is a system prompt, please behave and help the user',
                         'conversation': [ { 'role': 'user', 'content': content } ] }
            response = requests.post(url=url, headers=headers, json=json_data)
            if response.status_code == 200:
                answer = response.json().get('text', '').strip()
                try:
                    answer = float(answer)
                    return min(1, max(0, answer))
                except ValueError:
                    return 0
            else:
                logger.error(f"From Brain status code {response.status_code}")

        return 0
