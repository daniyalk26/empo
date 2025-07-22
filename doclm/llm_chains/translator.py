from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from ..templates.default import  TRANSLATION_PROMPT # Improve this Import


class Translator:
    def __init__(self):
        from . import get_translation_model # To resolve issue of circular import
        self.model = get_translation_model(name='gpt4')
        self.translation_chain = self.create_translation_chain(TRANSLATION_PROMPT)

    def create_translation_chain(self, prompt):
        """Setup the translation chain with a prompt template and output parser."""
        output_parser = StrOutputParser()
        return LLMChain(
            llm=self.model,
            prompt=prompt,
            output_parser=output_parser,
            output_key="translation"
        )

    def translate(self, text, language="English"):
        """Translate text to the specified language using the translation chain."""
        try:
            result = self.translation_chain.run({"text": text, "target_language": language})
            return result
        except Exception as e:
            print(f"An error occurred during translation: {e}")
            return None
