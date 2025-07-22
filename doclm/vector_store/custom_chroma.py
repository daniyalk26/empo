import uuid
from abc import ABC
import logging
from typing import Any, Dict, Iterable, List, Optional

from langchain.vectorstores.chroma import Chroma
from .base import Base

log = logging.getLogger("doclogger")
log.disabled = False


class CustomChroma(Chroma, Base, ABC):
    def delete_file(self, files: List[Dict]) -> None:
        for item in files:
            doc_ids = self._collection.get(where={"source": item[self.file_tag]})["ids"]
            if not doc_ids:
                log.warning("%s not found", item[self.file_tag])
                continue
            self.delete(doc_ids)

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        if self._embedding_function is not None:
            embeddings = []
            texts = list(texts)
            for i in range(0, len(texts), 16):
                embeddings += self._embedding_function.embed_documents(texts[i: i + 16])
            # embeddings = self._embedding_function.embed_documents(list(texts))

        # texts, metadatas = self.encrypt_data(texts, metadatas)
        self._collection.upsert(
            metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids
        )
        return ids
