from pathlib import Path

from weaviate import WeaviateClient
from weaviate.classes.config import Configure, Property, DataType
from weaviate.collections import Collection

from chatbot.settings import Settings, get_settings
from chatbot.utils.weaviate_utils import get_weaviate_client, get_weaviate_vector_store

from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


__CATEGORIES = {
    "General Information": "general",
    "Locations": "general",
    "Parking Details": "general",
    "Booking Process": "booking",
    "Rules & Policies": "policies",
    "Questions & Answers": "faq",
    "Contact & Support": "general",
}


def create_collection(
    w: WeaviateVectorStore, collection_name: str
) -> Collection:
    return w.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(
                name="content",
                data_type=DataType.TEXT,
                description="File chuck content",
            ),
            Property(
                name="content_category",
                data_type=DataType.TEXT,
                description="File chuck category",
            ),
        ]
    )


def load_documents(
    vs: WeaviateVectorStore, input_data_path: str, settings: Settings
) -> None:
    path = Path(input_data_path).resolve()

    for f in path.glob("*.md"):
        documents = split_document(
            f,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        vs.add_documents(documents)


def split_document(
    md_file_path: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    md_file_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "h2"), ("##", "h3")])
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    sections = md_file_splitter.split_text(content)

    all_documents: list[Document] = []
    for section in sections:
        # todo refactor
        category = __CATEGORIES[section.metadata["h2"]]
        doc_chunks = text_splitter.split_documents([section])

        for d in doc_chunks:
            d.metadata["category"] = category
            all_documents.append(d)

    return all_documents


def delete_collection(w: WeaviateClient, collection_name: str) -> None:
    w.collections.delete(collection_name)


def main():
    settings = get_settings()
    w = get_weaviate_client(settings)
    collection = settings.weaviate_collection
    try:
        if w.collections.exists(collection):
            w.collections.delete(collection)
        _ = create_collection(w, collection)
        vs = get_weaviate_vector_store(w, settings)
        load_documents(vs, settings.weaviate_init_data_path, settings)
    finally:
        w.close()


if __name__ == "__main__":
    main()
