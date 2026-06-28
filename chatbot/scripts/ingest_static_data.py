from pathlib import Path

from weaviate import WeaviateClient
from weaviate.classes.config import Configure, Property, DataType
from weaviate.collections import Collection

from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from chatbot.settings import Settings, get_settings
from chatbot.utils.weaviate_utils import get_weaviate_client

__CATEGORIES = {
    "General Information": "general",
    "Locations": "general",
    "Parking Details": "general",
    "Booking Process": "booking",
    "Rules & Policies": "policies",
    "Questions & Answers": "faq",
    "Contact & Support": "general",
}


class UndefinedCategoryException(BaseException):
    pass


def define_category(header: str) -> str:
    for h, category in __CATEGORIES.items():
        if h in header:
            return category
    raise UndefinedCategoryException(f"Failed to define category for [{header}]")


def split_document(
    md_file_path: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    md_file_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "h2"), ("##", "h3")]
    )
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
        header = " -> ".join(section.metadata.values())
        category = define_category(header)
        doc_chunks = text_splitter.split_documents([section])

        for d in doc_chunks:
            d.metadata["doc_file"] = str(md_file_path)
            d.metadata["header"] = header
            d.metadata["category"] = category
            all_documents.append(d)

    return all_documents


def parse_files(input_data_path: str, settings: Settings) -> list[Document]:
    path = Path(input_data_path).resolve()

    # volume ?
    return [
        chunk
        for fp in path.glob("*.md")
        for chunk in split_document(
            fp,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
    ]


def create_collection(w: WeaviateClient, collection_name: str) -> Collection:
    return w.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.text2vec_transformers(),
        reranker_config=Configure.Reranker.transformers(),
        properties=[
            Property(
                name="doc_file",
                data_type=DataType.TEXT,
                description="Source document file name",
            ),
            Property(
                name="header",
                data_type=DataType.TEXT,
                description="Chuck header/path",
            ),
            Property(
                name="content",
                data_type=DataType.TEXT,
                description="File chuck content",
            ),
            Property(
                name="category",
                data_type=DataType.TEXT,
                description="File chuck category",
            ),
        ],
    )


def load_documents_in_vector_store(
    vs: WeaviateVectorStore, documents: list[Document]
) -> None:
    vs.add_documents(documents)


def load_documents_in_collection(collection: Collection, documents: list[Document]):
    with collection.batch.dynamic() as batch:
        for document in documents:
            batch.add_object(
                properties={
                    "content": document.page_content,
                    **document.metadata,
                }
            )


def delete_collection(w: WeaviateClient, collection_name: str) -> None:
    w.collections.delete(collection_name)


def load_static_data(settings: Settings) -> None:

    with get_weaviate_client(settings) as w_client:
        collection = settings.weaviate_collection
        if w_client.collections.exists(collection):
            w_client.collections.delete(collection)
        w_collection = create_collection(w_client, collection)

        documents = parse_files(settings.weaviate_init_data_path, settings)

        load_documents_in_collection(w_collection, documents)


def main():
    settings = get_settings()
    load_static_data(settings)


if __name__ == "__main__":
    main()
