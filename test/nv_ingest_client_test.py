import src.utils
from typing import List
import logging
import nv_ingest_client
import os
import pandas as pd

logger = logging.getLogger(__name__)

ENABLE_NV_INGEST_VDB_UPLOAD=True

def get_nv_ingest_ingestor(
        nv_ingest_client_instance,
        filepaths: List[str],
        **kwargs
    ):
    """
    Prepare NV-Ingest ingestor instance based on nv-ingest configuration

    Returns:
        - ingestor: Ingestor - NV-Ingest ingestor instance with configured tasks
    """
    config = src.utils.get_config()

    logger.info("Preparing NV Ingest Ingestor instance for filepaths: %s", filepaths)
    # Prepare the ingestor using nv-ingest-client
    ingestor = nv_ingest_client.Ingestor(client=nv_ingest_client_instance)

    # Add files to ingestor
    ingestor = ingestor.files(filepaths)
    
    # Add metadata to files, ingestor requires a dataframe
    meta_df = pd.DataFrame(
        {
            "source": filepaths,
            #"meta_a": kwargs.get("custom_metadata")
            **kwargs.get("custom_metadata", {})
        }
    )
    metadata_dict=kwargs.get("custom_metadata")
    metadata_keys=list(metadata_dict.keys())

    # Add extraction task
    extraction_options = kwargs.get("extraction_options", {})
    ingestor = ingestor.extract(
                    extract_text=extraction_options.get("extract_text", config.nv_ingest.extract_text),
                    extract_tables=extraction_options.get("extract_tables", config.nv_ingest.extract_tables),
                    extract_charts=extraction_options.get("extract_charts", config.nv_ingest.extract_charts),
                    extract_images=extraction_options.get("extract_images", config.nv_ingest.extract_images),
                    extract_method=extraction_options.get("extract_method", config.nv_ingest.extract_method),
                    text_depth=extraction_options.get("text_depth", config.nv_ingest.text_depth),
                )

    # Add splitting task (By default only works for text documents)
    split_options = kwargs.get("split_options", {})
    ingestor = ingestor.split(
                    tokenizer=config.nv_ingest.tokenizer,
                    chunk_size=split_options.get("chunk_size", config.nv_ingest.chunk_size),
                    chunk_overlap=split_options.get("chunk_overlap", config.nv_ingest.chunk_overlap),
                )

    # Add captioning task if extract_images is enabled
    if extraction_options.get("extract_images", config.nv_ingest.extract_images):
        logger.info("Adding captioning task to NV-Ingest Ingestor")
        ingestor = ingestor.caption(
                        api_key=src.utils.get_env_variable(variable_name="NVIDIA_API_KEY", default_value=""),
                        endpoint_url=config.nv_ingest.caption_endpoint_url,
                        model_name=config.nv_ingest.caption_model_name,
                    )

    # Add Embedding task
    if ENABLE_NV_INGEST_VDB_UPLOAD:
        ingestor = ingestor.embed()

    # Add Vector-DB upload task
    if ENABLE_NV_INGEST_VDB_UPLOAD:
        ingestor = ingestor.vdb_upload(
            # Milvus configurations
            collection_name=kwargs.get("collection_name"),
            milvus_uri=kwargs.get("vdb_endpoint", config.vector_store.url),

            # Minio configurations
            minio_endpoint=os.getenv("rag-minio:9000"),
            access_key=os.getenv("minioadmin"),
            secret_key=os.getenv("minioadmin"),

            # Hybrid search configurations
            sparse=(config.vector_store.search_type == "hybrid"),

            # Additional configurations
            enable_images=extraction_options.get("extract_images", config.nv_ingest.extract_images),
            recreate=False, # Don't re-create milvus collection
            dense_dim=config.embeddings.dimensions,

            #add metadata
            meta_dataframe=meta_df, 
            meta_source_field="source", 
            meta_fields=metadata_keys,

            gpu_index = config.vector_store.enable_gpu_index,
            gpu_search = config.vector_store.enable_gpu_search,
        )

    return ingestor




data={"collection_name": "test3", "extraction_options": { "extract_text": True, "extract_tables": True, "extract_charts": True, "extract_images": False, "extract_method": "pdfium", "text_depth": "page", }, "split_options": { "chunk_size": 1024, "chunk_overlap": 150 }, "custom_metadata": {"response_id":[123,456],"user_id": [12, 24]}}

if __name__ == "__main__":
    NV_INGEST_CLIENT_INSTANCE = src.utils.get_nv_ingest_client()
    src.utils.get_nv_ingest_ingestor(NV_INGEST_CLIENT_INSTANCE, filepaths=["testfile.txt","testfile2.txt"], kwargs=data )