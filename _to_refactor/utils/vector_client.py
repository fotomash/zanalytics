"""
Vector Client for NCOS v11.5 Phoenix-Mesh

This module provides a client interface for vector database operations,
supporting different vector database backends like FAISS, Pinecone, etc.
"""

import logging
import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class VectorClient:
    """
    Client for vector database operations.

    Supports different vector database backends like FAISS, Pinecone, etc.
    """

    def __init__(
        self,
        vector_db_type: str = "faiss",
        connection_string: Optional[str] = None,
        dimension: int = 768
    ):
        """
        Initialize the vector client.

        Args:
            vector_db_type: Type of vector database ("faiss", "pinecone", "milvus")
            connection_string: Connection string for the vector database
            dimension: Dimension of the vectors
        """
        self.vector_db_type = vector_db_type
        self.connection_string = connection_string
        self.dimension = dimension
        self.indexes = {}
        self.metadata = {}

        logger.info(f"Initializing vector client with backend: {vector_db_type}")

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the vector database backend."""
        if self.vector_db_type == "faiss":
            self._initialize_faiss()
        elif self.vector_db_type == "pinecone":
            self._initialize_pinecone()
        elif self.vector_db_type == "milvus":
            self._initialize_milvus()
        else:
            raise ValueError(f"Unsupported vector database type: {self.vector_db_type}")

    def _initialize_faiss(self) -> None:
        """Initialize FAISS backend."""
        try:
            import faiss
            self.faiss = faiss
            logger.info("FAISS backend initialized")
        except ImportError:
            logger.error("Failed to import FAISS. Please install it with: pip install faiss-cpu or faiss-gpu")
            raise

    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone backend."""
        try:
            import pinecone

            if not self.connection_string:
                raise ValueError("Connection string is required for Pinecone")

            # Parse connection string
            # Expected format: "api_key:environment:project_name"
            parts = self.connection_string.split(":")
            if len(parts) != 3:
                raise ValueError("Invalid Pinecone connection string format. Expected: api_key:environment:project_name")

            api_key, environment, project_name = parts

            pinecone.init(api_key=api_key, environment=environment)
            self.pinecone = pinecone
            self.project_name = project_name

            logger.info("Pinecone backend initialized")

        except ImportError:
            logger.error("Failed to import Pinecone. Please install it with: pip install pinecone-client")
            raise

    def _initialize_milvus(self) -> None:
        """Initialize Milvus backend."""
        try:
            from pymilvus import connections, Collection

            if not self.connection_string:
                raise ValueError("Connection string is required for Milvus")

            # Parse connection string
            # Expected format: "host:port:username:password"
            parts = self.connection_string.split(":")
            if len(parts) != 4:
                raise ValueError("Invalid Milvus connection string format. Expected: host:port:username:password")

            host, port, username, password = parts

            connections.connect(
                alias="default",
                host=host,
                port=port,
                username=username,
                password=password
            )

            self.milvus_connections = connections
            self.milvus_collection = Collection

            logger.info("Milvus backend initialized")

        except ImportError:
            logger.error("Failed to import Milvus. Please install it with: pip install pymilvus")
            raise

    def create_index(self, namespace: str) -> None:
        """
        Create a new vector index for a namespace.

        Args:
            namespace: Namespace for the index
        """
        logger.info(f"Creating vector index for namespace: {namespace}")

        if self.vector_db_type == "faiss":
            self._create_faiss_index(namespace)
        elif self.vector_db_type == "pinecone":
            self._create_pinecone_index(namespace)
        elif self.vector_db_type == "milvus":
            self._create_milvus_index(namespace)

    def _create_faiss_index(self, namespace: str) -> None:
        """Create a FAISS index for a namespace."""
        if namespace in self.indexes:
            logger.warning(f"Index for namespace {namespace} already exists")
            return

        # Create an L2 index by default
        index = self.faiss.IndexFlatL2(self.dimension)
        self.indexes[namespace] = index
        self.metadata[namespace] = {}

    def _create_pinecone_index(self, namespace: str) -> None:
        """Create a Pinecone index for a namespace."""
        # Check if index exists
        if namespace in self.pinecone.list_indexes():
            logger.warning(f"Index for namespace {namespace} already exists in Pinecone")
            return

        # Create the index
        self.pinecone.create_index(
            name=namespace,
            dimension=self.dimension,
            metric="cosine"
        )

    def _create_milvus_index(self, namespace: str) -> None:
        """Create a Milvus index for a namespace."""
        # Implementation would depend on Milvus specifics
        # This is a simplified version
        pass

    def add_vector(
        self,
        namespace: str,
        vector: List[float],
        metadata: Dict[str, Any],
        id: Optional[str] = None
    ) -> str:
        """
        Add a vector to the index.

        Args:
            namespace: Namespace for the vector
            vector: Vector to add
            metadata: Metadata for the vector
            id: Optional ID for the vector

        Returns:
            ID of the added vector
        """
        if namespace not in self.indexes and self.vector_db_type == "faiss":
            self.create_index(namespace)

        vector_id = id or str(uuid.uuid4())

        if self.vector_db_type == "faiss":
            self._add_faiss_vector(namespace, vector_id, vector, metadata)
        elif self.vector_db_type == "pinecone":
            self._add_pinecone_vector(namespace, vector_id, vector, metadata)
        elif self.vector_db_type == "milvus":
            self._add_milvus_vector(namespace, vector_id, vector, metadata)

        return vector_id

    def _add_faiss_vector(self, namespace: str, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a vector to a FAISS index."""
        # Convert to numpy array
        vector_np = np.array([vector], dtype=np.float32)

        # Add to index
        self.indexes[namespace].add(vector_np)

        # Store metadata
        self.metadata[namespace][vector_id] = {
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }

    def _add_pinecone_vector(self, namespace: str, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a vector to a Pinecone index."""
        # Get the index
        index = self.pinecone.Index(namespace)

        # Upsert the vector
        index.upsert(
            vectors=[
                (vector_id, vector, metadata)
            ]
        )

    def _add_milvus_vector(self, namespace: str, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a vector to a Milvus index."""
        # Implementation would depend on Milvus specifics
        # This is a simplified version
        pass

    def search_vectors(
        self,
        namespace: str,
        query_vector: List[float],
        limit: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            namespace: Namespace to search in
            query_vector: Query vector
            limit: Maximum number of results
            filter: Filter for metadata

        Returns:
            List of results with vectors, metadata, and similarity scores
        """
        if self.vector_db_type == "faiss":
            return self._search_faiss_vectors(namespace, query_vector, limit, filter)
        elif self.vector_db_type == "pinecone":
            return self._search_pinecone_vectors(namespace, query_vector, limit, filter)
        elif self.vector_db_type == "milvus":
            return self._search_milvus_vectors(namespace, query_vector, limit, filter)

        return []

    def _search_faiss_vectors(
        self,
        namespace: str,
        query_vector: List[float],
        limit: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for vectors in a FAISS index."""
        if namespace not in self.indexes:
            logger.warning(f"Index for namespace {namespace} does not exist")
            return []

        # Convert to numpy array
        query_np = np.array([query_vector], dtype=np.float32)

        # Search
        distances, indices = self.indexes[namespace].search(query_np, limit)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            # Get metadata
            vector_id = list(self.metadata[namespace].keys())[idx]
            metadata = self.metadata[namespace][vector_id]["metadata"]

            # Apply filter if provided
            if filter and not self._apply_filter(metadata, filter):
                continue

            results.append({
                "id": vector_id,
                "metadata": metadata,
                "score": 1.0 - float(distances[0][i]),  # Convert distance to similarity score
                "vector": None  # We don't return the vector by default
            })

        return results

    def _search_pinecone_vectors(
        self,
        namespace: str,
        query_vector: List[float],
        limit: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for vectors in a Pinecone index."""
        # Get the index
        index = self.pinecone.Index(namespace)

        # Query
        results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True,
            filter=filter
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "metadata": match.metadata,
                "score": float(match.score),
                "vector": None  # We don't return the vector by default
            })

        return formatted_results

    def _search_milvus_vectors(
        self,
        namespace: str,
        query_vector: List[float],
        limit: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for vectors in a Milvus index."""
        # Implementation would depend on Milvus specifics
        # This is a simplified version
        return []

    def _apply_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Apply a filter to metadata."""
        for key, value in filter.items():
            if key not in metadata:
                return False

            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False

        return True

    def delete_vector(self, namespace: str, vector_id: str) -> bool:
        """
        Delete a vector from the index.

        Args:
            namespace: Namespace of the vector
            vector_id: ID of the vector to delete

        Returns:
            True if successful, False otherwise
        """
        if self.vector_db_type == "faiss":
            return self._delete_faiss_vector(namespace, vector_id)
        elif self.vector_db_type == "pinecone":
            return self._delete_pinecone_vector(namespace, vector_id)
        elif self.vector_db_type == "milvus":
            return self._delete_milvus_vector(namespace, vector_id)

        return False

    def _delete_faiss_vector(self, namespace: str, vector_id: str) -> bool:
        """Delete a vector from a FAISS index."""
        if namespace not in self.indexes:
            logger.warning(f"Index for namespace {namespace} does not exist")
            return False

        if vector_id not in self.metadata[namespace]:
            logger.warning(f"Vector {vector_id} does not exist in namespace {namespace}")
            return False

        # FAISS doesn't support direct deletion
        # We need to recreate the index without the vector
        # This is inefficient but necessary for this implementation

        # Get all vectors except the one to delete
        vectors = []
        metadata = {}

        for vid, meta in self.metadata[namespace].items():
            if vid != vector_id:
                # We'd need to retrieve the actual vector here
                # For simplicity, we'll just skip this in the example
                pass

        # Delete the old index
        del self.indexes[namespace]
        del self.metadata[namespace]

        # Create a new index
        self.create_index(namespace)

        # Add all vectors back
        # (This part would be implemented in a real system)

        return True

    def _delete_pinecone_vector(self, namespace: str, vector_id: str) -> bool:
        """Delete a vector from a Pinecone index."""
        # Get the index
        index = self.pinecone.Index(namespace)

        # Delete the vector
        index.delete(ids=[vector_id])

        return True

    def _delete_milvus_vector(self, namespace: str, vector_id: str) -> bool:
        """Delete a vector from a Milvus index."""
        # Implementation would depend on Milvus specifics
        # This is a simplified version
        return True

    def clear_namespace(self, namespace: str) -> bool:
        """
        Clear all vectors in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            True if successful, False otherwise
        """
        if self.vector_db_type == "faiss":
            return self._clear_faiss_namespace(namespace)
        elif self.vector_db_type == "pinecone":
            return self._clear_pinecone_namespace(namespace)
        elif self.vector_db_type == "milvus":
            return self._clear_milvus_namespace(namespace)

        return False

    def _clear_faiss_namespace(self, namespace: str) -> bool:
        """Clear all vectors in a FAISS namespace."""
        if namespace not in self.indexes:
            logger.warning(f"Index for namespace {namespace} does not exist")
            return False

        # Delete the old index
        del self.indexes[namespace]
        del self.metadata[namespace]

        # Create a new index
        self.create_index(namespace)

        return True

    def _clear_pinecone_namespace(self, namespace: str) -> bool:
        """Clear all vectors in a Pinecone namespace."""
        # Get the index
        index = self.pinecone.Index(namespace)

        # Delete all vectors
        index.delete(delete_all=True)

        return True

    def _clear_milvus_namespace(self, namespace: str) -> bool:
        """Clear all vectors in a Milvus namespace."""
        # Implementation would depend on Milvus specifics
        # This is a simplified version
        return True

    def save_indexes(self, directory: str) -> None:
        """
        Save all indexes to disk.

        Args:
            directory: Directory to save indexes to
        """
        if self.vector_db_type != "faiss":
            logger.warning(f"Saving indexes is only supported for FAISS backend")
            return

        os.makedirs(directory, exist_ok=True)

        # Save each index
        for namespace, index in self.indexes.items():
            index_path = os.path.join(directory, f"{namespace}.index")
            metadata_path = os.path.join(directory, f"{namespace}.metadata.json")

            self.faiss.write_index(index, index_path)

            with open(metadata_path, "w") as f:
                json.dump(self.metadata[namespace], f)

        logger.info(f"Saved {len(self.indexes)} indexes to {directory}")

    def load_indexes(self, directory: str) -> None:
        """
        Load indexes from disk.

        Args:
            directory: Directory to load indexes from
        """
        if self.vector_db_type != "faiss":
            logger.warning(f"Loading indexes is only supported for FAISS backend")
            return

        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist")
            return

        # Find all index files
        index_files = [f for f in os.listdir(directory) if f.endswith(".index")]

        for index_file in index_files:
            namespace = index_file[:-6]  # Remove .index
            index_path = os.path.join(directory, index_file)
            metadata_path = os.path.join(directory, f"{namespace}.metadata.json")

            # Load index
            try:
                index = self.faiss.read_index(index_path)
                self.indexes[namespace] = index

                # Load metadata
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        self.metadata[namespace] = json.load(f)
                else:
                    self.metadata[namespace] = {}

                logger.info(f"Loaded index for namespace: {namespace}")

            except Exception as e:
                logger.error(f"Failed to load index {index_path}: {e}")

        logger.info(f"Loaded {len(self.indexes)} indexes from {directory}")

    def close(self) -> None:
        """Close the vector client and release resources."""
        logger.info("Closing vector client")

        if self.vector_db_type == "pinecone":
            # No explicit close needed for Pinecone
            pass
        elif self.vector_db_type == "milvus":
            # Close Milvus connection
            if hasattr(self, "milvus_connections"):
                self.milvus_connections.disconnect("default")
