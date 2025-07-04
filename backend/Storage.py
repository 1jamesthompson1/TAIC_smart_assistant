"""
Azure Blob Storage implementation for storing conversation histories.
This stores the raw JSON conversation as blobs and metadata in Table Storage.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from azure.storage.blob import BlobServiceClient
import os


class ConversationBlobStore:
    """
    Handles conversation storage using Azure Blob Storage for JSON data
    and Table Storage for metadata.
    """
    
    def __init__(self, connection_string: str, container_name: str):
        """
        Initialize the Blob Storage client.
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the container to store conversation blobs
        """
        self.connection_string = connection_string or self._get_connection_string()
        self.container_name = container_name
        
        # Initialize Blob client
        self.blob_client = BlobServiceClient.from_connection_string(self.connection_string)
        
        # Create container if it doesn't exist
        self._setup_container()
    
    def _get_connection_string(self) -> str:
        """
        Build connection string from environment variables.
        Uses the same storage account as your existing Table Storage.
        """
        account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
        account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
        
        if not account_name or not account_key:
            raise ValueError("AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY must be set")
        
        return f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    
    def _setup_container(self):
        """Create blob container if it doesn't exist."""
        try:
            self.container_client = self.blob_client.get_container_client(self.container_name)
            
            # Create container if it doesn't exist
            if not self.container_client.exists():
                self.container_client.create_container()
                print(f"Created blob container: {self.container_name}")
            
        except Exception as e:
            raise Exception(f"Failed to setup blob container: {e}")
    
    def _get_blob_name(self, username: str, conversation_id: str) -> str:
        """Generate blob name for a conversation."""
        return f"{username}/{conversation_id}.json"
    
    def store_conversation_blob(self, username: str, conversation_id: str, history: List[Dict]) -> str:
        """
        Store conversation history as a JSON blob.
        
        Args:
            username: Username of the conversation owner
            conversation_id: Unique identifier for the conversation
            history: List of conversation messages
            
        Returns:
            Blob name/path for the stored conversation
        """
        if not history:
            return None
            
        blob_name = self._get_blob_name(username, conversation_id)
        
        try:
            # Convert conversation to JSON
            conversation_json = json.dumps(history, ensure_ascii=False, indent=None)
            
            # Upload to blob storage
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                conversation_json, 
                content_type="application/json",
                overwrite=True,
                encoding='utf-8'
            )
            
            print(f"✓ Stored conversation blob: {blob_name} ({len(conversation_json)} bytes)")
            return blob_name
            
        except Exception as e:
            print(f"✗ Failed to store conversation blob {blob_name}: {e}")
            # Save failed conversation to local file for debugging
            self._save_failed_conversation(conversation_id, history, str(e))
            raise Exception(f"Failed to store conversation blob: {e}")
    
    def retrieve_conversation_blob(self, username: str, conversation_id: str) -> Optional[List[Dict]]:
        """
        Retrieve conversation history from blob storage.
        
        Args:
            username: Username of the conversation owner
            conversation_id: ID of the conversation to retrieve
            
        Returns:
            List of conversation messages or None if not found
        """
        blob_name = self._get_blob_name(username, conversation_id)
        
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Download blob content
            blob_data = blob_client.download_blob()
            conversation_json = blob_data.readall().decode('utf-8')
            
            # Parse JSON
            history = json.loads(conversation_json)
            print(f"✓ Retrieved conversation blob: {blob_name}")
            return history
            
        except Exception as e:
            print(f"✗ Failed to retrieve conversation blob {blob_name}: {e}")
            return None
    
    def delete_conversation_blob(self, username: str, conversation_id: str) -> bool:
        """
        Delete a conversation blob.
        
        Args:
            username: Username of the conversation owner
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if successful, False otherwise
        """
        blob_name = self._get_blob_name(username, conversation_id)
        
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            print(f"✓ Deleted conversation blob: {blob_name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to delete conversation blob {blob_name}: {e}")
            return False
    
    def _save_failed_conversation(self, conversation_id: str, history: List[Dict], error: str):
        """Save failed conversation to file for debugging."""
        try:
            filename = f"Failed_blob_conversation_{conversation_id}.json"
            with open(filename, "w", encoding='utf-8') as f:
                json.dump({
                    "error": error,
                    "history": history,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            print(f"Saved failed conversation to {filename}")
        except Exception as e:
            print(f"Could not save failed conversation: {e}")


class ConversationMetadataStore:
    """
    Handles conversation metadata storage in Table Storage.
    This works alongside the blob storage for complete conversation management.
    """
    
    def __init__(self, table_client, blob_store: ConversationBlobStore):
        """
        Initialize with existing table client and blob store.
        
        Args:
            table_client: Azure Table Storage client (reuse existing)
            blob_store: ConversationBlobStore instance
        """
        self.table_client = table_client
        self.blob_store = blob_store
    
    def create_or_update_conversation(self, username: str, conversation_id: str, 
                                    history: List[Dict], conversation_title: str = None) -> bool:
        """
        Store conversation: blob for JSON data, Table Storage for metadata.
        
        Args:
            username: Username of the conversation owner
            conversation_id: Unique identifier for the conversation
            history: List of conversation messages
            conversation_title: Title for the conversation
            
        Returns:
            True if successful, False otherwise
        """
        if not history:
            return False
            
        try:
            # 1. Store conversation JSON in blob storage
            blob_name = self.blob_store.store_conversation_blob(username, conversation_id, history)
            if not blob_name:
                return False
            
            # 2. Store metadata in Table Storage
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if conversation already exists to preserve created_at
            existing_entity = None
            try:
                existing_entity = self.table_client.get_entity(
                    partition_key=username, 
                    row_key=conversation_id
                )
            except:
                pass  # Entity doesn't exist, which is fine
            
            entity = {
                "PartitionKey": username,
                "RowKey": conversation_id,
                "conversation_title": conversation_title or "Untitled Conversation",
                "blob_name": blob_name,
                "message_count": len(history),
                "last_updated": now,
                "created_at": existing_entity.get("created_at", now) if existing_entity else now,
            }
            
            # Upsert the entity (create or update)
            self.table_client.upsert_entity(entity=entity)
            
            print(f"✓ Stored conversation metadata for {conversation_id}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to store conversation {conversation_id}: {e}")
            return False
    
    def get_user_conversations(self, username: str) -> List[Dict]:
        """
        Retrieve all conversations for a user from Blob Storage.
        
        Args:
            username: Username to retrieve conversations for
            
        Returns:
            List of conversations with full message history
        """
        try:
            # Get metadata from Table Storage
            entities = self.table_client.query_entities(
                query_filter=f"PartitionKey eq '{username}'"
            )
            
            conversations = []
            
            for entity in entities:
                conversation_id = entity.get("RowKey")
                
                history = self.blob_store.retrieve_conversation_blob(username, conversation_id)
                if history is None:
                    print(f"⚠ Could not retrieve blob for conversation {conversation_id}")
                    continue
                
                conversations.append({
                    "conversation_title": entity.get("conversation_title"),
                    "messages": history,
                    "id": conversation_id,
                    "last_updated": entity.get("last_updated"),
                    "created_at": entity.get("created_at"),
                    "message_count": entity.get("message_count", len(history)),
                })
            
            # Sort by last updated
            conversations.sort(
                key=lambda x: datetime.strptime(x["last_updated"], "%Y-%m-%d %H:%M:%S"),
                reverse=True
            )
            
            print(f"✓ Retrieved {len(conversations)} conversations for {username}")
            return conversations
            
        except Exception as e:
            print(f"✗ Failed to retrieve conversations for {username}: {e}")
            return []
    
    def delete_conversation(self, username: str, conversation_id: str) -> bool:
        """
        Delete both metadata and blob for a conversation.
        
        Args:
            username: Username of the conversation owner
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete blob first
            self.blob_store.delete_conversation_blob(username, conversation_id)
            
            # Delete metadata from table
            self.table_client.delete_entity(
                partition_key=username,
                row_key=conversation_id
            )
            
            print(f"✓ Deleted conversation {conversation_id} for {username}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to delete conversation {conversation_id}: {e}")
            return False
