import json
from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime

class Tool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for the tool parameters."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters and return result as string."""
        pass
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

class SearchTool(Tool):
    """Tool for searching the knowledge base."""
    
    def __init__(self, searcher):
        self.searcher = searcher
    
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return """Search for safety issues and recommendations from the New Zealand Transport Accident Investigation Commission. This function searches a vector database."""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for. If left as an empty string it will return all results that match the other parameters.",
                },
                "type": {
                    "type": "string",
                    "enum": ["fts", "vector"],
                    "description": "The type of search to perform. fts should be used if the query is asking a specific question about an organisation, etc. Otherwise for more general information use vector, it will embed your query and search the vector database.",
                },
                "year_range": {
                    "type": "array",
                    "description": f"An array specifying the start and end years for filtering results. Valid range is 2000-{datetime.now().year}.",
                    "items": {"type": "number"},
                },
                "document_type": {
                    "type": "array",
                    "description": "A list of document types to filter the search results. Safety issues and recommendations follow definitions given, while report sections are reports chunked into sections/pages, summary are brief overviews of the reports scrapped from the agencies report webpages, they are only present for TAIC and ATSB. Valid types are 'safety_issue', 'recommendation', 'report_section' and 'summary'.",
                    "items": {"type": "string"},
                },
                "modes": {
                    "type": "array",
                    "description": "A list of modes to filter the search results. Valid modes are 0, 1, and 2. Which are aviation, rail, and marine respectively.",
                    "items": {"type": "string"},
                },
                "agencies": {
                    "type": "array",
                    "description": "A list of agencies to filter the search results. Valid agencies are TSB, ATSB, and TAIC. These are Transport Safety Board (Canada), Australian Transport Safety Board, and Transport Accident Investigation Commission (New Zealand) respectively.",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "query",
                "type",
                "year_range",
                "document_type",
                "modes",
                "agencies",
            ],
        }
    
    def execute(self, **kwargs) -> str:
        results, info, plots = self.searcher.knowledge_search(**kwargs)
        return results.to_json(orient="records")

class ReadReportTool(Tool):
    """Tool for reading individual reports."""
    
    def __init__(self, searcher):
        self.searcher = searcher
    
    @property
    def name(self) -> str:
        return "read_report"
    
    @property
    def description(self) -> str:
        return "Read the full content of a specific report by its ID."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "report_id": {
                    "type": "string",
                    "description": "The ID of the report to read.",
                },
            },
            "required": ["report_id"],
        }
    
    def execute(self, report_id: str, **kwargs) -> str:
        # Assuming searcher has a method to get full report
        # You may need to implement this in Searching.py
        return "Not yet implemented yet"

class ReasoningTool(Tool):
    """Tool for sending information to a powerful reasoning model."""
    
    def __init__(self, openai_client, reasoning_model: str = "gpt-4o"):
        self.openai_client = openai_client
        self.reasoning_model = reasoning_model
    
    @property
    def name(self) -> str:
        return "reason"
    
    @property
    def description(self) -> str:
        return "Send information to a powerful reasoning model for deeper analysis."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "information": {
                    "type": "string",
                    "description": "The information to analyze.",
                },
                "task": {
                    "type": "string",
                    "description": "The reasoning task to perform.",
                },
            },
            "required": ["information", "task"],
        }
    
    def execute(self, information: str, task: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": "You are a powerful reasoning AI. Analyze the provided information and perform the specified task."},
            {"role": "user", "content": f"Task: {task}\n\nInformation: {information}"}
        ]
        response = self.openai_client.chat.completions.create(
            model=self.reasoning_model,
            messages=messages,
        )
        return response.choices[0].message.content

class InternalThoughtTool(Tool):
    """Tool for the assistant to talk to itself."""
    
    def __init__(self, openai_client, model: str = "gpt-4"):
        self.openai_client = openai_client
        self.model = model
    
    @property
    def name(self) -> str:
        return "internal_thought"
    
    @property
    def description(self) -> str:
        return "Generate internal thoughts for planning or reflection."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "The internal thought or question.",
                },
            },
            "required": ["thought"],
        }
    
    def execute(self, thought: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": "You are thinking internally. Reflect on the thought provided."},
            {"role": "user", "content": thought}
        ]
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content
