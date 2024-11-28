import gradio as gr
import openai
import voyageai
import lancedb
import dotenv
import os
import json
import argparse
from datetime import datetime
dotenv.load_dotenv()

class Chatbot:
    def __init__(self, openai_api_key, voyageai_api_key, db_uri):
        self.vector_db = lancedb.connect(db_uri)
        self.all_document_types_table = self.vector_db.open_table("all_document_types")
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.voyageai_client = voyageai.Client(api_key=voyageai_api_key)

    def embed_query(self, query: str):
        return self.voyageai_client.embed(
            query, model="voyage-large-2-instruct", input_type="query", truncation=False
        ).embeddings[0]

    def knowledge_search(self, query: str, type: str, year_range: tuple[int, int], document_type: list[str], modes: list[int]):
        limit = 100
        where_statement = []
        if year_range:
            where_statement.append(f"year >= {year_range[0]} and year <= {year_range[1]}")
        if document_type:
            document_types = ', '.join(f'"{dt}"' for dt in document_type)
            where_statement.append(f"document_type in ({document_types})")
        if modes and len(modes) > 1:
            where_statement.append(f"mode in {tuple(modes)}")
        elif modes and len(modes) == 1:
            where_statement.append(f"mode = {modes[0]}")


        where_statement = ' AND '.join(where_statement)

        print(where_statement)
        if query == "" or query is None:
            final_query = None
        elif type == "fts":
            final_query = query
            limit = 1000
        elif type == "vector":
            final_query = self.embed_query(query)
        else:
            raise ValueError(f"type must be 'fts' or 'vector' not {type}")

        print(f"Conducting search with query: {final_query if isinstance(final_query, str) else 'vector embeddings of ' + query}, and filters '{where_statement}'")

        results = (self.all_document_types_table
            .search(final_query, query_type=type)
            .where(where_statement, prefilter=True)
            .limit(limit)
            .to_pandas()).drop(columns=["vector"])

        return results

    def process_input(self,history=[]):
        system_message = {
            "role": "system",
            "content": f"""
You are a expert working at the New Zealand transport accident investigation commision. Your job is to assistant users with their queries. The day is {datetime.now()}.
You will be provided the conversation history and a query from the user. You can either respond directly or can call a function that searches a database of all of TAICs accident investigation reports.
When talking about reports it is important to use the document ID and report IDs to provide references.
"""}


        response_stream = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message] + history,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description":
"""Search for safety issues and recommendations from the New Zealand Transport Accident Investigation Commission. This function searches a vector database.
Eample function calls:
```json
{
    "query": "What are common elements in floatation devices and fires?",
    "type": "vector",
    "year_range": [2020, 2023],
    "document_type": ["safety_issue", "recommendation"],
    "modes": [0, 1, 2]
}
```

```json
{
    "query": "",
    "type": "vector",
    "year_range": [2020, 2023],
    "document_type": ["safety_issue"],
    "modes": [2]
}
```

""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The query to search for. If left as an empty string it will return all results that match the other paramters."
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["fts", "vector"],
                                    "description": "The type of search to perform. fts should be used if the query is asking a specific question about a organisation, organisation etc. Otherwsie for more general information use vector, it will embed your query and search the vector database."
                                },
                                "year_range": {
                                    "type": "array",
                                    "description": "An array specifying the start and end years for filtering results. Valid range is 2000-2023.",
                                    "items": {"type": "number"}
                                },
                                "document_type": {
                                    "type": "array",
                                    "description": "A list of document types to filter the search results. Valid types are 'safety_issue', 'recommendation', 'report_section'.",
                                    "items": {"type": "string"}
                                },
                                "modes": {
                                    "type": "array",
                                    "description": "A list of modes to filter the search results. Valid modes are 0, 1, and 2. Which are aviation, rail, and marine respectively.",
                                    "items": {"type": "number"}
                                }
                            },
                            "required": ["query", "type", "year_range", "document_type", "modes"],
                            "additionalProperties": False,
                            "strict": True
                        }

                    }
                }
            ],
            stream=True
        )   
        function_arguments_str = ""
        function_name = ""
        tool_call_id = ""
        is_collecting_function_args = False

        history.append({
            "role": "assistant",
            "content": ""
        })

        for part in response_stream:
            delta = part.choices[0].delta
            finish_reason = part.choices[0].finish_reason

            # Process assistant content
            if delta.content:
                history[-1]["content"] += delta.content
                yield history

            if delta.tool_calls:
                is_collecting_function_args = True
                tool_call = delta.tool_calls[0]

                if tool_call.id:
                    tool_call_id = tool_call.id
                if tool_call.function.name:
                    function_name = tool_call.function.name
                
                # Process function arguments delta
                if tool_call.function.arguments:
                    function_arguments_str += tool_call.function.arguments

            # Process tool call with complete arguments
            if finish_reason == "tool_calls" and is_collecting_function_args:
                break


        if not is_collecting_function_args:
            return history

        function_arguments = json.loads(function_arguments_str)

        history[-1]["metadata"] = {"title": f"🔍 Searching database for more information" }
        history[-1]["content"] += f"Using these parameters to search the database: {function_arguments_str}"
        yield history

        results = self.knowledge_search(function_arguments["query"], function_arguments["type"], function_arguments["year_range"], function_arguments["document_type"], function_arguments["modes"])

        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_arguments_str
                    }
                }
            ]
        }

        messages = [system_message] + history + [tool_call_message] + [
            {
                "role": "tool",
                "content": results.to_json(orient="records"),
                "tool_call_id": tool_call_id
            },
        ]

        html_table = f"<style>table {{ width: 100%; }}</style>{results.to_html(index=False)}"

        history.append({
            "role": "assistant",
            "content": html_table,
            "metadata": {"title": f"📖 Reading {results.shape[0]} results"}
        })
        yield history

        rag_response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True
        )

        history.append({
            "role": "assistant",
            "content": ""
        })

        for part in rag_response:
            delta = part.choices[0].delta
            finish_reason = part.choices[0].finish_reason

            # Process assistant content
            if delta.content:
                history[-1]["content"] += delta.content
                yield history


        return history


def handle_submit(user_input, history=None):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})
    return "", history


parser = argparse.ArgumentParser(description="Launch the RAG Chatbot app.")
parser.add_argument('--debug', action='store_true', help='Run the app in debug mode')
parser.add_argument('--share', action='store_true', help='Share the app on a public URL')
args = parser.parse_args()

chatbot_instance = Chatbot(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    voyageai_api_key=os.getenv("VOYAGEAI_API_KEY"),
    db_uri=os.getenv("db_URI")
)

with gr.Blocks(
    title="TAIC smart assistant",
    theme=gr.themes.Base(),
    fill_height=True,
    fill_width=True
) as demo:
    gr.Markdown("# TAIC smart assistant")
    chatbot_interface = gr.Chatbot(
        type="messages",
        height="90%",
        min_height=400,
    )

    input_text = gr.Textbox(placeholder="Type your message here...", show_label=False)

    input_text.submit(fn=handle_submit, inputs=[input_text, chatbot_interface], outputs=[input_text, chatbot_interface], queue=False).then(
        chatbot_instance.process_input, inputs=[chatbot_interface], outputs=[chatbot_interface]
    )

    demo.launch(share=args.share)
