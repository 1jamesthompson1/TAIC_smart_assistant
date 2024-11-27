import gradio as gr
import openai
import voyageai
import lancedb
import dotenv
import os
import json
import argparse

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
        elif type == "vector":
            final_query = self.embed_query(query)
        else:
            raise ValueError(f"type must be 'fts' or 'vector' not {type}")

        results = (self.all_document_types_table
            .search(final_query, query_type=type)
            .where(where_statement, prefilter=True)
            .to_pandas())

        return results

    def process_input(self, input_text, history=[]):
        history.append({
            "role": "user",
            "content": input_text
        })

        messages = [
            {"role": "system", "content": """
 You are a helpful chatbot assistant working at the New Zealand transport accident investigation commision.
 You will be provided the conversation history and a query fro the user. You can either respond directly or can call a function that searches a database of all of TAICs accident investigation reports.
 """},
        ] + history

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for safety issues and recommendations from the New Zealand Transport Accident Investigation Commission.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The query to search for. Left as None and it will return everything."
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["fts", "vector"],
                                    "description": "The type of search to perform."
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
                            "additionalProperties": False
                        }
                    }
                }
            ]
        )   

        response_message = response.choices[0].message

        if response_message.content:
            history.append({
                "role": "assistant",
                "content": response_message.content
            })
            return "", history, None

        tool_call = response_message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)

        results = self.knowledge_search(arguments["query"], arguments["type"], arguments["year_range"], arguments["document_type"], arguments["modes"])

        messages = history + [
            response_message,
            {
                "role": "tool",
                "content": results.to_json(orient="records"),
                "tool_call_id": tool_call.id
            },
        ]

        rag_response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        history.append({
            "role": "assistant",
            "content": rag_response.choices[0].message.content
        })

        return "", history, results


def main():
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
        fill_height=True
    ) as demo:
        gr.Markdown("# TAIC smart assistant")
        chatbot_interface = gr.Chatbot(
            type="messages",
            height="90%",
            min_height=300,
            layout="bubble",
        )
        input_text = gr.Textbox(placeholder="Type your message here...", show_label=False)
        results_table = gr.DataFrame(interactive=False)

        input_text.submit(fn=chatbot_instance.process_input, inputs=[input_text, chatbot_interface], outputs=[input_text, chatbot_interface, results_table])

    demo.launch(debug=args.debug, share=args.share)

if __name__ == "__main__":
    main()
