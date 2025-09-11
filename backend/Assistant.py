import json
import openai
from datetime import datetime
from rich import print
from .AssistantTools import SearchTool, ReadReportTool, ReasoningTool, InternalThoughtTool


class Assistant:
    def __init__(self, searcher, openai_api_key, openai_endpoint=None, reasoning_model="gpt-4o"):
        print("[bold]Creating Chatbot[/bold]")
        self.searcher = searcher
        if openai_endpoint:
            self.openai_client = openai.AzureOpenAI(
                api_version="2024-12-01-preview",
                api_key=openai_api_key,
                azure_endpoint=openai_endpoint
            )
        else:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize tools
        self.tools = [
            SearchTool(searcher),
            ReadReportTool(searcher),
            ReasoningTool(self.openai_client, reasoning_model),
            InternalThoughtTool(self.openai_client),
        ]
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        print("[bold]Chatbot created[/bold]")

    def provide_conversation_title(self, history=[]):
        if history == []:
            raise ValueError("history is empty")

        return (
            self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": """
                    You are part of a chatbot assistant at the Transport Accident Investigation Commission that help users add titles to their conversation. You will receive the conversation and you are too response with a 5 word summary of the conversation.
                    Provide a title that will best help the user identify what conversation it was.
                    Just respond with the title and nothing else.
                    """,
                    },
                    history[-1],
                ],
            )
            .choices[0]
            .message.content
        )

    def process_input(self, history=[]):
        system_message = {
            "role": "system",
            "content": f"""
You are a expert working at the New Zealand transport accident investigation commission. Your job is to assistant employees of TAIC with their queries. The day is {datetime.now()}. You should respond as if you are a senior accident investigator/research who is speaking to your colleagues. Keep your responses short and to the point.
You will be provided the conversation history and a query from the user. You can either respond directly or can call functions to gather more information. You may make multiple function calls in sequence if needed to achieve the goal.
For each report you should provide use the report IDs, if you reference any other document you should provide the document type and document ID.

Here is some more dataset information
There are {len(self.searcher.all_document_types_table.schema.names)} columns with {self.searcher.all_document_types_table.count_rows()} rows.
The columns available are: {"".join(self.searcher.all_document_types_table.schema.names)}
The data was last updated on {self.searcher.last_updated}.

Here are some definitions from TAIC:

Safety factor - Any (non-trivial) events or conditions, which increases safety risk. If they occurred in the future, these would
increase the likelihood of an occurrence, and/or the
severity of any adverse consequences associated with the
occurrence.

Safety issue - A safety factor that:
• can reasonably be regarded as having the
potential to adversely affect the safety of future
operations, and
• is characteristic of an organisation, a system, or an
operational environment at a specific point in time.
Safety Issues are derived from safety factors classified
either as Risk Controls or Organisational Influences.

Safety theme - Indication of recurring circumstances or causes, either across transport modes or over time. A safety theme may
cover a single safety issue, or two or more related safety
issues.
""",
        }

        messages = [system_message] + history

        while True:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                tools=[tool.to_openai_format() for tool in self.tools],
                parallel_tool_calls=True,
            )

            choice = response.choices[0]
            message = choice.message

            # If the model wants to call tools
            if message.tool_calls:

                # Append the assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })

                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # Add tool execution to history
                    history.append({
                        "role": "assistant",
                        "content": f"Executing {tool_name} with parameters: {tool_args}",
                        "metadata": {"title": f"🔧 Executing {tool_name}", "status": "pending"}
                    })
                    yield history
                    
                    tool = self.tool_map.get(tool_name)
                    if not tool:
                        result = f"Error: Unknown tool {tool_name}"
                    else:
                        result = tool.execute(**tool_args)
                    
                    history[-1]["metadata"]["status"] = "done"
                    
                    # Add tool result to history
                    history.append({
                        "role": "assistant",
                        "content": f"Tool result from {tool_name}: {result[:500]}{'...' if len(result) > 500 else ''}",
                        "metadata": {"title": f"📖 Result from {tool_name}", "status": "done"}
                    })
                    yield history
                    
                    # Append tool result
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })
                
                # Continue the loop for more processing
                continue
            else:
                # Final response, no tool calls - stream it
                history.append({"role": "assistant", "content": ""})
                for chunk in self.openai_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=messages,
                    stream=True,
                ):
                    if len(chunk.choices) == 0:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        history[-1]["content"] += delta.content
                        yield history
                return history
        function_arguments_str = ""
        function_name = ""
        tool_call_id = ""
        is_collecting_function_args = False
        tool_calls = []

        history.append({"role": "assistant", "content": ""})

        for part in response_stream:
            delta = part.choices[0].delta
            finish_reason = part.choices[0].finish_reason

            # Process assistant content
            if delta.content:
                history[-1]["content"] += delta.content
                yield history

            if delta.tool_calls:
                is_collecting_function_args = True
                for tool_call_delta in delta.tool_calls:
                    if tool_call_delta.id:
                        tool_call_id = tool_call_delta.id
                    if tool_call_delta.function and tool_call_delta.function.name:
                        function_name = tool_call_delta.function.name
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        function_arguments_str += tool_call_delta.function.arguments

            # Process tool call with complete arguments
            if finish_reason == "tool_calls" and is_collecting_function_args:
                break

        if not is_collecting_function_args:
            return history

        try:
            function_arguments = json.loads(function_arguments_str)
        except json.JSONDecodeError as e:
            print(f"[bold red]Error decoding JSON {e}:[/bold red] {function_arguments_str}")
            raise ValueError("Assistant tried to make a function call but failed due to malformed function call")

        history[-1]["metadata"] = {
            "title": f"� Executing {function_name}",
            "status": "pending",
        }
        history[-1]["content"] += (
            f"Executing {function_name} with parameters: {function_arguments_str}"
        )
        yield history

        # Execute the tool
        tool = self.tool_map.get(function_name)
        if not tool:
            raise ValueError(f"Unknown tool: {function_name}")
        
        result = tool.execute(**function_arguments)
        
        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_arguments_str,
                    },
                }
            ],
        }
        history[-1]["metadata"]["status"] = "done"

        messages = (
            [system_message]
            + history
            + [tool_call_message]
            + [
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call_id,
                },
            ]
        )

        history.append(
            {
                "role": "assistant",
                "content": f"Tool result: {result[:500]}...",  # Truncate for display
                "metadata": {"title": f"📖 Tool result from {function_name}", 'status': 'done'},
            }
        )
        yield history

        # Continue with RAG response
        rag_response = self.openai_client.chat.completions.create(
            model="gpt-4.1", messages=messages, stream=True
        )

        history.append({"role": "assistant", "content": ""})

        for part in rag_response:
            delta = part.choices[0].delta
            finish_reason = part.choices[0].finish_reason

            # Process assistant content
            if delta.content:
                history[-1]["content"] += delta.content
                yield history

        return history
