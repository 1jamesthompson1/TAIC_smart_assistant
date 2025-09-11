import json
import openai
from datetime import datetime
from rich import print
from .AssistantTools import SearchTool, ReadReportTool, ReasoningTool, InternalThoughtTool


class Assistant:
    def __init__(self, searcher, openai_api_key, openai_endpoint=None, reasoning_model="gpt-4.1"):
        print("[bold]Creating Chatbot[/bold]")
        self.searcher = searcher
        if openai_endpoint:
            self.openai_client = openai.AzureOpenAI(
                api_version="2025-04-01-preview",
                api_key=openai_api_key,
                azure_endpoint=openai_endpoint
            )
        else:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize tools
        self.tools = [
            SearchTool(searcher),
            # ReadReportTool(searcher),
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
        system_message = f"""
You are a expert working at the New Zealand transport accident investigation commission. Your job is to assistant employees of TAIC with their queries. The day is {datetime.now()}. You should respond as if you are a senior accident investigator/research who is speaking to your colleagues. Keep your responses short and to the point.

**Follow a methodical approach for each query:**
1. **Observe**: Analyze the user's query, identify key elements, and gather relevant information using available tools (search, read_report, reason).
2. **Plan**: Use the internal_thought tool to reflect on what you've observed and plan your response strategy. Consider what additional information you need and how to structure your final answer.
3. **Act**: Execute your plan by calling appropriate tools or providing the final response. If you need more information, gather it first before responding.

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
"""

        # Strip out the metadata for the messages
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]

        while True:
            response = self.openai_client.responses.create(
                model="gpt-4.1",
                instructions=system_message,
                input=messages,
                tools=[tool.to_openai_format() for tool in self.tools],
                parallel_tool_calls=True,
                store=False,
                stream=True,
            )

            print("[bold]Assistant response received, processing...[/bold]")

            function_calls = {}
            for chunk in response:
                if chunk.type == 'response.output_item.added':
                    if chunk.item.type == "function_call":
                        print(f"Function call: {chunk.item.name} {chunk.item.arguments}")
                        function_calls[chunk.output_index] = chunk.item
                    elif chunk.item.type == "message":
                        history.append({"role": "assistant", "content": ""})
                        yield history
                
                elif chunk.type == 'response.function_call_arguments.delta':
                    index = chunk.output_index

                    if function_calls[index]:
                        function_calls[index].arguments += chunk.delta
                    
                elif chunk.type == 'response.output_text.delta':
                    history[-1]["content"] += chunk.delta
                    yield history
                
                elif chunk.type == 'response.output_item.done':
                    if chunk.item.type == "function_call":
                        messages += [chunk.item.to_dict()]
                        message = function_calls[chunk.output_index]
                        tool_name = message.name
                        tool_args = json.loads(message.arguments)

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
                            "content": result,
                            "metadata": {"title": f"📖 Result from {tool_name}", "status": "done"}
                        })
                        yield history
                        
                        # Append tool result
                        messages.append({
                            "type": "function_call_output",
                            "output": result,
                            "call_id": chunk.item.call_id
                        })
                    elif chunk.item.type == "message":
                        messages += [{"role": "assistant", "content": history[-1]["content"]}]
                        yield history

            
            # If I made some function calls, continue the loop to reflect on the new information
            if function_calls:
                messages = messages + [
                    {
                        "role": "system",
                        "content": "You have new information from the tools you executed. Reflect on this and decide your next steps.",
                    }
                ]
                continue
            else:
                break
            
        return history
