import json
import openai
from datetime import datetime
from rich import print
from .AssistantTools import SearchTool, ReadReportTool, ReasoningTool, InternalThoughtTool

class CompleteHistory(list):
    '''
    This is a modified list that holds the complete history of the conversation.
    
    It stores all the information needed for both OpenAI and Gradio formats.
    '''
    def add_message(self, role, content):

        message = {
            "role": role,
            "content": content,
        }

        self.append({
            "display": message.copy(),
            "ai": message.copy(),
        })

    def update_last_message(self, delta_content):
        '''
        Used for streaming updates to the last message.
        '''

        if len(self) == 0:
            raise ValueError("No messages to update")

        if self[-1]["display"]["role"] != "assistant":
            raise ValueError("Can only update the last assistant message")


        self[-1]["display"]["content"] += delta_content
        self[-1]["ai"]["content"] += delta_content
        
    def add_function_call(self, ai_call):
        
        self.append(
            {
                "display": {
                    "role": "assistant",
                    "content": f"Executing {ai_call['name']} with parameters: {ai_call['arguments']}",
                    "metadata": {
                        "title": f"🔧 Executing {ai_call['name']}",
                        "status": "pending",
                    }
                },
                "ai": ai_call,
            }
        )
    def complete_function_call(self, output, call_id):
        if len(self) == 0 or self[-1]["ai"].get("type") != "function_call":
            raise ValueError("No function call to complete")

        self[-1]["display"]["metadata"]["status"] = "done"

        self.append({
            "display": {
                "role": "assistant",
                "content": output,
                "metadata": {
                    "title": f"📖 Result from {self[-1]['ai']['name']}",
                    "status": "done",
                }
            },
            "ai": {
                "type": "function_call_output",
                "output": output,
                "call_id": call_id
            }
        })
        
    def undo(self, index):
        '''
        Undo to a specific index in the history.
        This removes all messages after the specified index. Reverts to a previous state

        Args:
            index (int): The index to revert to. Must be a user message.
        Returns:
            str: The content of the last user message after undo.
        '''
        if index < 0 or index >= len(self):
            raise ValueError(f"Index out of range for undo: {index}")

        last_message = self[index]['display']['content']

        del self[index:]

        return last_message

    def edit(self, index, new_content):
        '''
        Edit the content of a specific message in the history.
        Will only let you edit user message.
        
        Args:
            index (int): The index of the message to edit.
            new_content (str): The new content for the message.
        '''
        if index < 0 or index >= len(self):
            raise ValueError(f"Index out of range for edit: {index}")

        if self[index]["display"]["role"] != "user":
            raise ValueError("Can only edit user messages")

        self[index]["display"]["content"] = new_content
        self[index]["ai"]["content"] = new_content
        
        del self[index+1:]

    def openai_format(self):
        '''
        Convert to OpenAI message format.
        This means the only two formats are:
        messages with a "role" and "content"
        or function calls with "type", "output", "call_id"
        '''
        return [msg["ai"] for msg in self]


    def gradio_format(self):
        '''
        Convert to Gradio message format. 
        All formats are the same, they must have atleast role and content, but could also have metadata and status.
        '''
        return [msg["display"] for msg in self]

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

        # Clear history to only include the last 5 user or assistant messages, no system messages or tool calls
        msg = [
            m['display']
            for m in history
            if m['display']["role"] in ["user", "assistant"] and m['display'].get("metadata") is None
        ]
        
        if len(msg) > 5:
            msg = msg[-5:]

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
                    *msg,
                ],
            )
            .choices[0]
            .message.content
        )

    def process_input(self, history: CompleteHistory=[]):
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


        if not isinstance(history, CompleteHistory):
            raise ValueError("history must be a CompleteHistory instance")

        while True:
            response = self.openai_client.responses.create(
                model="gpt-4.1",
                instructions=system_message,
                input=history.openai_format(),
                tools=[tool.to_openai_format() for tool in self.tools],
                parallel_tool_calls=True,
                store=False,
                stream=True,
            )

            print("[bold]Assistant response received, processing...[/bold]")

            function_calls = {}
            for chunk in response:
                # Prepare to collect either function calls or text deltas
                if chunk.type == 'response.output_item.added':
                    if chunk.item.type == "function_call":
                        print(f"Function call: {chunk.item.name} {chunk.item.arguments}")
                        function_calls[chunk.output_index] = chunk.item
                    elif chunk.item.type == "message":
                        history.add_message("assistant", "")
                        yield history, history.gradio_format()
                
                # Collect the function call arguments as they stream in
                elif chunk.type == 'response.function_call_arguments.delta':
                    index = chunk.output_index

                    if function_calls[index]:
                        function_calls[index].arguments += chunk.delta
                    
                # Collect the message text as it streams in
                elif chunk.type == 'response.output_text.delta':
                    history.update_last_message(chunk.delta)
                    yield history, history.gradio_format()
                
                # Handle function call
                elif chunk.type == 'response.output_item.done' and chunk.item.type == "function_call":

                    # Handling of functions is sequential, so we will execute them in order.

                    history.add_function_call(chunk.item.to_dict())

                    yield history, history.gradio_format()

                    message = function_calls[chunk.output_index]
                    tool_name = message.name
                    tool_args = json.loads(message.arguments)
                    
                    tool = self.tool_map.get(tool_name)
                    if not tool:
                        result = f"Error: Unknown tool {tool_name}"
                    else:
                        result = tool.execute(**tool_args)
                    
                    history.complete_function_call(
                        output=result, call_id=chunk.item.call_id
                    )
                    yield history, history.gradio_format()
                
                # Handle message done and yield final message
                elif chunk.type == 'response.output_item.done' and chunk.item.type in "message":
                    yield history, history.gradio_format()

            
            # If I have function calls, I need to loop again to get the final answer
            if not function_calls:
                break

        return history, history.gradio_format()
