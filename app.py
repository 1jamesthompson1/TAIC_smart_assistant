import uuid
from fastapi import FastAPI, Request, Depends, HTTPException
from starlette.config import Config
from starlette.status import HTTP_302_FOUND
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth, OAuthError
import gradio as gr
from gradio_rangeslider import RangeSlider
import dotenv
import os
import tempfile
from openpyxl import Workbook
import uvicorn
from rich import print
import logging
from azure.data.tables import TableServiceClient
import json
from datetime import datetime
import pandas as pd
import traceback

from backend import Assistant, Searching, Storage

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv(override=True)

# Setup the storage connection
print("[bold green]✓ Initializing Azure Storage connection[/bold green]")
connection_string = f"AccountName={os.getenv('AZURE_STORAGE_ACCOUNT_NAME')};AccountKey={os.getenv('AZURE_STORAGE_ACCOUNT_KEY')};EndpointSuffix=core.windows.net"
client = TableServiceClient.from_connection_string(conn_str=connection_string)
knowledgesearchlogs_table_name = os.getenv("KNOWLEDGE_SEARCH_LOGS_TABLE_NAME")
conversation_metadata_table_name = os.getenv("CONVERSATION_METADATA_TABLE_NAME")
conversation_container_name = os.getenv("CONVERSATION_CONTAINER_NAME")

# Check all environment variables are set
if not all([knowledgesearchlogs_table_name, conversation_metadata_table_name, conversation_container_name]):
    missing_vars = [var for var in ["KNOWLEDGE_SEARCH_LOGS_TABLE_NAME", "CONVERSATION_METADATA_TABLE_NAME", "CONVERSATION_CONTAINER_NAME"] if not locals()[var]]
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

knowledgesearchlogs = client.create_table_if_not_exists(
    table_name=knowledgesearchlogs_table_name
)
chatlogs = client.create_table_if_not_exists(table_name=conversation_metadata_table_name)
blob_store = Storage.ConversationBlobStore(connection_string, conversation_container_name)
conversation_store = Storage.ConversationMetadataStore(chatlogs, blob_store)
print("[bold green]✓ Blob storage conversation store initialized[/bold green]")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Azure AD OAuth settings
AZURE_CLIENT_ID = os.getenv("CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AZURE_TENANT_ID = os.getenv("TENANT_ID")
SECRET_KEY = os.urandom(24)

config_data = {
    "AZURE_CLIENT_ID": AZURE_CLIENT_ID,
    "AZURE_CLIENT_SECRET": AZURE_CLIENT_SECRET,
}
starlette_config = Config(environ=config_data)
oauth = OAuth(starlette_config)
oauth.register(
    name="azure",
    client_id=AZURE_CLIENT_ID,
    client_secret=AZURE_CLIENT_SECRET,
    server_metadata_url=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0/.well-known/openid-configuration",
    authorize_url=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/authorize",
    authorize_params=None,
    access_token_url=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/token",
    access_token_params=None,
    refresh_token_url=None,
    redirect_uri="/auth",
    client_kwargs={"scope": "openid profile email"},
)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


# Dependency to get the current user
def get_user(request: Request):
    user = request.session.get("user")
    if user:
        return user["name"]
    raise HTTPException(
        status_code=HTTP_302_FOUND,
        detail="Not authenticated",
        headers={"Location": "/login-page"},
    )


@app.get("/")
def public(user: dict = Depends(get_user)):
    if user:
        return RedirectResponse(url="/tools")
    else:
        return RedirectResponse(url="/login-page")


@app.route("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/")


@app.route("/login")
async def login(request: Request):
    # Forcing deployed url to use https. This is because it seems to default to using http.
    if "localhost" not in request.url.hostname and "https" not in request.url.scheme:
        redirect_uri = "https://" + request.url.hostname + request.url_for("auth").path
    else:
        redirect_uri = request.url_for("auth")
    return await oauth.azure.authorize_redirect(request, redirect_uri)


@app.route("/auth")
async def auth(request: Request):
    try:
        token = await oauth.azure.authorize_access_token(request)
        user = await oauth.azure.parse_id_token(token, token["userinfo"]["nonce"])
    except OAuthError:
        return RedirectResponse(url="/")
    request.session["user"] = user
    return RedirectResponse(url="/")


# =====================================================================
#
# Assistant UI functions
#
# =====================================================================
def handle_undo(history, undo_data: gr.UndoData):
    return history[: undo_data.index], history[undo_data.index]["content"]

def handle_edit(history, edit_data: gr.EditData):
    new_history = history[:edit_data.index+1]
    new_history[-1 if len(new_history) > 0 else 0]['content'] = edit_data.value
    return new_history, None

def handle_submit(user_input, history=None):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})

    return gr.Textbox(interactive=False, value=None), history


def create_or_update_conversation(request: gr.Request, conversation_id, history):
    if history == []:  # ignore instance when history is empty
        return
    
    username = request.username
    
    # Generate conversation title if this is a new conversation
    conversation_title = assistant_instance.provide_conversation_title(history)
    
    # Store using blob storage (JSON in blob + metadata in table)
    success = conversation_store.create_or_update_conversation(
        username=username,
        conversation_id=conversation_id,
        history=history,
        conversation_title=conversation_title
    )
    
    if success:
        print(f"[bold green]✓ Stored conversation {conversation_id} in blob storage[/bold green]")
    else:
        print(f"[bold red]✗ Failed to store conversation {conversation_id}[/bold red]")


def get_user_conversations_metadata(request: gr.Request):
    username = request.username
    
    # Get only conversation metadata (no full message history)
    conversations = conversation_store.get_user_conversations_metadata(username)
    print(f"[bold green]✓ Retrieved metadata for {len(conversations)} conversations[/bold green]")
    return conversations


def load_conversation(request: gr.Request, conversation_id: str):
    username = request.username

    print(f"[orange]Loading conversation {conversation_id} for user {username}[/orange]")
    
    # Load the full conversation with message history
    conversation = conversation_store.load_single_conversation(username, conversation_id)
    if conversation:
        print(f"[bold green]✓ Loaded conversation {conversation_id}[/bold green]")
        formatted_id = f"`{conversation['id']}`"
        formatted_title = f"**{conversation['conversation_title']}**"
        return conversation["messages"], formatted_id, formatted_title
    else:
        print(f"[bold red]✗ Failed to load conversation {conversation_id}[/bold red]")
        return [], f"`{conversation_id}`", "*Failed to load*"


searching_instance = Searching.Searcher(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    voyageai_api_key=os.getenv("VOYAGEAI_API_KEY"),
    db_uri=os.getenv("VECTORDB_PATH"),
)

assistant_instance = Assistant.Assistant(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    searcher=searching_instance,
)

# =====================================================================
#
# Knowledge search UI functions
#
# =====================================================================


def perform_search(
    username: str,
    query: str,
    year_range: list[int],
    document_type: list[str],
    modes: list[str],
    agencies: list[str],
    relevance: float,
):
    error, error_trace = None, None
    created_search, search, clean_results = True, True, True

    search_start_time, search_settings = None, None
    try:
        search_start_time = datetime.now()
        search_type = (
            "none"
            if (query == "" or query is None)
            else ("fts" if query[0] == '"' and query[-1] == '"' else "vector")
        )
        if search_type == "fts":
            query = query[1:-1]

        document_type_mapping = {
            "Safety Issues": "safety_issue",
            "Safety Recommendations": "recommendation",
            "Report sections": "report_section",
            "Entire Reports": "report_text",
        }

        mapped_document_type = [
            document_type_mapping[dt]
            for dt in document_type
            if dt in document_type_mapping
        ]
        search_settings = {
            "query": query,
            "year_range": (year_range[0], year_range[1]),
            "document_type": mapped_document_type,
            "modes": modes,
            "agencies": agencies,
            "type": search_type,
            "limit": 5000,
            "relevance": relevance,
        }
    except Exception as e:
        print(f"[bold red]Error in search settings: {e}[/bold red]")
        error = e
        error_trace = traceback.format_exc()
        search_settings = None
        created_search = False
        search = False
        clean_results = False

    results, info, plots = None, None, None
    if error is None:
        try:
            results, info, plots = searching_instance.knowledge_search(
                **search_settings
            )
        except Exception as e:
            print(f"[bold red]Error in search: {e}[/bold red]")
            error = e
            error_trace = traceback.format_exc()
            search = False
            clean_results = False

    results_to_download, download_dict = None, None
    if error is None:
        try:
            results_to_download = results.copy()
            # Format the results to be displayed in the dataframe
            if not results.empty:
                results["agency_id"] = results.apply(
                    lambda x: f"<a href='{x['url']}' style='color: #1a73e8; text-decoration-line: underline;'>{x['agency_id']}</a>",
                    axis=1,
                )
                results.drop(columns=["url"], inplace=True)
                message = f"""Found {info["relevant_results"]} results from database.  
        _These are the relevant results (out of {info["total_results"]}) from the search of the database, there is a no guarantee of its completeness._"""
            else:
                message = "No results found for the given criteria."
                # Ensure plots are None if no results
                plots = {
                    k: None
                    for k in ["document_type", "mode", "year", "agency", "event_type"]
                }

            download_dict = {
                "settings": search_settings,
                "results": results_to_download,
                "search_start_time": search_start_time,
                "username": username,
            }
        except Exception as e:
            print(f"[bold red]Error in formatting results: {e}[/bold red]")
            error = e
            error_trace = traceback.format_exc()
            clean_results = False

    # Logging search
    basics = {
        "PartitionKey": username,
        "RowKey": str(uuid.uuid4()),
        "search_start_time": search_start_time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    search = (
        {
            "settings": json.dumps(search_settings),
        }
        if created_search
        else {
            "settings": None,
        }
    )
    
    results_log = (
        {
            "results": results.head(100)
            .drop(columns=["document"])
            .to_json(orient="records"),
            "relevant_results": info["relevant_results"],
            "total_results": info["total_results"],
        }
        if clean_results
        else {
            "results": None,
            "relevant_results": None,
            "total_results": None,
        }
    )

    # Make sure that the results are not too large to be stored in the usage logs. I dont need all the items but having an idea of what documents were returned is useful.
    if results_log["results"] is not None:
        results_size = len(results_log["results"])
        while results_size > 32_000:
            print("Trimming results to fit size limit, current size:", results_size)
            current_df = pd.read_json(results_log["results"])
            if current_df.shape[0] > 20:
                results_log["results"] = current_df.iloc[:-10].to_json(
                    orient="records"
                )
            else:
                results_log["results"] = current_df.iloc[:-1].to_json(
                    orient="records"
                )
            results_size = len(results_log["results"])



    knowledgesearchlogs.create_entity(
        entity={
            **basics,
            **search,
            **results_log,
            "error": error_trace if error else None,
        }
    )

    if error is not None:
        raise gr.Error(
            f"An error has occurred during your search, please refresh page and try again.\nError: {error}",
            duration=5,
        )

    # Return plots along with results and message
    return (
        results,
        download_dict,
        message,
        plots.get("document_type"),
        plots.get("mode"),
        plots.get("year"),
        plots.get("agency"),
        plots.get("event_type"),
    )


def update_download_button(download_dict: dict):
    """
    Update the download button to to point to a new temporary file that is the results ready to be downloaded.
    """
    if download_dict["results"] is None or download_dict["results"].empty:
        return gr.DownloadButton(visible=False)
    else:
        save_name = f"{download_dict['settings']['query'][:min(20, len(download_dict['settings']['query']))]}_{download_dict['search_start_time'].strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, save_name)

        summary_data = [
            [
                "This spreadsheet contains the results of a search conducted using the TAIC knowledge search tool.",
                "",
            ],
            [
                f"The search was conducted on {download_dict['search_start_time'].strftime('%Y-%m-%d %H:%M:%S')} by {download_dict['username']}. The settings used for the search can be found below and the full results can be found in the 'Results' sheet.",
                "",
            ],
        ]
        for key, value in download_dict["settings"].items():
            summary_data.append([f"{key}:", str(value)])
        summary_df = pd.DataFrame(summary_data, columns=["Setting", "Value"])

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, header=False, sheet_name="Summary")
            download_dict["results"].to_excel(writer, index=False, sheet_name="Results")


        return gr.DownloadButton(
            label="Download results (spreadsheet)",
            value=file_path,
            visible=True,
        )


# =====================================================================
#
# Gradio UI
#
# =====================================================================


def get_welcome_message(request: gr.Request):
    return request.username, f"Data last updated: {searching_instance.last_updated}"


def get_user_name(request: gr.Request):
    return request.username


TAIC_theme = gr.themes.Default(
    primary_hue=gr.themes.utils.colors.Color(
        name="primary_hue",
        c50="#2e679c",
        **{f"c{n}": "#2e679c" for n in range(100, 901, 100)},
        c950="#2e679c",
    ),
    secondary_hue=gr.themes.utils.colors.Color(
        name="secondary_hue",
        c50="#e6dca1",
        **{f"c{n}": "#e6dca1" for n in range(100, 901, 100)},
        c950="#e6dca1",
    ),
    neutral_hue="gray",
)


def get_footer():
    return gr.HTML(f"""
<style>                   
    .custom-footer {{
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        background-color: {TAIC_theme.primary_50};
        color: {TAIC_theme.neutral_50};
    }}
</style>
<div class="custom-footer">
    <p>Created by <a href="https://github.com/1jamesthompson1">James Thompson</a> for the <a href="https://www.taic.org.nz">New Zealand Transport Accident Investigation Commission.</a></p>
    <p>Contact directed to <a href="mailto:james.thompson@taic.org.nz">james.thompson@taic.org.nz</a> or for suggestions and/or bug reports please use the provided <a href="https://forms.office.com/Pages/ResponsePage.aspx?id=RmxQlKGu1key34UuP4dPFavrlUtUJCpGvY1oQw3ObrlUQjZSTFRFUDRZRk8wUUxPWkVYVEw1SUVDUy4u" target="_blank">feeback form</a>.</p>
    <p>Project is being developed openly on <a href="https://github.com/1jamesthompson1/TAIC_smart_assistant">https://github.com/1jamesthompson1/TAIC-report-summary</a></p>
    <p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0<img style="height:22px!important;margin-left:3px;vertical-align:middle;display: inline-block;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt="CC logo"><img style="height:22px!important;margin-left:3px;vertical-align:middle;display: inline-block;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt="BY logo"></a></p>                  
</div>
""")


with gr.Blocks(
    title="TAIC smart tools",
    theme=TAIC_theme,
    fill_height=True,
    fill_width=True,
    head='<link rel="icon" href="/static/favicon.png" type="image/png">',
) as smart_tools:
    username = gr.State()
    smart_tools.load(get_user_name, inputs=None, outputs=username)

    user_conversations = gr.State([])
    smart_tools.load(get_user_conversations_metadata, inputs=None, outputs=user_conversations)

    with gr.Row():
        gr.Markdown("# TAIC smart tools")
        data_update = gr.Markdown("Data last updated: ")
        gr.Markdown("Logged in as:")
        username_display = gr.Markdown()
        logout_button = gr.Button("Logout", link="/logout")

    smart_tools.load(
        get_welcome_message, inputs=None, outputs=[username_display, data_update]
    )

    with gr.Tabs():
        with gr.TabItem("Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Current conversation")
                    with gr.Group():
                        gr.Markdown("**Conversation ID:**")
                        conversation_id = gr.Markdown(f"`{str(uuid.uuid4())}`")
                        gr.Markdown("**Title:**")
                        conversation_title = gr.Markdown("*New conversation*")
                    gr.Markdown("## Previous conversations")

                    @gr.render(inputs=[user_conversations])
                    def render_conversations(conversations):
                        for conversation in conversations:
                            with gr.Row():
                                gr.Markdown(
                                    f"### {conversation['conversation_title']}",
                                    container=False,
                                )

                                def create_load_function(conv_id):
                                    def load_func(request: gr.Request):
                                        return load_conversation(request, conv_id)
                                    return load_func

                                gr.Button("load").click(
                                    fn=create_load_function(conversation["id"]),
                                    inputs=None,
                                    outputs=[chatbot_interface, conversation_id, conversation_title],
                                )

                with gr.Column(scale=3):
                    with gr.Row():
                        gr.Markdown("### Chat: ")

                    chatbot_interface = gr.Chatbot(
                        type="messages",
                        height="90%",
                        min_height=400,
                        show_label=False,
                        watermark="This paragraph was generated by AI, please check thoroughly before using.",
                        editable="user",
                        avatar_images=(
                            None,
                            "https://www.taic.org.nz/themes/custom/taic/favicon/android-icon-192x192.png",
                        ),
                    )

                    input_text = gr.Textbox(
                        placeholder="Please type your message here...",
                        show_label=False,
                        submit_btn="Send",
                    )
            chatbot_interface.undo(
                fn=handle_undo,
                inputs=chatbot_interface,
                outputs=[chatbot_interface, input_text],
            )
            chatbot_interface.edit(
                fn=handle_edit,
                inputs=chatbot_interface,
                outputs=[chatbot_interface, input_text],
                queue=False,
            ).then(
                assistant_instance.process_input,
                inputs=[chatbot_interface],
                outputs=[chatbot_interface],
            ).then(
                create_or_update_conversation,
                inputs=[conversation_id, chatbot_interface],
                outputs=None,
            ).then(
                get_user_conversations_metadata,
                inputs=None,
                outputs=user_conversations,
            ).then(
                lambda: gr.Textbox(interactive=True),
                None,
                input_text,
            )

            chatbot_interface.clear(
                lambda: (f"`{str(uuid.uuid4())}`", [], "*New conversation*"),
                None,
                [conversation_id, chatbot_interface, conversation_title],
                queue=False,
            ).then(
                get_user_conversations_metadata,
                inputs=None,
                outputs=user_conversations,
            )

            input_text.submit(
                fn=handle_submit,
                inputs=[input_text, chatbot_interface],
                outputs=[input_text, chatbot_interface],
                queue=False,
            ).then(
                assistant_instance.process_input,
                inputs=[chatbot_interface],
                outputs=[chatbot_interface],
            ).then(
                create_or_update_conversation,
                inputs=[conversation_id, chatbot_interface],
                outputs=None,
            ).then(
                get_user_conversations_metadata,
                inputs=None,
                outputs=user_conversations,
            ).then(
                lambda: gr.Textbox(interactive=True),
                None,
                input_text,
            )

        with gr.TabItem("Knowledge Search"):
            search_results_to_download = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    query = gr.Textbox(label="Search Query")
                    search_button = gr.Button("Search")
                    with gr.Row():
                        with gr.Column():
                            search_summary = gr.Markdown()
                        with gr.Column():
                            download_button = gr.DownloadButton(
                                "Download results", visible=False
                            )
                with gr.Column(scale=1):
                    with gr.Accordion("Advanced Search Options", open=True):
                        current_year = datetime.now().year
                        year_range = RangeSlider(
                            label="Year Range",
                            minimum=2000,
                            maximum=current_year,
                            step=1,
                            value=[2007, current_year],
                        )
                        document_type = gr.CheckboxGroup(
                            label="Document Type",
                            choices=[
                                "Safety Issues",
                                "Safety Recommendations",
                                "Report sections",
                                "Entire Reports",
                            ],
                            value=["Safety Issues", "Safety Recommendations"],
                        )
                        modes = gr.CheckboxGroup(
                            label="Modes of Transport",
                            choices=["Aviation", "Rail", "Maritime"],
                            value=["Aviation", "Rail", "Maritime"],
                            type="index",
                        )
                        agencies = gr.CheckboxGroup(
                            label="Agencies",
                            choices=["TAIC", "ATSB", "TSB"],
                            value=["TAIC"],
                        )
                        relevance = gr.Slider(
                            label="Relevance",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.6,
                        )

                    search = [
                        username,
                        query,
                        year_range,
                        document_type,
                        modes,
                        agencies,
                        relevance,
                    ]
            with gr.Accordion(label="Result graphs", open=False):
                with gr.Row():
                    doc_type_plot = gr.Plot()
                    mode_plot = gr.Plot()
                    agency_plot = gr.Plot()
                    year_hist = gr.Plot()
                    event_type_plot = gr.Plot()

            with gr.Row():
                search_results = gr.Dataframe(
                    max_chars=1500,
                    max_height=10000,
                    pinned_columns=1,
                    wrap=True,
                    type="pandas",
                    datatype=[
                        "number",
                        "str",
                        "str",
                        "str",
                        "number",
                        "str",
                        "str",
                        "str",
                        "html",
                        "str",
                    ],
                    show_fullscreen_button=True,
                    show_search="search",
                )

            search_outputs = [
                search_results,
                search_results_to_download,
                search_summary,
                doc_type_plot,
                mode_plot,
                year_hist,
                agency_plot,
                event_type_plot,
            ]

            search_event = search_button.click(
                perform_search, inputs=search, outputs=search_outputs
            ).then(update_download_button, search_results_to_download, download_button)
            query.submit(
                perform_search,
                inputs=search,
                outputs=search_outputs,
            ).then(update_download_button, search_results_to_download, download_button)
    footer = get_footer()

app = gr.mount_gradio_app(
    app, smart_tools, path="/tools", auth_dependency=get_user, show_api=False
)


# Add head parameter to login_page Blocks
with gr.Blocks(
    title="TAIC smart tools login",
    theme=TAIC_theme,
    fill_height=True,
    head='<link rel="icon" href="/static/favicon.png" type="image/png">',
    css="""
.complete-center {
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
}
""",
) as login_page:
    with gr.Column(elem_classes="complete-center"):
        gr.Markdown("# TAIC smart tools")
        gr.Markdown("Please login to continue:")
        gr.Button("Login", link="/login")

    footer = get_footer()


app = gr.mount_gradio_app(app, login_page, path="/login-page", show_api=False)

if __name__ == "__main__":
    uvicorn.run(app)
