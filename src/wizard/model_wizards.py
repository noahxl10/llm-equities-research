
import json
import os
import time
from asyncio import Semaphore
from dataclasses import dataclass
from io import BufferedReader
from typing import Tuple

import google.generativeai as genai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from dba.data_models import Companies, OaiAssistants, WizardRequest, WizardResponse
from dba.db import Engine, db_session
from src.AZ import Azure

from ..decs import time_async_request, time_request, timeout
from .wizard import Wizard


@dataclass
class Gemini(Wizard):
    """
    A Gemini-based wizard for interfacing with Google's generative AI (Gemini).
    Supports chat-based interactions, attribute-engagement contexts, and
    synchronous/asynchronous request handling.
    """

    def __init__(self):
        """Configure the generative AI client using an environment variable for the API key."""
        genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

    async def chat_context(self, message: str, chat=None) -> list:
        """
        Sends a message to the chat context. If no chat exists, one is initiated.

        :param message: The message/prompt to send.
        :param chat: Existing chat session object; if None, a new one is created.
        :return: A list [response_text, chat_session].
        """
        if chat is None:
            chat = self.initiate_chat_context()

        response = chat.send_message(message)
        return [response.text, chat]

    def parse_response(self, response_text: str) -> str:
        """
        Attempts to parse an assumed JSON string from response_text.

        :param response_text: The raw response text (JSON).
        :return: The parsed message content or an error string.
        """
        try:
            response_json = json.loads(response_text)
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, json.JSONDecodeError):
            return "Error: Invalid response format or content"

    def initiate_chat_context(self):
        """Initiates a new Gemini chat context, returning the chat session object."""
        return self.gemini_pro_model.start_chat(history=[])

    async def attr_eng_chat_context(self, attribute_message: str, engagement_message: str) -> Tuple[str, str]:
        """
        Creates two back-to-back messages in the same chat: an 'attribute' prompt
        and an 'engagement' prompt, returning both responses.

        :param attribute_message: The first prompt.
        :param engagement_message: The second prompt.
        :return: Tuple of (attribute_response_text, engagement_response_text).
        """
        attr_response_text, chat = await self.chat_context(attribute_message)
        eng_response_text, _ = await self.chat_context(engagement_message, chat=chat)
        return attr_response_text, eng_response_text

    def send_request(self, request: WizardRequest) -> WizardResponse:
        """
        Synchronously sends a request to a generative model, returning a WizardResponse.
        Currently supports single-prompt requests only.
        """
        try:
            if request.request_type == "single_prompt":
                model = genai.GenerativeModel(request.model_qualified_api_name)
                raw_response = model.generate_content(request.request)
                return WizardResponse(
                    request_guid=request.guid,
                    status="success",
                    raw_api_response=raw_response.__repr__(),
                    response=raw_response.text,
                    is_success=True,
                )
        except Exception as e:
            print(f"Request failed: {e}")
            return WizardResponse(
                request_guid=request.guid,
                status="failed",
                error=str(e),
            )

    @retry(wait=wait_random_exponential(min=1, max=25), stop=stop_after_attempt(2))
    @time_async_request
    @timeout(40)
    async def send_request_async(
        self, request: WizardRequest, semaphore: Semaphore
    ) -> Tuple[WizardResponse, WizardRequest]:
        """
        Asynchronously sends a request to a generative model using a semaphore
        for concurrency management. Retries on failure with an exponential backoff.

        :param request: The WizardRequest object containing prompt info.
        :param semaphore: A Semaphore controlling concurrency.
        :return: (WizardResponse, WizardRequest) for logging or next steps.
        """
        async with semaphore:
            try:
                model = genai.GenerativeModel(request.model_qualified_api_name)
                raw_response = await model.generate_content_async(request.request)
                return (
                    WizardResponse(
                        request_guid=request.guid,
                        status=raw_response.status,
                        raw_api_response=raw_response.__repr__(),
                        request_response=raw_response.text,
                        is_success=True,
                    ),
                    request,
                )
            except Exception as e:
                return (
                    WizardResponse(
                        request_guid=request.guid,
                        status="failed",
                        error=str(e),
                    ),
                    request,
                )


class OAI(Wizard):
    """
    An OpenAI-based wizard for fine-tuning, vector store management, and
    conversation threads. Capable of storing and retrieving documents from
    Azure Blob Storage.
    """

    def __init__(self):
        """Initialize the OpenAI client, Azure, and database session."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.azure = Azure()
        self.session = db_session()

    def upload_file(self, file_path: str):
        """
        Example usage: Upload a file to OpenAI for fine-tuning or other purposes.
        :param file_path: The local path to the file to be uploaded.
        """
        self.client.files.create(file=open("mydata.jsonl", "rb"), purpose="fine-tune")

    def list_files(self):
        """Lists all files on the OpenAI account."""
        return self.client.files.list()

    def fine_tune(self, file_id, model):
        """
        Initiates a fine-tuning job on OpenAI with a given file.
        :param file_id: The ID of the previously uploaded file.
        :param model: The base model to fine-tune against.
        """
        self.client.fine_tuning.jobs.create(training_file=file_id, model=model)

    def create_vector_store(self, company_guid: str):
        """
        Creates a new vector store for a given company (by guid).
        :param company_guid: Unique identifier for the company.
        :return: The newly created vector store object.
        """
        return self.client.beta.vector_stores.create(name=f"financial_statements_for_{company_guid}")

    def create_and_add_files_to_vector_store(
        self, company_guid: str, file_streams: list[BufferedReader]
    ):
        """
        Creates a vector store for the given company and uploads multiple
        file streams to it.
        """
        vector_store = self.create_vector_store(company_guid)
        try:
            self.client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
            )
        except Exception as e:
            print(e)
        return vector_store

    def get_financial_documents_for_company(self, company_guid: str):
        """
        Lists all blobs in Azure that belong to a company's financial filings folder.

        :param company_guid: The unique identifier for the company.
        :return: Filtered list of Azure blob objects matching the company's filings.
        """
        parent_blob_name = f"uploads/{company_guid}/company_filings/"
        blob_list = self.azure.list_blobs()
        return [blob for blob in blob_list if parent_blob_name in blob.name]

    def get_buffered_file_streams(self, company_guid=None, file_paths=None):
        """
        Retrieves buffered file streams from Azure for a given company. 
        If file_paths is provided, it can be used to override or specify local paths.

        :param company_guid: The company's unique identifier.
        :param file_paths: Optional local paths (unused in this snippet).
        :return: A list of in-memory file streams.
        """
        if file_paths is None:
            file_paths = []
        blob_list = self.get_financial_documents_for_company(company_guid)
        return self.azure.get_buffered_streams(blob_list=blob_list)

    def create_message(self, thread, content: str):
        """
        Creates a user message in a given thread.
        :param thread: The thread object to post the message to.
        :param content: Message content (prompt or query).
        """
        return self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content,
        )

    def get_vector_store(self, company_guid: str):
        """
        Retrieves a vector store ID from the OaiAssistants table for the
        specified company. If none is found, builds a new financial file assistant.

        :param company_guid: The company's unique identifier.
        :return: The vector store ID or a newly created store's ID.
        """
        result = (
            self.session.query(OaiAssistants)
            .filter_by(company_guid=company_guid)
            .first()
        )
        if result:
            return result.vector_store_id
        else:
            # If not found, build a new store with a file assistant
            vector_store = self.build_financial_file_assistant()
            return vector_store.id

    def build_financial_file_assistant(self, name="", model="", company=None):
        """
        Helper method to build a file assistant for analyzing a company's
        financial data. Uploads documents to a vector store for retrieval.

        :param name: The name of the assistant to create.
        :param model: The OpenAI model to use.
        :param company: A Companies instance from the database.
        :return: The created assistant object.
        """
        file_assistant = self.build_file_assistant(
            instructions=(
                "You are an expert financial analyst for an investment firm. "
                "You have been asked to analyze a company's financial statements "
                "and provide a true answer."
            ),
            name=name,
            model=model,
            company=company,
        )
        return file_assistant

    def update_vector_store_file(self, company_guid: str, blob_path: str, local_path: str):
        """
        Updates the vector store with a new file. If none exists, creates
        a new store and uploads the file.

        :param company_guid: The unique identifier of the company.
        :param blob_path: The Azure blob path for the file.
        :param local_path: Local file path to be uploaded.
        """
        vector_store_id = self.get_vector_store(company_guid)

        if vector_store_id is None:
            self.create_and_add_files_to_vector_store(
                company_guid, file_streams=[local_path]
            )

        file_message = self.client.files.create(
            file=open(local_path, "rb"), purpose="assistants"
        )

        self.client.beta.vector_stores.files.create_and_poll(
            vector_store_id=vector_store_id, file_id=file_message.id
        )
        return True

    def build_file_assistant(
        self,
        instructions: str,
        name: str,
        model: str,
        company: Companies,
        tools: list = None,
    ):
        """
        Creates an assistant with optional instructions and tools, uploads company docs to a
        vector store, and updates the assistant to reference that store.

        :param instructions: High-level instructions for the assistant.
        :param name: The name for the new assistant.
        :param model: Which OpenAI model to use.
        :param company: A Companies object.
        :param tools: List of tool definitions; defaults to [{'type': 'file_search'}].
        """
        if tools is None:
            tools = [{"type": "file_search"}]

        file_assistant = self.client.beta.assistants.create(
            instructions=instructions,
            name=name,
            tools=tools,
            model=model,
        )

        buffered_file_streams = self.get_buffered_file_streams(company.guid)
        vector_store = self.create_and_add_files_to_vector_store(
            company.guid, buffered_file_streams
        )
        assistant = self.update_assistant(file_assistant, vector_store)

        # Insert a record in OaiAssistants to remember this for future queries
        new_assistant = OaiAssistants(
            company_guid=company.guid,
            assistant_id=assistant.id,
            vector_store_id=vector_store.id,
        )
        self.session.add(new_assistant)
        self.session.commit()

        return assistant

    def update_assistant(self, assistant, vector_store):
        """
        Updates an existing assistant's tool_resources to point to
        the given vector store.

        :param assistant: The existing assistant object.
        :param vector_store: The vector store object to link.
        :return: The updated assistant object.
        """
        return self.client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
        )

    def get_assistant(self, company_guid: str, model_qualified_api_name: str):
        """
        Retrieves an assistant for a company. If none exists in the DB, builds a new one.

        :param company_guid: Unique identifier for the company.
        :param model_qualified_api_name: Which model to use for new builds if needed.
        :return: The retrieved or newly created assistant object.
        """
        result = (
            self.session.query(OaiAssistants)
            .filter_by(company_guid=company_guid)
            .first()
        )

        if result:
            assistant_id = result.assistant_id
        else:
            assistant_id = None

        # Query the company for the ticker
        company = self.session.query(Companies).filter_by(guid=company_guid).first()
        ticker = company.ticker

        if assistant_id is None:
            # If no existing assistant, build a new one
            assistant = self.build_financial_file_assistant(
                name=f"{ticker}_financial_analysis",
                model=model_qualified_api_name,
                company=company,
            )
            assistant_id = assistant.id
        else:
            assistant = self.client.beta.assistants.retrieve(assistant_id)

        return assistant

    def upload_file_to_assistant(self, company_guid: str):
        """
        Uploads all relevant buffered file streams to the company's vector store.
        :param company_guid: The company's unique identifier.
        """
        vector_store_id = self.get_vector_store(company_guid)
        buffered_file_streams = self.get_buffered_file_streams(company_guid)
        self.client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id, files=buffered_file_streams
        )

    def run_assistant(self, requests: list[WizardRequest]):
        """
        Execute a batch of requests (all with the same model and strategy) against an assistant.
        The first message in the run's output holds the summarized response to all user prompts.

        :param requests: A list of WizardRequest objects to process in one run.
        :return: A list of WizardResponse objects correlating to each request.
        """
        # All requests assumed to share the same company_guid/model
        assistant = self.get_assistant(
            requests[0].company_guid, requests[0].model_qualified_api_name
        )
        thread = self.client.beta.threads.create()

        # Create a message for each request in the thread
        for request in requests:
            message = self.create_message(thread, request.request)
            request.external_id = message.id
            self.session.add(request)
            self.session.commit()

        # Poll the thread for completion
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant.id, max_completion_tokens=1000
        )

        responses = []

        if run.status == "completed":
            # The first message typically carries answers to all questions
            sync_cursor_object = self.client.beta.threads.messages.list(thread_id=thread.id)
            messages = sync_cursor_object.data

            for message in messages[0:1]:
                # Attempt to parse out the final content
                try:
                    parsed_message = dict(message.content[0].text)["value"]
                    model_response = WizardResponse(
                        request_guid=request.guid,
                        status=run.status,
                        raw_api_response=str(message),
                        response=parsed_message,
                        is_success=1,
                        internal_parameters=request.internal_parameters,
                    )
                except Exception as e:
                    print(e)
                    model_response = WizardResponse(
                        request_guid=request.guid,
                        status=run.status,
                        raw_api_response=message,
                        error=e,
                    )

                self.session.add(model_response)
                self.session.commit()
                responses.append(model_response)

            return responses
        else:
            print("Run failed")
            print(run)
            print(run.status)
            raise ValueError(f"Run Failed: {run.status}")

    def parse_assistant_messages(self, messages):
        """
        Helper to parse messages from an assistant run, returning the 'value' field
        within the JSON structure of each message.
        :param messages: Collection of messages from the assistant.
        :return: A list of message values.
        """
        parsed_messages = []
        for message in messages:
            try:
                parsed_message = dict(message.content[0].text)
                message_value = parsed_message["value"]
                parsed_messages.append(message_value)
            except Exception:
                pass
        return parsed_messages

    def chat_completion(self, request: WizardRequest) -> WizardResponse:
        """
        Sends a chat-based completion request to OpenAI's API for a single user prompt.
        :param request: The WizardRequest containing the user prompt and model info.
        :return: WizardResponse with the final content or error info.
        """
        role_guidance = "You are a helpful financial analyst assistant."
        try:
            completion = self.client.chat.completions.create(
                model=request.model_qualified_api_name,
                messages=[
                    {"role": "system", "content": role_guidance},
                    {"role": "user", "content": request.request},
                ],
            )
            response = WizardResponse(
                request_guid=request.guid,
                status="success",
                raw_api_response=str(completion.choices),
                response=completion.choices[0].message.content,
                is_success=1,
                internal_parameters=request.internal_parameters,
            )
        except Exception as e:
            response = WizardResponse(
                request_guid=request.guid,
                status="failed",
                error=e,
                internal_parameters=request.internal_parameters,
            )

        self.session.add(response)
        self.session.commit()
        return response

    def send_request(self, request: WizardRequest = None) -> WizardResponse:
        """
        Sends a single chat completion request to OpenAI and returns a WizardResponse.
        """
        return self.chat_completion(request)

    # The below lines demonstrate how you might handle multiple requests
    # or asynchronous chat completions. They are commented out for reference.
    #
    # def send_request(self, requests):
    #     responses = []
    #     for request in requests:
    #         response = self.send_request(request)
    #         responses.append(response)
    #     return responses

    # @retry(wait=wait_random_exponential(min=1, max=25), stop=stop_after_attempt(2))
    # @timeout(40)
    # async def chatcompletion(self, model_name, _input, max_tokens=100):
    #     completion = await openai.ChatCompletion.acreate(
    #         model=model_name,
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": "You are a helpful assistant who must provide a yes or no answer."
    #             },
    #             {"role": "user", "content": _input}
    #         ],
    #         max_tokens=max_tokens
    #     )
    #     return completion
