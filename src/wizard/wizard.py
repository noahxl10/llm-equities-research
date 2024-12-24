
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from sqlalchemy import Column, insert, select, and_, MetaData, Table, UUID

from ..decs import timeout, time_request
from .._types import ModelRequest, ModelResponse
from dba.data_models import WizardRequest, WizardResponse


# Base class acts as a template for all sub wizards
@dataclass
class Wizard(ABC):
    db = None


class SuperWizard:
    def __init__(self, sub_wizard):
        self.sub_wizard = sub_wizard

    def add_unanswered_requests(self, requests: list[ModelRequest]):
        session = None
        engine = None

        incoming_request_uuids = [request.id for request in requests]

        query = select(WizardRequest).where(
            and_(
                WizardRequest.id.in_(incoming_request_uuids),
                WizardRequest.is_success == 1,
            )
        )

        if len(incoming_request_uuids) > 1000:
            incoming_request_stage_table = Table(
                "staging_table", Column("id", UUID(as_uuid=True), primary_key=True)
            )
            incoming_request_stage_table.create(engine, checkfirst=True)

            session.execute(
                incoming_request_stage_table.insert(), incoming_request_uuids
            )
            session.commit()

            query = (
                select(WizardRequest)
                .select_from(
                    WizardRequest.__table__.join(
                        incoming_request_stage_table,
                        WizardRequest.id == incoming_request_stage_table.c.id,
                    )
                )
                .where(WizardRequest.is_success == 1)
            )

        matching_rows = session.execute(query).scalars().all()

        return matching_rows

    def insert_request(self, request: ModelRequest, response: ModelResponse):
        stmt = insert(WizardRequest).values(
            guid=request.id,
            model_guid=request.model.id,
            request_type=request.request_type,
            request=request.request,
            request_hash=request.request_hash,
            time_to_complete_request=response.time_to_complete_request,
        )

    async def send_sub_request_async(self, request: ModelRequest, session, semaphore):
        runtime, response = self.sub_wizard.send_request_async(
            request, session, semaphore
        )
        self.insert_request(request, response)

    def send_sub_request(self, request: ModelRequest):
        response = self.sub_wizard.send_request(request)
        # response.seconds_to_complete_request = runtime
        # self.insert_request(request, response)
        return response

    async def send_requests_async(
        self, requests: list[ModelRequest], prioritize_new: bool = False
    ):
        # find what requests need to be answered still

        semaphore = asyncio.Semaphore(self.sub_wizard.REQUESTS_PER_BATCH)
        async with aiohttp.ClientSession() as session:
            while True:
                if requests == []:
                    break

                if not prioritize_new:
                    requests = self.add_unanswered_requests(requests)

                task_list = [
                    self.send_sub_request_async(request, session, semaphore)
                    for request in requests
                ]

                L = await asyncio.gather(*task_list)

    def send_requests(
        self, requests: list[WizardRequest], prioritize_new: bool = False
    ):
        respnoses = []

        request_type_dict: dict[str, list[WizardRequest]] = {}
        for request in requests:
            if request.request_type not in request_type_dict:
                request_type_dict[request.request_type] = []
            request_type_dict[request.request_type].append(request)

        for strategy in request_type_dict:
            if strategy == "assistant":
                responses = self.sub_wizard.run_assistant(request_type_dict[strategy])
            if strategy == "one_pager_v1":
                pass
            if strategy == "single_prompt":
                responses = [
                    self.send_sub_request(request)
                    for request in request_type_dict[strategy]
                ]
        return responses
