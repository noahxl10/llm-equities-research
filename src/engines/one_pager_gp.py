import os
import re
import threading
import traceback
import pandas as pd
from typing import List

from src.AZ import Azure
import src.config as config
import engine_utils as utils
from dba.db import db_session
import src.excel_editor as ee
from src.emailer import Emailer
from ..wizard import wizard as wizard, model_wizards as model_wizards
from dba.data_models import WizardRequest, OnePagerPrompts, Companies, EngineBuildStatus


class Engine:
    def __init__(self, company: Companies):
        self.wizard = wizard.SuperWizard(model_wizards.OAI())
        self.model = config.models["OpenAI"]
        self.company = company
        self.session = db_session
        self.azure = Azure()

    def get_prompts(self) -> List[dict]:
        results = (
            self.session()
            .query(OnePagerPrompts)
            .filter(OnePagerPrompts.company_guid == self.company.guid)
            .all()
        )

        if not results:
            raise ValueError("One Pager: no questions found for this company")

        return [
            {
                "beginning_of_prompt": result.beginning_of_prompt,
                "core_prompt": result.core_prompt,
                "override_core_prompt": result.override_core_prompt,
                "cell": result.cell,
            }
            for result in results
        ]

    def process_local_file(self) -> pd.DataFrame:
        df = pd.read_csv(config.binaried_output_base_path)
        beginning_of_prompt = utils.get_one_pager_beginning_prompt(self.company.ticker)

        for index, row in df.iterrows():
            prompt = f"{beginning_of_prompt}{row['question']}"
            response = self.wizard.send_requests(
                [
                    WizardRequest(
                        model_guid=self.model.guid,
                        model_qualified_api_name=self.model.qualified_api_name,
                        company_guid=self.company.guid,
                        request=prompt,
                        request_type="assistant",
                    )
                ]
            )[0]

            df.loc[index, "response"] = str(response.response)
            df.loc[index, "y_n"] = utils.get_y_n(response.response)
            df.loc[index, "ticker"] = self.company.ticker
            df.loc[index, "company"] = self.company.name

        output_path = f"{config.binaried_output_company_path}_{self.company.guid}.csv"
        df.to_csv(output_path, index=False)
        return df

    def get_binaried_output(self, local_file=False) -> pd.DataFrame:
        if local_file:
            try:
                return pd.read_csv(
                    f"{config.binaried_output_company_path}_{self.company.guid}.csv"
                )
            except FileNotFoundError:
                return self.process_local_file()
        return self.process_local_file()

    def build_requests(self) -> List[WizardRequest]:
        prompts = self.get_prompts()
        return [
            WizardRequest(
                model_guid=self.model.guid,
                model_qualified_api_name=self.model.qualified_api_name,
                company_guid=self.company.guid,
                request=utils.build_one_pager_request_string(
                    self.company.ticker,
                    prompt["beginning_of_prompt"],
                    prompt["override_core_prompt"],
                    prompt["core_prompt"],
                ),
                request_type="assistant",
                internal_parameters=prompt["cell"],
            )
            for prompt in prompts
        ]

    def send_requests(self, requests: List[WizardRequest]) -> List:
        responses = []
        for request in requests:
            try:
                response = self.wizard.send_requests([request])[0]
                responses.append(response)
            except Exception:
                print(traceback.format_exc())
        return responses

    def upload_all_quality_ratings_to_blob(self):
        df = pd.read_csv(os.path.join(self.UPLOAD_FOLDER, self.file))
        tickers = df["ticker"].unique()

        threads = []
        for ticker in tickers:
            thread = threading.Thread(target=self.process_ticker, args=(ticker,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def run(self, recipients=None, build_status_guid=None) -> str:
        segment_requests = self.build_requests()

        segment_responses = [
            {
                "ticker": self.company.ticker,
                "response": re.sub(
                    r"[【\[].*?[】\]]\s*\.", "", response.response
                ).strip(),
                "cell": response.internal_parameters,
            }
            for response in self.send_requests(segment_requests)
        ]

        df = self.get_binaried_output()

        export_path, blob_name = ee.OnePager(self.company).build_one_pager(
            df, segment_responses
        )

        self.azure.upload_blob(export_path, blob_name, overwrite=True)

        if recipients:
            for recipient in recipients:
                Emailer().send_one_pager(recipient, self.company.ticker, export_path)

        os.remove(export_path)

        if build_status_guid:
            build_status = (
                self.session()
                .query(EngineBuildStatus)
                .filter(EngineBuildStatus.guid == build_status_guid)
                .first()
            )
            build_status.status = "completed"
            build_status.blob_path = blob_name
            self.session().commit()

        return blob_name

    def test_run(self):
        Engine(
            Companies(
                name="BURLINGTON STORES, INC.",
                ticker="BURL",
                gp_ticker="BURL US",
                guid="f311d670-406c-434f-b8cc-1319685bf7fd",
            )
        ).run(recipients="")
