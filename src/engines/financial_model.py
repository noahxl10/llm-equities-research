import sys
import os
import re
import json
import ast
import uuid

import pandas as pd
from sqlalchemy import and_, update
import sqlalchemy

from src.emailer import Emailer
from src.AZ import Azure
from ..wizard import wizard as wizard_module, model_wizards as model_wizards_module
from ..config import models, SENTIMENT_DICTIONARY
from ..engines.parser import Parser
from dba.db import db_session, Engine as DatabaseEngine
from dba.data_models import (
    FinancialModelMeasurePrompts,
    Companies,
    WizardRequest,
    WizardResponse,
    Segments,
    EngineBuildStatus,
)
import src.excel_editor as excel_editor


class Engine:
    def __init__(self, company):
        self.super_wizard = wizard_module.SuperWizard(model_wizards_module.OAI())
        self.company = company
        self.model = models["OpenAI"]
        self.ticker = self.company.ticker
        self.company_guid = self.company.guid
        self.azure = Azure()
        self.session = db_session

        self.strategy = "assistant"
        self.request_batch_size = 6
        self.assumption_prompt = """"""

    @staticmethod
    def get_first_non_none(*args):
        """Return the first non-None argument."""
        return next((arg for arg in args if arg is not None), None)

    def get_questions(self, prompt_type: str) -> list:
        """Retrieve questions based on the prompt type."""
        prompts = (
            self.session()
            .query(FinancialModelMeasurePrompts)
            .filter_by(prompt_type=prompt_type, company_guid=self.company_guid)
            .all()
        )
        questions = [
            {
                "measure": prompt.measure,
                "command": self.get_first_non_none(prompt.command_1, prompt.command_2),
                "column": prompt.column,
                "section_1": prompt.section_1,
                "section_2": prompt.section_2,
            }
            for prompt in prompts
        ]
        return questions

    def get_companies(self) -> list:
        """Retrieve companies matching the ticker and strategy."""
        companies = (
            self.session()
            .query(Companies)
            .filter(
                Companies.ticker == self.ticker,
                Companies.strategies.contains("financial_model_v1"),
            )
            .all()
        )
        return [
            {
                "ticker": company.ticker,
                "name": company.name,
                "strategies": "financial_model_v1",
            }
            for company in companies
        ]

    @staticmethod
    def remove_substring(substring: str, string: str) -> str:
        """Remove a substring from a string."""
        return string.replace(substring, "")

    def send_requests(self, requests: list) -> list:
        """Send requests using the SuperWizard instance."""
        return self.super_wizard.send_requests(requests)

    def parse_responses(self, responses: list) -> dict:
        """Parse the list of responses into a dictionary."""
        # Implementation needed based on specific parsing logic
        pass

    def get_projections(self, projection_type: str) -> dict:
        """Retrieve projection data based on the projection type."""
        projections_path = ""
        projections_backup_path = ""

        projection_time_periods = self.financial_model.get_projection_time_periods()
        formatted_time_periods = self.format_time_periods(projection_time_periods, projection_type)

        try:
            with open(projections_backup_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            responses = self.read_responses_from_file(projections_path)
            filtered_responses = self.filter_responses(responses)
            return self.parse_responses(filtered_responses)

    def format_time_periods(self, time_periods: list, projection_type: str) -> list:
        """Format time periods based on the projection type."""
        formatted = []
        for period in time_periods:
            period_str = str(period)
            if projection_type == "quarter" and "q" in period_str:
                formatted.append(period_str.replace("q", "quarter"))
            elif projection_type == "year" and "q" not in period_str:
                formatted.append(period_str.replace("year", ""))
        return formatted

    def read_responses_from_file(self, file_path: str) -> str:
        """Read responses from a file."""
        with open(file_path, "r") as file:
            return file.read()

    @staticmethod
    def filter_responses(responses: str) -> list:
        """Filter and clean the raw responses."""
        split_responses = responses.split("||")
        return [response.strip() for response in split_responses if response.strip()]

    def get_historicals(self) -> dict:
        """Retrieve historical data."""
        historicals_path = ""
        historicals_backup_path = ""
        try:
            with open(historicals_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            requests = self.build_requests(projections=False, historicals=True)
            responses = self.collect_responses(requests, historicals_backup_path)
            filtered_responses = self.filter_responses(responses)
            parsed_data = self.parse_responses(filtered_responses)
            print(parsed_data)
            return parsed_data

    def collect_responses(self, requests: list, backup_path: str) -> str:
        """Send requests in batches and collect responses."""
        all_responses = []
        for i in range(0, len(requests), self.request_batch_size):
            batch = requests[i:i + self.request_batch_size]
            batch_responses = self.send_requests(batch)
            all_responses.extend(batch_responses)
            self.append_responses_to_file(batch_responses, backup_path)
        return self.read_responses_from_file(backup_path)

    @staticmethod
    def append_responses_to_file(responses: list, file_path: str):
        """Append responses to a file."""
        with open(file_path, "a") as file:
            for response in responses:
                file.write(f"||{response.response}||\n")

    def get_measure_cell_map(self, responses: list, prompt_type: str) -> list:
        """Map measures to their corresponding cells."""
        measure_cell_map = []
        questions = self.get_questions(prompt_type)
        for measure in responses:
            measure_name = measure["measure"].lower()
            column = self.find_matching_column(measure_name, questions)
            time_period = self.safe_parse_time_period(measure.get("time_period"))
            measure_cell_map.append({
                "measure": measure["measure"],
                "time_period": time_period,
                "amount": measure["amount"],
                "unit": measure["unit"],
                "column": column,
            })
        return measure_cell_map

    def find_matching_column(self, measure_name: str, questions: list) -> str:
        """Find the appropriate column for a given measure name."""
        for question in questions:
            command = question["command"].lower()
            if measure_name in command:
                return question["column"]
            elif all(word in command for word in measure_name.split()):
                return question["column"]
        return ""

    def safe_parse_time_period(self, time_period):
        """Parse the time period safely."""
        try:
            return self.parse_time_period(time_period)
        except Exception:
            return time_period

    def build_requests_from_csv(self, df: pd.DataFrame, regrade: bool, update: bool, prompt_type: str) -> pd.DataFrame:
        """Build requests from a CSV DataFrame."""
        year_map = self.get_year_map()
        response_guids = []

        for _, row in df.iterrows():
            if prompt_type != "projection":
                time_period_type = row["time_period_type"]
                time_periods = year_map.get(row["prompt_type"], {}).get(time_period_type, [])
                financial_statement_prompt = self.construct_financial_statement_prompt(row["applicable_financial_statement"])
                time_period_type_plural = f"{time_period_type}s"

            if prompt_type == "historical":
                request_text = self.construct_historical_request(row)
            elif prompt_type == "segment":
                request_text = self.construct_segment_request(row)
            else:
                request_text = ""

            if prompt_type == "projection":
                response_guids.append(row["response_guid"])
            elif regrade or not row.get("response_guid"):
                response_guid = self.handle_regrade_or_new_request(row, request_text, regrade)
                response_guids.append(response_guid)
            else:
                response_guids.append(row["response_guid"])

        if update:
            self.update_response_guids(df, response_guids)
        else:
            df["response_guid"] = response_guids
            df.to_csv("static/engine/financial_model/segment_responses.csv", index=False)

        return df

    def get_year_map(self) -> dict:
        """Get the mapping of prompt types to their respective time periods."""
        return {
            "model_measure": {
                "year": self.financial_model.historical_years_for_requests,
                "quarter": self.financial_model.historical_quarters_for_requests,
            },
            "segment_percent_of_revenue": {"annual": self.financial_model.historical_years_for_requests},
            "segment_revenue": {"annual": self.financial_model.historical_years_for_requests},
            "segment_growth_rate": {"annual": self.financial_model.historical_years_for_requests},
        }

    @staticmethod
    def construct_financial_statement_prompt(financial_statement: str) -> str:
        """Construct the financial statement prompt."""
        if financial_statement:
            return f"from the {financial_statement} "
        return ""

    def construct_historical_request(self, row: pd.Series) -> str:
        """Construct a historical request based on the row data."""
        # Placeholder for constructing historical request
        return ""

    def construct_segment_request(self, row: pd.Series) -> str:
        """Construct a segment request based on the row data."""
        # Placeholder for constructing segment request
        return ""

    def handle_regrade_or_new_request(self, row: pd.Series, request_text: str, regrade: bool) -> str:
        """Handle regrading or creating a new request."""
        model_request = WizardRequest(
            company_guid=self.company_guid,
            request=request_text,
            model_guid=self.model.guid,
            model_qualified_api_name=self.model.qualified_api_name,
            request_type=self.strategy,
            internal_parameters=row["measure_guid"],
        )
        responses = self.send_requests([model_request])
        try:
            return responses[0].guid
        except IndexError:
            return None

    def update_response_guids(self, df: pd.DataFrame, response_guids: list):
        """Update the response GUIDs in the database."""
        for _, row in df.iterrows():
            stmt = (
                update(FinancialModelMeasurePrompts)
                .where(FinancialModelMeasurePrompts.measure_guid == str(row["measure_guid"]))
                .values(response_guid=str(response_guids[_]))
            )
            self.session.execute(stmt)
        self.session.commit()

    def build_requests(self, projections: bool = False, projection_time_periods: list = None, historicals: bool = True) -> list:
        """Build model requests based on the provided flags."""
        if projection_time_periods is None:
            projection_time_periods = []

        model_requests = []
        if projections:
            questions = self.get_questions("projection")
            model_requests = [
                WizardRequest(
                    model=self.model,
                    company_guid=self.company_guid,
                    request_type=self.strategy,
                    question=""  # Add appropriate question text
                )
                for _ in questions
            ]
        elif historicals:
            questions = self.get_questions("historical")
            model_requests = [
                WizardRequest(
                    model=self.model,
                    company_guid=self.company_guid,
                    request_type=self.strategy,
                    request=""  # Add appropriate request text
                )
                for _ in questions
            ]
        elif hasattr(self, 'segments') and self.segments:
            segment_config = {
                "segment_revenue": "",
                "segment_percent_of_revenue": "",
                "segment_growth_rate": "",
            }
            for segment in self.segments:
                for config_key, config_value in segment_config.items():
                    request_text = config_value.replace("{segment}", segment)
                    model_requests.append(
                        WizardRequest(
                            model=self.model,
                            company_guid=self.company_guid,
                            request_type=self.strategy,
                            request=request_text,
                        )
                    )
        return model_requests

    def get_segments(self, override: bool = False) -> list:
        """Retrieve or generate segments for the company."""
        prompts = (
            self.session.query(FinancialModelMeasurePrompts, WizardResponse)
            .outerjoin(
                WizardResponse,
                FinancialModelMeasurePrompts.response_guid == WizardResponse.guid,
            )
            .filter(
                FinancialModelMeasurePrompts.prompt_type == "segment_strategy",
                FinancialModelMeasurePrompts.company_guid == self.company_guid,
            )
            .order_by(WizardResponse.created_at.desc())
            .all()
        )

        if not prompts:
            return []

        prompt, response = prompts[0]
        question = self.get_first_non_none(prompt.override_core_measure_prompt, prompt.core_measure_prompt)
        response_success = response.is_success if response else False

        if not response_success or override:
            segment_answer = self.fetch_segment_answer(question)
            self.update_segment_response(prompt, response)
            segments = self.parse_segment_answer(segment_answer)
        else:
            segments = self.parse_segment_answer(response.response)

        return segments

    def fetch_segment_answer(self, question: str) -> str:
        """Fetch the segment answer from the model."""
        segment_request = WizardRequest(
            company_guid=self.company_guid,
            model_guid=self.model.guid,
            model_qualified_api_name=self.model.qualified_api_name,
            request_type=self.strategy,
            request=f"For {self.company.name}, {question}",
        )
        responses = self.send_requests([segment_request])
        return responses[0].response if responses else ""

    def update_segment_response(self, prompt, response):
        """Update the segment response in the database."""
        query = (
            update(FinancialModelMeasurePrompts)
            .where(
                FinancialModelMeasurePrompts.company_guid == self.company_guid,
                FinancialModelMeasurePrompts.prompt_type == "segment_strategy",
            )
            .values(response_guid=response.guid if response else None)
        )
        self.session.execute(query)
        self.session.commit()

    def parse_segment_answer(self, segment_answer: str) -> list:
        """Parse the segment answer into a list of segments."""
        segments = []
        if "segments" in segment_answer:
            pos = any(word in segment_answer[:10] for word in SENTIMENT_DICTIONARY["Positive"])
            neg = any(word in segment_answer[:10] for word in SENTIMENT_DICTIONARY["Negative"])
            if pos and not neg:
                segments = self.extract_segments(segment_answer)
        else:
            segments = self.extract_segments(segment_answer)
        return segments

    def extract_segments(self, answer: str) -> list:
        """Extract segments from the answer using regex."""
        matches = re.findall(r"\[[^\]]*\]", answer)
        segments = []
        for match in matches:
            if "†" not in match:
                pattern = r"\[(.*?)\]"
                found = re.search(pattern, match)
                if found:
                    segment = [option.strip() for option in found.group(1).split(",")]
                    segments.append(segment)
        return segments

    def get_prompts(self) -> list:
        """Retrieve all financial model measure prompts."""
        return self.session.query(FinancialModelMeasurePrompts).all()

    def get_segment_responses(self) -> dict:
        """Retrieve and process segment responses."""
        if self.session.query(Segments).filter(Segments.company_guid == self.company_guid).count() == 0:
            self.segments = self.get_segments(override=True)
            new_segment = Segments(company_guid=self.company_guid, segments=str(self.segments))
            self.session.add(new_segment)
            self.session.commit()
        else:
            segments_record = self.session.query(Segments).filter(Segments.company_guid == self.company_guid).first()
            self.segments = ast.literal_eval(segments_record.segments)

        try:
            df = pd.read_sql(
                f"""
                SELECT *
                FROM llm."segment_responses_{self.company_guid}" 
                LEFT JOIN llm.responses ON guid = response_guid::uuid
                """,
                DatabaseEngine,
            )
        except Exception:
            df = self.generate_segment_responses()

        parsed_segment_data = self.parse_segment_data(df)
        return parsed_segment_data

    def generate_segment_responses(self) -> pd.DataFrame:
        """Generate segment responses if they do not exist."""
        prompts = (
            self.session.query(FinancialModelMeasurePrompts)
            .filter(
                FinancialModelMeasurePrompts.prompt_type.notin_([
                    "model_measure",
                    "segment_strategy",
                ]),
                FinancialModelMeasurePrompts.company_guid == self.company_guid,
            )
            .all()
        )
        prompts_dicts = [prompt.__dict__ for prompt in prompts]
        for prompt_dict in prompts_dicts:
            prompt_dict.pop("_sa_instance_state", None)
        df = pd.DataFrame(prompts_dicts)

        segments_df = pd.DataFrame(self.segments, columns=["segments"])
        df["key"] = 1
        segments_df["key"] = 1
        cross_joined_df = pd.merge(df, segments_df, on="key").drop("key", axis=1)
        cross_joined_df = cross_joined_df.rename(columns={"measure_guid": "original_measure_guid"})
        cross_joined_df["measure_guid"] = [uuid.uuid4() for _ in range(len(cross_joined_df))]

        self.build_requests_from_csv(
            cross_joined_df, regrade=True, update=False, prompt_type="segment"
        )
        df = pd.read_csv("")  # Specify the correct CSV path

        df.to_sql(
            f"segment_responses_{self.company_guid}",
            DatabaseEngine,
            schema="llm",
            if_exists="replace",
            index=False,
        )
        df = pd.read_sql(
            f"""
            SELECT *
            FROM llm."segment_responses_{self.company_guid}" 
            LEFT JOIN llm.responses ON guid = response_guid::uuid
            """,
            DatabaseEngine,
        )
        return df

    def parse_segment_data(self, df: pd.DataFrame) -> dict:
        """Parse the segment data into a dictionary."""
        parser = Parser()
        historical_years = self.financial_model.historical_years_for_requests
        segment_dict = {segment: {str(year): {
            "segment_growth_rate": None,
            "segment_percent_of_revenue": None,
            "segment_revenue": None,
        } for year in historical_years} for segment in self.segments}

        for _, row in df.iterrows():
            parsed_results = parser.parse_segments(row.get("response", ""))
            for result in parsed_results:
                time_period = str(result.get("time_period"))
                if time_period not in historical_years:
                    result["amount"] = 0
                prompt_type = row.get("prompt_type")
                segment = row.get("segments")
                if segment and prompt_type:
                    current_value = segment_dict.get(segment, {}).get(time_period, {}).get(prompt_type)
                    if current_value in (None, "0"):
                        segment_dict[segment][time_period][prompt_type] = result.get("amount")
        return segment_dict

    def run(self, build_status_guid: str, recipients: list = None) -> str:
        """Execute the engine run process."""
        self.financial_model = excel_editor.FinancialModel(self.company)
        segment_dict = self.get_segment_responses()

        df = pd.read_sql(
            f"""
            SELECT *
            FROM llm.financial_model_measure_prompts
            LEFT JOIN llm.responses ON guid = response_guid::uuid
            WHERE company_guid = '{self.company_guid}'
              AND prompt_type = 'model_measure'
              AND projection_or_historical = 'historical'
            """,
            DatabaseEngine,
        )

        parser = Parser()
        full_responses = []
        for _, row in df.iterrows():
            parsed = self.parse_financial_measure(row, parser)
            full_responses.extend(parsed)

        export_path, blob_name = self.financial_model.build_financial_model(
            full_responses, segments_dict=segment_dict, segments=self.segments
        )

        self.azure.upload_blob(export_path, blob_name, overwrite=True)

        if recipients:
            for recipient in recipients:
                Emailer().send_financial_model(recipient, self.company.ticker, export_path)

        self.update_build_status(build_status_guid, blob_name)
        os.remove(export_path)

        return blob_name

    def parse_financial_measure(self, row: pd.Series, parser: Parser) -> list:
        """Parse financial measure data from a row."""
        column = row.get("base_column")
        answer = row.get("response")
        format_type = row.get("number_format_type")

        parsed_data = parser.just_parse(
            answer=answer,
            column=column,
            format_type=format_type,
            projection_or_historical=row.get("projection_or_historical"),
            is_yoy=row.get("is_yoy"),
            core_measure_prompt=row.get("core_measure_prompt"),
            number_format_type=row.get("number_format_type"),
        )

        parsed_time_periods = {entry["time_period"] for entry in parsed_data}
        historical_periods = (
            self.financial_model.historical_years_for_requests
            if row.get("time_period_type") == "year"
            else self.financial_model.historical_quarters_for_requests
        )
        missing_periods = [str(period) for period in historical_periods if str(period) not in parsed_time_periods]

        for period in missing_periods:
            parsed_data.append({
                "core_measure_prompt": row.get("core_measure_prompt"),
                "time_period": period,
                "amount": 0,
                "column": column,
                "unit": row.get("example_prompt_unit"),
                "is_yoy": row.get("is_yoy"),
                "number_format_type": row.get("number_format_type"),
            })

        return parsed_data

    def update_build_status(self, build_status_guid: str, blob_path: str):
        """Update the build status in the database."""
        build_status = (
            self.session()
            .query(EngineBuildStatus)
            .filter(EngineBuildStatus.guid == build_status_guid)
            .first()
        )
        if build_status:
            build_status.status = "completed"
            build_status.blob_path = blob_path
            self.session().commit()

    def parse_time_period(self, time_period):
        """Parse the time period. Implementation depends on specific requirements."""
        # Placeholder for actual parsing logic
        return time_period

