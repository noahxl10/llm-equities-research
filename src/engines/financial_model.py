"""
Author: Noah Alex
Contact: noahcalex@gmail.com
Year: 2024
Company: Grandeur Peak Global Advisors
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ast
import re
import json
import pandas as pd
from sqlalchemy import and_
import sqlalchemy
from src.emailer import Emailer
from src.AZ import Azure
from ..wizard import wizard as wizard, model_wizards as model_wizards
from ..config import (
    models,
    SENTIMENT_DICTIONARY,
)
from ..engines.parser import Parser
from dba.db import db_session, Engine as eng
from dba.data_models import (
    FinancialModelMeasurePrompts,
    Companies,
    WizardRequest,
    WizardResponse,
    Segments,
    EngineBuildStatus,
)
import src.excel_editor as ee


class Engine:
    def __init__(self, company):
        self.W = wizard.SuperWizard(model_wizards.OAI())
        self.C = company
        self.model = models["OpenAI"]
        self.ticker = self.C.ticker
        self.company_guid = self.C.guid
        self.azure = Azure()
        self.session = db_session

        self.strategy = "assistant"
        self.months = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
        self.request_batch_size = 6
        self.assumption_prompt = """
            , please make assumptions for these future quarterly and annual periods, 
            synthesizing how the variable in question has developed over history, 
            how it’s been trending more recently, and any guidance or relevant commentary from the management team.
        """

    def coalesce(self, *args):
        return next((arg for arg in args if arg is not None), None)

    def get_questions(self, prompt_type) -> dict:
        results = (
            self.session()
            .query(FinancialModelMeasurePrompts)
            .filter_by(prompt_type=prompt_type, company_guid=self.company_guid)
            .all()
        )
        result = [
            {
                "measure": result.measure,
                "command": self.coalesce(result.command_1, result.command_2),
                "column": result.column,
                "section_1": result.section_1,
                "section_2": result.section_2,
            }
            for result in results
        ]
        return result

    def get_companies(self) -> dict:
        results = (
            self.session()
            .query(Companies)
            .filter(
                Companies.ticker == self.ticker,
                Companies.strategies.contains("financial_model_v1"),
            )
            .all()
        )

        result = [
            {
                "ticker": result.ticker,
                "name": result.name,
                "strategies": "financial_model_v1",
            }
            for result in results
        ]
        return result

    def remove_substring_from_string(self, substring, string):
        return string.replace(substring, "")

    def get_missing_responses(self):
        pass

    def send_requests(self, requests):
        responses = self.W.send_requests(requests)
        return responses

    def get_projections(self, projection_type):
        pth = f"static/engine/financial_model/{self.ticker}_projection_data_raw_responses_{projection_type}.txt"
        pth2 = f"static/engine/financial_model/{self.ticker}_projection_data_{projection_type}.json"

        projection_time_periods = self.FinancialModel.get_projection_time_periods()

        projection_time_periods_new = []

        if projection_type == "quarter":
            for time_period in projection_time_periods:
                time_period = str(time_period)
                if "q" in time_period:
                    projection_time_periods_new.append(
                        time_period.replace("q", "quarter")
                    )
        elif projection_type == "year":
            for time_period in projection_time_periods:
                time_period = str(time_period)
                if "q" not in time_period:
                    projection_time_periods_new.append(time_period.replace("year", ""))

        try:
            with open(pth2, "r") as f:
                d = json.load(f)
                return d
        except:
            # projection_requests = self.build_requests(projections=True, projection_time_periods=projection_time_periods_new)

            # responses = []
            # for i in range(0, len(projection_requests), self.request_batch_size):
            #     responses.extend(self.send_requests(projection_requests[i:i+self.request_batch_size]))
            #     with open(pth, 'a') as file:
            #         for response in responses:
            #             file.write(f"||{response.response}||" + '\n')

            with open(pth, "r") as file:
                responses = file.read()

            split_list = responses.split("||")
            filtered_responses = [
                response
                for response in split_list
                if response != "" and response != "\n"
            ]

            d = self.parse_responses(filtered_responses)

    def get_historicals(self):
        pth = f"static/engine/financial_model/{self.ticker}_historical_data.json"
        pth2 = f"static/engine/financial_model/{self.ticker}_historical_data_raw_responses.txt"
        try:
            with open(pth, "r") as f:
                d = json.load(f)
                return d
        except:
            projection_requests = self.build_requests(
                projections=False, historicals=True
            )

            responses = []
            for i in range(0, len(projection_requests), self.request_batch_size):
                responses.extend(
                    self.send_requests(
                        projection_requests[i : i + self.request_batch_size]
                    )
                )
                with open(pth2, "a") as file:
                    for response in responses:
                        file.write(f"||{response.response}||" + "\n")

            with open(pth2, "r") as file:
                responses = file.read()
            split_list = responses.split("||")
            filtered_responses = [
                response
                for response in split_list
                if response != "" and response != "\n"
            ]
            d = self.parse_responses(filtered_responses)
            print(d)

    def get_measure_cell_map(self, responses, prompt_type):
        measure_cell_map = []
        questions = self.get_questions(prompt_type)
        for measure in responses:
            measure_name = measure["measure"].lower()

            for question in questions:
                # print(measure_name)
                # print(question["command"])
                if measure_name in question["command"].lower():
                    col = question["column"]
                    break
                else:
                    splits = measure_name.split(" ")
                    c = 0
                    for word in splits:
                        if word in question["command"].lower():
                            c += 1
                    if len(splits) == c:
                        col = question["column"]
                        break
                    else:
                        col = ""

            try:
                time_period = self.parse_time_period(measure["time_period"])
            except:
                time_period = measure["time_period"]

            measure_cell_map.append(
                {
                    "measure": measure["measure"],
                    "time_period": time_period,
                    "amount": measure["amount"],
                    "unit": measure["unit"],
                    "column": col,
                }
            )

        return measure_cell_map

    def build_requests_from_csv(self, df, regrade, update, prompt_type):
        example_map = {
            "quarters": "Q1 2022",
            "years": "fy2023",
            "annuals": "fy2023",
            "percent": "1.10",
            "percentage": "1.10",
            "thousands": "1000",
        }

        years = self.FinancialModel.historical_years_for_requests
        quarters = self.FinancialModel.historical_quarters_for_requests
        year_map = {
            "model_measure": {"year": years, "quarter": quarters},
            "segment_percent_of_revenue": {"annual": years},
            "segment_revenue": {"annual": years},
            "segment_growth_rate": {"annual": years},
        }

        response_guids = []

        for index, row in df.iterrows():
            if prompt_type != "projection":
                time_period_type = row["time_period_type"]  # "year"

                time_periods = year_map[row["prompt_type"]][time_period_type]

                example_unit = row["example_prompt_unit"]
                initial_prompt = row["beginning_of_prompt"]

                measure = self.coalesce(
                    row["override_core_measure_prompt"], row["core_measure_prompt"]
                )
                end_prompt = row["end_of_prompt"]
                financial_statement = row["applicable_financial_statement"]

                if financial_statement != None:
                    financial_statement_prompt = "from the " + financial_statement + " "
                else:
                    financial_statement_prompt = ""

                time_period_type = f"{time_period_type}s"

            # if prompt_type == "projection":
            # q = f"""
            #     For BURL, I want you to {initial_prompt} {measure} {end_prompt} {financial_statement_prompt}for the future {time_period_type} {time_periods} into nicely formatted brackets
            #         [ {measure}, units if they apply, time period], for example, [{example_map[example_unit]}, {example_unit}, {example_map[time_period]}].
            #         Return just the bracketed response. Please be accurate, synthesizing how the variable in question has developed over history,
            #         how it’s been trending more recently, and any guidance or relevant commentary from the management team.
            # """

            if prompt_type == "historical":
                q = f"""
                    For {self.C.ticker}, I want you to {initial_prompt} {measure} {end_prompt} {financial_statement_prompt} for the historical fiscal {time_period_type} {time_periods} into nicely formatted brackets 
                        [ the number, units if they apply, time period], for example, [{example_map[example_unit]}, {example_unit}, {example_map[time_period_type]}]. 
                        Return just the bracketed response. Please be accurate, using the same exact string/amount that is in the financial statements.
                """
            elif prompt_type == "segment":
                q = f"""
                    For {self.C.ticker}, I want you to answer: {initial_prompt} {measure} for the segment {row["segments"]} historical fiscal {time_period_type} {time_periods} into nicely formatted brackets 
                        [the number, units if they apply, time period], for example, [{example_map[example_unit]}, {example_unit}, {example_map[time_period_type]}]. 
                        Return just the bracketed response. Please be accurate, using the same exact string/amount that is in the financial statements.
                """

            if prompt_type == "projection":
                response_guids.append(row["response_guid"])

            elif regrade:
                print("regrade")
                # if str(row["re_answer"]) == str(1.0):
                model_request = WizardRequest(
                    company_guid=self.company_guid,
                    request=q,
                    model_guid=self.model.guid,
                    model_qualified_api_name=self.model.qualified_api_name,
                    request_type=self.strategy,
                    internal_parameters=row["measure_guid"],
                )

                resp = self.send_requests([model_request])
                try:
                    print(resp[0].response)
                    response_guid = resp[0].guid
                except:
                    response_guid = None

            elif row["response_guid"] == "" or row["response_guid"] is None:
                print("Response is empty. Generating new request.")
                model_request = WizardRequest(
                    company_guid=self.company_guid,
                    request=q,
                    model_guid=self.model.guid,
                    model_qualified_api_name=self.model.qualified_api_name,
                    request_type=self.strategy,
                    internal_parameters=row["measure_guid"],
                )
                resp = self.send_requests([model_request])
                print(resp[0].response)
                response_guid = resp[0].guid
            else:
                response_guid = row["response_guid"]

            response_guids.append(response_guid)
            upd_response_guid = response_guid

            # Construct the update statement
            if update:
                stmt = (
                    sqlalchemy.update(FinancialModelMeasurePrompts)
                    .where(
                        FinancialModelMeasurePrompts.measure_guid
                        == str(row["measure_guid"])
                    )
                    .values(response_guid=str(upd_response_guid))
                )
                self.session.execute(stmt)
                self.session.commit()

        if not update:
            df["response_guid"] = response_guids
            df.to_csv(
                "static/engine/financial_model/segment_responses.csv", index=False
            )
        return df

    def build_requests(
        self, projections=False, projection_time_periods=[], historicals=True
    ):  # -> list[ModelRequest]:
        companies = self.get_companies()

        if projections:
            questions = self.get_questions("projection")
            model_requests = [
                WizardRequest(
                    model=self.model,
                    company_guid=self.company_guid,
                    question=f"""For {question['command'].replace('Populate', '')}, 
                            please make assumptions for these future periods {str(projection_time_periods)}, 
                            synthesizing how the variable in question has developed over history, 
                            how it’s been trending more recently, and any guidance or relevant commentary from the management team.
                            You have to clarify which year and give me the description of measurement, time period, value, and units in quotes and brackets, 
                            like ['revenue', 'Fiscal Year 2022', '40000', 'millions'] or ['revenue', 'Quarter 1 2022', '40000', 'millions']""",
                    request_type=self.strategy,
                )
                for question in questions
            ]

        elif historicals:
            questions = self.get_questions("historical")
            model_requests = [
                WizardRequest(
                    model=self.model,
                    company_guid=self.company_guid,
                    request=f"""
                            Give me {question['command'].replace('Populate', '')} from the given financial documents for as many years 
                            AND quarters as you can find, you have to clarify if they are quarters or full fiscal years and give me the 
                            description of measurement, time period, value, and units in quotes and brackets, like 
                            ['revenue', 'Fiscal Year 2022', '40000', 'millions'] or if it is a quarter amount ['COGS', 'Quarter 1 2022', '40200', 'thousands']""",
                    request_type=self.strategy,
                )
                for company, question in questions
            ]

        # Add segment requests
        elif len(self.segments) > 0:
            model_requests = []

            segment_config = {
                "segment_revenue": "What was the {segment} segment revenue for as many years as you can find? (Please provide the segment, measurement, time period, value, and units in brackets in this format, for example ['ladies activewear', 'revenue', 'fiscal year 2022', '3000', 'millions']  so I can parse easily)",
                "segment_percent_of_revenue": "What was the {segment} segment revenue as a percentage of total revenue for as many years as you can find? (Please provide the measurement, time period, value, and units in brackets in this format, for example ['ladies activewear', 'revenue', 'fiscal year 2022', '3000', 'millions']  so I can parse easily)",
                "segment_growth_rate": "What was the year-over-year growth rate of the {segment} segment revenue for as many years as you can find? (Please provide the measurement, time period, value, and units in brackets in this format, for example ['ladies activewear', 'revenue', 'fiscal year 2022', '3000', 'millions'] so I can parse easily)",
            }

            for segment in self.segments:
                for segment_question in segment_config:
                    model_requests.append(
                        WizardRequest(
                            model=self.model,
                            company_guid=self.company_guid,
                            request=segment_config[segment_question].replace(
                                "{segment}", segment
                            ),
                            request_type=self.strategy,
                        )
                    )

        return model_requests

    def get_segments(self, override=False):
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

        prompt, response = prompts[0]
        if response is None:
            response_is_success = False
        else:
            response_is_success = response.is_success or None

        question = (
            self.coalesce(
                prompt.override_core_measure_prompt, prompt.core_measure_prompt
            )
            or None
        )

        if response is None:
            response = None
        else:
            response_is_success = response.response or None

        # Failed response or no response
        if response_is_success != 1 or override:
            if "segments" in question:
                question = """
                    According to the financial documents, are revenues reported by segments? 
                    Start your response with Yes or No, and if it is reported by segment then 
                    give them all to me listed in brackets, like [online, brick and mortar] """

                segment_request = WizardRequest(
                    company_guid=self.company_guid,
                    model_guid=self.model.guid,
                    model_qualified_api_name=self.model.qualified_api_name,
                    request_type=self.strategy,
                    request=f"For {self.C.name}, {question}",
                )

                responses = self.W.send_requests([segment_request])
                segment_answer = responses[0].response

                pos = False
                neg = False
                for word in SENTIMENT_DICTIONARY["Positive"]:
                    if word in segment_answer[0:10]:
                        pos = True

                for word in SENTIMENT_DICTIONARY["Negative"]:
                    if word in segment_answer[0:10]:
                        neg = True
                if pos and not neg:
                    parse = True

            else:
                question = f"""
                    I would like to {question}.
                    Give them all to me listed in brackets, like [online, brick and mortar] 
                """
                # If you can't get the numbers for a segment/category, then don't include it.
                # Give the segments/categories with their total revenue, pct of total revenue, and year over year growth_rate, to me formatted in brackets, alongside each fiscal year you can gather it for. Format it in brackets like [catogory1, revenue, growth_rate, time_period],[catogory2, revenue, growth_rate, time_period],[catogory3, revenue, growth_rate, time_period]. Just give me the bracketed response. Nothing else.

                segment_request = WizardRequest(
                    company_guid=self.company_guid,
                    model_guid=self.model.guid,
                    model_qualified_api_name=self.model.qualified_api_name,
                    request_type=self.strategy,
                    request=f"For {self.C.name}, {question}",
                )

                responses = self.W.send_requests([segment_request])
                segment_answer = responses[0].response
                parse = True

            query = (
                sqlalchemy.update(FinancialModelMeasurePrompts)
                .where(
                    FinancialModelMeasurePrompts.company_guid == self.company_guid,
                    FinancialModelMeasurePrompts.prompt_type == "segment_strategy",
                )
                .values(
                    response_guid=responses[0].guid,
                )
            )

            self.session.execute(query)
            self.session.commit()

            if parse:
                matches = re.findall(r"\[[^\]]*\]", segment_answer)
                for match in matches:
                    if "†" not in match:
                        pattern = r"\[(.*?)\]"
                        match = re.search(pattern, match)
                        segment = [
                            option.strip() for option in match.group(1).split(",")
                        ]

        else:
            matches = re.findall(r"\[[^\]]*\]", response)
            for match in matches:
                if "†" not in match:
                    pattern = r"\[(.*?)\]"
                    match = re.search(pattern, match)
                    segment = [option.strip() for option in match.group(1).split(",")]
        return segment

    def get_prompts(self):
        prompts = self.db_session.query(FinancialModelMeasurePrompts).all()
        return prompts

    def cloud_run(self):
        pass

    def get_segment_responses(self):
        if (
            self.session.query(Segments)
            .filter(Segments.company_guid == self.company_guid)
            .count()
            == 0
        ):
            self.segments = self.get_segments(override=True)
            segment = Segments(
                company_guid=self.company_guid, segments=str(self.segments)
            )
            self.session.add(segment)
            self.session.commit()
        else:
            self.segments = (
                self.session.query(Segments)
                .filter(Segments.company_guid == self.company_guid)
                .first()
                .segments
            )
        self.segments = ast.literal_eval(self.segments)

        try:
            df = pd.read_sql(
                f"""select 
                                *
                            from llm."segment_responses_{self.company_guid}" 
                            left join llm.responses on guid = response_guid::uuid
                            """,
                eng,
            )
        except Exception:
            results = (
                self.session.query(FinancialModelMeasurePrompts)
                .filter(
                    and_(
                        FinancialModelMeasurePrompts.prompt_type != "model_measure",
                        FinancialModelMeasurePrompts.prompt_type != "segment_strategy",
                        FinancialModelMeasurePrompts.company_guid == self.company_guid,
                    )
                )
                .all()
            )
            results_dicts = [result.__dict__ for result in results]
            for result_dict in results_dicts:
                result_dict.pop("_sa_instance_state", None)
            df = pd.DataFrame(results_dicts)

            values_df = pd.DataFrame(self.segments, columns=["segments"])
            df["key"] = 1
            values_df["key"] = 1

            cross_joined_df = pd.merge(df, values_df, on="key").drop("key", axis=1)
            import uuid

            cross_joined_df = cross_joined_df.rename(
                columns={"measure_guid": "original_measure_guid"}
            )
            cross_joined_df["measure_guid"] = [
                uuid.uuid4() for i in range(len(cross_joined_df))
            ]

            self.build_requests_from_csv(
                cross_joined_df, regrade=True, update=False, prompt_type="segment"
            )
            df = pd.read_csv(
                "/Users/noahalex/develop/granduerpeak/gpllm-core/static/engine/financial_model/segment_responses.csv"
            )

            df.to_sql(
                f"segment_responses_{self.company_guid}",
                eng,
                schema="llm",
                if_exists="replace",
                index=False,
            )
            df = pd.read_sql(
                f"""select 
                                    *
                                from llm."segment_responses_{self.company_guid}" 
                                left join llm.responses on guid = response_guid::uuid
                                """,
                eng,
            )

        parsed_column = []
        segment_dict = {}
        p = Parser()

        historical_years = self.FinancialModel.historical_years_for_requests
        # print(historical_years)
        # exit()
        for index, row in df.iterrows():
            # print(row['response'])
            result = p.parse_segments(row["response"])
            result_years = [r["time_period"] for r in result]
            for historical_year in historical_years:
                if historical_year not in result_years:
                    result.append(
                        {
                            "amount": 0,
                            "unit": row["example_prompt_unit"],
                            "time_period": str(historical_year),
                        }
                    )
            parsed_column.append(result)
        df["parsed_column"] = parsed_column
        # segment_dict = {}

        # initialize empty dictionary
        for segment in self.segments:
            segment_dict[segment] = {}
            for historical_year in historical_years:
                segment_dict[segment][str(historical_year)] = {
                    "segment_growth_rate": None,
                    "segment_percent_of_revenue": None,
                    "segment_revenue": None,
                }

            segment_df = df[df["segments"] == segment]
            for index, row in segment_df.iterrows():
                for result in list(row["parsed_column"]):
                    try:
                        if (
                            segment_dict[segment][str(result["time_period"])][
                                row["prompt_type"]
                            ]
                            != "0"
                            and segment_dict[segment][str(result["time_period"])][
                                row["prompt_type"]
                            ]
                            is not None
                        ):
                            pass
                        else:
                            segment_dict[segment][str(result["time_period"])][
                                row["prompt_type"]
                            ] = result["amount"]
                    except Exception as e:
                        print(e)
                        # print("Year doesn't exist: ", result["time_period"])
        print(json.dumps(segment_dict, indent=2))
        return segment_dict

    def run(self, build_status_guid, recipients=None):
        self.FinancialModel = ee.FinancialModel(self.C)
        segment_dict = self.get_segment_responses()
        # print(json.dumps(segment_dict, indent=2))
        # exit()

        # results = self.session.query(FinancialModelMeasurePrompts).filter(
        #     and_(
        #         FinancialModelMeasurePrompts.projection_or_historical == "historical",
        #         FinancialModelMeasurePrompts.prompt_type != "segment_strategy",
        #         FinancialModelMeasurePrompts.prompt_type != "segment_growth_rate",
        #         FinancialModelMeasurePrompts.prompt_type != "segment_percent_of_revenue",
        #         FinancialModelMeasurePrompts.prompt_type != "segment_revenue",
        #         FinancialModelMeasurePrompts.company_guid == self.company_guid
        #     )
        # ).all()

        # results_dicts = [result.__dict__ for result in results]
        # for result_dict in results_dicts:
        #     result_dict.pop('_sa_instance_state', None)
        # df = pd.DataFrame(results_dicts)

        # self.build_requests_from_csv(df, regrade=True, update=True, prompt_type="historical")

        df = pd.read_sql(
            f"""select 
                                *
                            from llm.financial_model_measure_prompts
                            left join llm.responses on guid = response_guid::uuid
                         where company_guid = '{self.company_guid}'
                         and prompt_type = 'model_measure'
                         and projection_or_historical = 'historical'
                            """,
            eng,
        )
        print(df)
        # exit()
        p = Parser()
        full_responses = []
        for index, row in df.iterrows():
            column = row["base_column"]
            answer = row["response"]
            format_type = row["number_format_type"]

            d = p.just_parse(
                answer,
                column,
                format_type,
                row["projection_or_historical"],
                row["is_yoy"],
                row["core_measure_prompt"],
                row["number_format_type"],
            )

            parsed_time_periods = [r["time_period"] for r in d]
            # parse

            if row["time_period_type"] == "year":
                historical_periods = self.FinancialModel.historical_years_for_requests
            else:
                historical_periods = (
                    self.FinancialModel.historical_quarters_for_requests
                )
            # print(historical_periods)
            # print(parsed_time_periods)
            time_periods_not_in_historical_periods = [
                str(time_period)
                for time_period in historical_periods
                if str(time_period) not in parsed_time_periods
            ]
            # print(time_periods_not_in_historical_periods)
            for time_period in time_periods_not_in_historical_periods:
                d.append(
                    {
                        "core_measure_prompt": row["core_measure_prompt"],
                        "time_period": time_period,
                        "amount": 0,
                        "column": column,
                        "unit": row["example_prompt_unit"],
                        "is_yoy": row["is_yoy"],
                        "number_format_type": row["number_format_type"],
                    }
                )

            full_responses.extend(d)

        export_path, blob_name = self.FinancialModel.build_financial_model(
            full_responses, segments_dict=segment_dict, segments=self.segments
        )

        self.azure.upload_blob(export_path, blob_name, overwrite=True)

        recipients = ["noahxl10@gmail.com"]
        if recipients:
            for recipient in recipients:
                Emailer().send_financial_model(recipient, self.C.ticker, export_path)

        build_status = (
            self.session()
            .query(EngineBuildStatus)
            .filter(EngineBuildStatus.guid == build_status_guid)
            .first()
        )
        build_status.status = "completed"
        build_status.blob_path = blob_name
        self.session().commit()

        # os.remove(export_path)

        return blob_name
