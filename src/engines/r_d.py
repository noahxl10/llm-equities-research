import traceback

import pandas as pd
from sqlalchemy import and_, or_, text, update
from sqlalchemy.exc import SQLAlchemyError
import src.config as config
import src.engines.engine_utils as utils
from dba.data_models import (
    Companies,
    QuestionRepositoryResearch,
    ResponseBase,
    WizardRequest,
)
from dba.db import db_session
from src.utils import now

from ..wizard import model_wizards as model_wizards
from ..wizard import wizard as wizard



def send_requests(group_key=None, ask_for_year=False):
    W = wizard.SuperWizard(model_wizards.OAI())
    model = config.models["OpenAI"]

    try:
        with db_session() as session:
            response_base_rows = (
                session.query(ResponseBase, Companies, QuestionRepositoryResearch)
                .join(Companies, ResponseBase.company_guid == Companies.guid)
                .join(
                    QuestionRepositoryResearch,
                    ResponseBase.question_guid
                    == QuestionRepositoryResearch.question_guid,
                )
                .filter(
                    and_(
                        or_(ResponseBase.success == 0, ResponseBase.success.is_(None)),
                        ResponseBase.year_asked_for == '2022'
                        # ResponseBase.group_key == group_key,
                    )
                )
                .order_by(ResponseBase.company_guid.desc())
                .all()
            )


        for (
            response_base,
            companies,
            question_repository_research,
        ) in response_base_rows:
            try:
                if ask_for_year:
                    prompt = f"" # REMOVED
                else:
                    prompt = f"" # REMOVED
                print(prompt)
                resp = W.send_requests(
                    [
                        WizardRequest(
                            model_guid=model.guid,
                            model_qualified_api_name=model.qualified_api_name,
                            company_guid=response_base.company_guid,
                            request=prompt,
                            request_type="assistant",
                            # request_type="single_prompt",
                        )
                    ]
                )
                split_response = resp[0].response

                val = utils.get_y_n(split_response)
                if str(val) == '-1':
                    print(split_response)
                
                with db_session() as session:
                    stmt = (
                        update(ResponseBase)
                        .where(
                            and_(
                                ResponseBase.company_guid == response_base.company_guid,
                                ResponseBase.question_guid
                                == response_base.question_guid,
                                ResponseBase.year_asked_for == response_base.year_asked_for,
                                # ResponseBase.group_key == group_key,
                            )
                        )
                        .values(
                            success=1,
                            response_guid=resp[0].guid,
                            parsed_response=val,
                            modified_at=now(),
                        )
                    )

                    # Execute the update
                    session.execute(stmt)
                    session.commit()

            except Exception:
                print(traceback.format_exc())
                continue

    except SQLAlchemyError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def re_ask_requests(group_key, year_to_ask_for='2023'):
    W = wizard.SuperWizard(model_wizards.OAI())
    model = config.models["OpenAI"]

    try:
        with db_session() as session:
            response_base_rows = (
                session.query(ResponseBase, Companies, QuestionRepositoryResearch)
                .join(Companies, ResponseBase.company_guid == Companies.guid)
                .join(
                    QuestionRepositoryResearch,
                    ResponseBase.question_guid
                    == QuestionRepositoryResearch.question_guid,
                )
                .filter(
                    and_(
                        ResponseBase.parsed_response == '-1',
                        ResponseBase.group_key == group_key,
                    )
                )
                .order_by(ResponseBase.company_guid.desc())
                .all()
            )


        for (
            response_base,
            companies,
            question_repository_research,
        ) in response_base_rows:
            try:

                prompt = f"Answer the following question for the company {companies.gp_ticker} with a 'yes' or 'no' first, specifically for the fiscal year {year_to_ask_for}: {question_repository_research.question}"
                resp = W.send_requests(
                    [
                        WizardRequest(
                            model_guid=model.guid,
                            model_qualified_api_name=model.qualified_api_name,
                            company_guid=response_base.company_guid,
                            request=prompt,
                            request_type="single_prompt",
                        )
                    ]
                )
                split_response = resp[0].response

                val = utils.get_y_n(split_response)
                if str(val) == '0' or str(val) == '1':
                    print("Status changed!")
                
                with db_session() as session:
                    stmt = (
                        update(ResponseBase)
                        .where(
                            and_(
                                ResponseBase.company_guid == response_base.company_guid,
                                ResponseBase.question_guid
                                == response_base.question_guid,
                                ResponseBase.group_key == group_key,
                            )
                        )
                        .values(
                            success=1,
                            response_guid=resp[0].guid,
                            parsed_response=val,
                            modified_at=now(),
                        )
                    )

                    # Execute the update
                    session.execute(stmt)
                    session.commit()

            except Exception:
                print(traceback.format_exc())
                continue

    except SQLAlchemyError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
