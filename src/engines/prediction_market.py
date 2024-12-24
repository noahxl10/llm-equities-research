import traceback
import uuid

import pandas as pd
from sqlalchemy import and_, or_, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert

import src.config as config
import src.engines.engine_utils as utils
from dba.utils import (
    upsert_companies,
    upsert_portfolio,
    build_requests,
    insert_portfolio
)
from dba.data_models import (
    Companies,
    ExposureCategories,
    ExposureCategoriesCompanies,
    PredictionMarket,
    PredictionMarketOrder,
    # QuestionRepositoryResearch,
    Portfolio,
    CompanyPortfolio,
    ResponseBase,
    WizardRequest,
)
from dba.db import db_session
from src.utils import now

from ..wizard import model_wizards as model_wizards
from ..wizard import wizard as wizard


# Will turn this into a class eventually... Some methods need to be moved out of this object
# and into the more obtuse dba object


def get_current_price(session, prediction_market_guid):
    BASE_PRICE = 0.5

    market_results = (
        session.query(PredictionMarketOrder)
        .filter(PredictionMarketOrder.prediction_market_guid == prediction_market_guid)
        .all()
    )

    buy_units = 0
    sell_units = 0

    for result in market_results:
        order = result[0]
        if order["type"] == "buy":
            buy_units += order["units"]
        elif order["type"] == "sell":
            sell_units += order["units"]

    current_market_price = (buy_units * BASE_PRICE + sell_units * BASE_PRICE) / (
        BASE_PRICE * (buy_units + sell_units)
    )

    return current_market_price


def create_prediction_market(session, exposure_category_guid, question):
    session.add(
        PredictionMarket(
            exposure_category_guid=exposure_category_guid,
            name=question,
        )
    )
    session.commit()
    return True


def place_order(session, user_guid, prediction_market_guid, order: dict):
    # user_guid = get_user_guid_from_email(user_email)
    session.add(
        PredictionMarketOrder(
            prediction_market_guid=prediction_market_guid,
            user_guid=user_guid,
            order=order,
        )
    )
    session.commit()
    return True




def send_exposure_requests():
    W = wizard.SuperWizard(model_wizards.OAI())
    model = config.models["OpenAI"]
    try:
        with db_session() as session:
            response_base_rows = (
                session.query(ResponseBase, Companies, ExposureCategories)
                .join(Companies, ResponseBase.company_guid == Companies.guid)
                .join(
                    ExposureCategories,
                    ResponseBase.question_guid == ExposureCategories.guid,
                )
                .filter(
                    and_(
                        or_(ResponseBase.success == 0, ResponseBase.success.is_(None)),
                        ResponseBase.engine == "exposure_categories_v1",
                    )
                )
                .order_by(ResponseBase.company_guid.desc())
                .all()
            )

            for (
                response_base,
                companies,
                exposure_categories,
            ) in response_base_rows:
                try:
                    prompt = f"Answer the following question with a 'yes' or 'no' first: Is {companies.gp_ticker} materially exposed to {exposure_categories.name}?"

                    resp = W.send_requests(
                        [
                            WizardRequest(
                                model_guid=model.guid,
                                model_qualified_api_name=model.qualified_api_name,
                                company_guid=response_base.company_guid,
                                request=prompt,
                                request_type="assistant",
                            )
                        ]
                    )
                    split_response = resp[0].response
                    val = utils.get_y_n(split_response)

                    # Insert company exposure category
                    if val == 1:
                        ecc = ExposureCategoriesCompanies(
                            company_guid=companies.guid,
                            exposure_category_guid=exposure_categories.guid,
                            is_active=1,
                        )
                        db_session().add(ecc)
                        db_session().commit()

                    stmt = (
                        update(ResponseBase)
                        .where(
                            and_(
                                ResponseBase.company_guid == response_base.company_guid,
                                ResponseBase.question_guid
                                == response_base.question_guid,
                                ResponseBase.engine == "exposure_categories_v1",
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
