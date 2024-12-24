from flask import Blueprint, request
from flask_executor import Executor
from db.db_session import db_session
from db.data_models import Companies
from services.build_service import build_file
from engines import financial_model, one_pager

engine_builds_blueprint = Blueprint('engine_builds_blueprint', __name__)

# You may want to initialize Executor here if not in create_app or pass it in some other way.
executor = Executor()

@engine_builds_blueprint.route("/build_financial_model", methods=["POST"])
def build_financial_model_api():
    """
    Triggers the build of a financial model for a given ticker.
    """
    ticker = request.json.get("ticker")
    with db_session() as session:
        company_guid = Companies.get_company_guid(session, ticker)

    engine_instance = financial_model.Engine(Companies(guid=company_guid, ticker=ticker))
    build_file(company_guid, "financial_model", engine_instance, executor)
    return "Success", 200

@engine_builds_blueprint.route("/build_one_pager", methods=["POST"])
def build_one_pager_api():
    """
    Triggers the build of a one-pager for a given ticker.
    """
    ticker = request.json.get("ticker")
    with db_session() as session:
        company_guid = Companies.get_company_guid(session, ticker)

    engine_instance = one_pager.Engine(Companies(guid=company_guid, ticker=ticker))
    build_file(company_guid, "one_pager", engine_instance, executor)
    return "Success", 200