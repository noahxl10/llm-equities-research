from flask import Blueprint, jsonify, request
from sqlalchemy import text
from db.db_session import db_session
from db.data_models import Companies

companies_blueprint = Blueprint('companies_blueprint', __name__)

@companies_blueprint.route("/fetch_companies", methods=["GET"])
def fetch_companies():
    """
    Returns a list of all gp_tickers in the Companies table.
    """
    with db_session() as session:
        tickers = session.query(Companies.gp_ticker).all()
    ticker_list = [result.gp_ticker for result in tickers]
    return {"tickers": ticker_list}