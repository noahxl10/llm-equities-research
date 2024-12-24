from flask import Blueprint, request, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename

from db.db_session import db_session
from db.data_models import Companies
from services.utils import now
import src.engines.prediction_market as pm

portfolio_blueprint = Blueprint('portfolio_blueprint', __name__)

@portfolio_blueprint.route("/request_portfolio", methods=["POST"])
def request_portfolio_responses():
    """
    Example endpoint demonstrating how a portfolio-related upload could be handled.
    """
    params = request.form.to_dict()
    portfolio_exposure_category = params.get("exposureCategory")
    file_type = params.get("fileType")

    if "files" not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    if file_type not in ["company_filings", "company_news", "company_bn"]:
        return jsonify({"error": "Wrong file type"}), 400

    file = request.files.getlist("files")[0]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(f"static/{file_type}", filename)
    file.save(file_path)

    # Example usage: upsert companies from the CSV
    pm.upsert_companies(file_path, "exposure_categories_v1")
    pm.build_exposure_requests(db_session, portfolio_exposure_category)
    return "Success", 200
