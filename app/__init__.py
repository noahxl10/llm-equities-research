mport os
import datetime
import sys
import traceback
import io

from flask import Flask, send_file, jsonify, make_response, request
from flask_cors import CORS
from flask_executor import Executor
from sqlalchemy import text, and_, distinct
from dotenv import load_dotenv

# Load environment variables if not set
if os.getenv("ENV") is None:
    load_dotenv()

# Local imports
from config.config import Config
from db.db_session import db_session
from db.data_models import EngineBuildStatus, Companies
from services.azure_service import Azure
from services.wizard_service import WizardService
from api.auth import auth_blueprint
from api.companies import companies_blueprint
from api.engine_builds import engine_builds_blueprint
from api.file_uploads import file_uploads_blueprint
from api.financial_model import financial_model_blueprint
from api.one_pager import one_pager_blueprint
from api.portfolio import portfolio_blueprint
from api.question_repository import question_repository_blueprint

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    executor = Executor(app)
    azure = Azure()
    wizard_service = WizardService()

    app.config["SECRET_KEY"] = "your_strong_secret_key"
    app.config["JWT_SECRET_KEY"] = "your_jwt_secret_key"
    app.config["JWT_TOKEN_LOCATION"] = ["headers"]

    # Register blueprints for modular routes
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    app.register_blueprint(companies_blueprint, url_prefix='/companies')
    app.register_blueprint(engine_builds_blueprint, url_prefix='/engine')
    app.register_blueprint(file_uploads_blueprint, url_prefix='/files')
    app.register_blueprint(financial_model_blueprint, url_prefix='/financial_model')
    app.register_blueprint(one_pager_blueprint, url_prefix='/one_pager')
    app.register_blueprint(portfolio_blueprint, url_prefix='/portfolio')
    app.register_blueprint(question_repository_blueprint, url_prefix='/questions')

    @app.route("/fetch_companies_built", methods=["GET"])
    def fetch_companies_built():
        """
        Example endpoint kept in app.py for demonstration 
        (but could be moved to companies.py or engine_builds.py).
        """
        with db_session() as session:
            query = text("""
                SELECT DISTINCT gp_ticker
                FROM llm.engine_build_status e
                LEFT JOIN llm.companies c ON c.guid = e.company_guid
                WHERE status = 'completed'
                  AND engine = 'one_pager'
            """)
            results = session.execute(query)
        tickers = [result[0] for result in results]
        return {"tickers": tickers}

    @app.route("/download_engine_file", methods=["POST"])
    def download_engine_file():
        """
        Downloads the latest file from the build status table for a given ticker and engine type.
        """
        ticker = request.json.get("ticker")
        file_type = request.json.get("fileType")

        with db_session() as session:
            company_guid = Companies.get_company_guid(session, ticker)

            build_record = (
                session.query(EngineBuildStatus)
                .filter(
                    EngineBuildStatus.company_guid == company_guid,
                    EngineBuildStatus.engine == file_type,
                    EngineBuildStatus.status == "completed",
                )
                .order_by(EngineBuildStatus.created_at.desc())
                .first()
            )

        if not build_record:
            return jsonify({"error": "No completed build found."}), 404

        try:
            blob_data = azure._get_blob(blob_name=build_record.blob_path)
            stream = io.BytesIO()
            blob_data.readinto(stream)
            stream.seek(0)
            return send_file(
                stream,
                as_attachment=True,
                download_name=build_record.blob_path.split("/")[-1],
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/teardown", methods=["GET"])
    def test_teardown():
        """
        A test endpoint to demonstrate that teardown_appcontext is invoked automatically.
        """
        return jsonify({"message": "Teardown test complete."})

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """
        Automatically remove database sessions at the end of each request
        or when the application shuts down.
        """
        db_session.remove()

    return app

if __name__ == "__main__":
    app = create_app()
    # Remove or lower debug mode in production
    app.run(host="0.0.0.0", port=5000, debug=True)