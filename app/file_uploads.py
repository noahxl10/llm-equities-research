import os
import io
import traceback
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from db.db_session import db_session
from db.data_models import Companies, OaiAssistants, AsisstantBlobs
from services.azure_service import Azure
from services.wizard_service import WizardService

file_uploads_blueprint = Blueprint('file_uploads_blueprint', __name__)
azure = Azure()
wizard_service = WizardService()

@file_uploads_blueprint.route("/upload", methods=["POST"])
def upload_files():
    """
    Uploads one or more files for a specific company ticker, storing them in Azure
    and updating the OaiAssistants / AsisstantBlobs records.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    if "ticker" not in request.form:
        return jsonify({"error": "No ticker part in the request"}), 400

    params = request.form.to_dict()
    ticker = params.get("ticker")
    file_type = params.get("fileType")

    if file_type not in ["company_filings", "company_news", "company_bn"]:
        return jsonify({"error": "Wrong file type or no file type"}), 400

    with db_session() as session:
        company_guid = Companies.get_company_guid(session, ticker)

    files = request.files.getlist("files")
    file_names = []

    for file in files:
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(f"static/{file_type}", filename)
        file.save(file_path)
        blob_name = f"uploads/{company_guid}/{file_type}/{filename}"

        try:
            azure.upload_blob(file_name=file_path, blob_name=blob_name, overwrite=True)
            is_uploaded = 1
        except:
            traceback.print_exc()
            is_uploaded = 0

        with db_session() as session:
            assistant_entry = (
                session.query(OaiAssistants)
                .filter(OaiAssistants.company_guid == company_guid)
                .first()
            )
            if assistant_entry:
                assistant_id = assistant_entry.assistant_id
            else:
                assistant_id = None

        with db_session() as session:
            assistant_blob = AsisstantBlobs(
                company_guid=company_guid,
                assistant_id=assistant_id,
                file_name=filename,
                blob_path=blob_name,
                is_uploaded=is_uploaded,
            )
            session.add(assistant_blob)
            session.commit()

        if assistant_id:
            wizard_service.update_vector_store_file(company_guid, blob_name, file_path)
        else:
            wizard_service.get_assistant(company_guid, blob_name, file_path)

        # Clean up local file
        os.remove(file_path)
        file_names.append(filename)

    return jsonify({"fileNames": file_names}), 200

@file_uploads_blueprint.route("/download_document", methods=["POST"])
def download_document():
    """
    Downloads a file from Azure based on ticker and fileType.
    """
    api_key = request.headers.get("Authorization")
    # This is a placeholder check; adapt for your actual auth logic
    if api_key != "Bearer <YourRealKeyHere>":
        return "Unauthorized", 401

    ticker = request.json.get("ticker")
    file_type = request.json.get("fileType")

    with db_session() as session:
        company_guid = Companies.get_company_guid(session, ticker)

    if file_type == "evidence_synthesis":
        path = f"uploads/engines/{company_guid}/{file_type}/llm_{file_type}_base_altered.xlsm"
    else:
        path = f"uploads/engines/{company_guid}/{file_type}/llm_{file_type}_base.xlsm"

    try:
        blob_data = azure._get_blob(blob_name=path)
        stream = io.BytesIO()
        blob_data.readinto(stream)
        stream.seek(0)
        return send_file(stream, as_attachment=True, download_name=path.split("/")[-1])
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Server error: {str(e)}"}, 500
