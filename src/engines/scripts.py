import os
import asyncio
import traceback
import pandas as pd
from dba.db import db_session
from concurrent.futures import ThreadPoolExecutor
from src.AZ import Azure
import src.config as config
from ..wizard import wizard as wizard, model_wizards as model_wizards
from dba.data_models import (
    Companies,
    AsisstantBlobs,
    OaiAssistants,
)


def get_company_guid(session, ticker):
    company_guid = (
        session.query(Companies)
        .filter(Companies.ticker == ticker or Companies.gp_ticker == ticker)
        .first()
    )
    return company_guid.guid


def upload_companies(company_file):
    session = db_session

    def ticker_exists(ticker):
        return session.query(Companies).filter_by(gp_ticker=ticker).first() is not None

    df = pd.read_csv(company_file)
    for index, row in df.iterrows():
        company = Companies(
            gp_ticker=row["gp_ticker"].upper(),
            name=row["name"],
            strategies="financial_model_v1,one_pager_v1",
        )
        if ticker_exists(company.gp_ticker):
            pass
        else:
            session().add(company)
            session().commit()


def upload_financial_documents(parent_folder):
    session = db_session()
    folders = [f.name for f in os.scandir(parent_folder) if f.is_dir()]
    azure = Azure()
    W = model_wizards.OAI()

    file_type = "company_filings"

    for folder in folders:
        ticker = folder.split("(")[1].replace(")", "")
        try:
            company_guid = get_company_guid(session, ticker)
            blobs = (
                db_session()
                .query(AsisstantBlobs)
                .filter(AsisstantBlobs.company_guid == company_guid)
                .all()
            )
            if len(blobs) == 0 or blobs is None:
                print("no blobs found: ", blobs)

                subfolder_path = os.path.join(parent_folder, folder)
                files = [f.name for f in os.scandir(subfolder_path) if f.is_file()]
                model = config.models["OpenAI"]
                W.get_assistant(company_guid, model.qualified_api_name)
                assistant_id = (
                    db_session()
                    .query(OaiAssistants)
                    .filter(OaiAssistants.company_guid == company_guid)
                    .first()
                )

                for file in files:
                    try:
                        file_name = os.path.join(parent_folder, folder, file)
                        blob_name = f"uploads/{company_guid}/{file_type}/{file}"

                        try:
                            azure.upload_blob(
                                file_name=file_name, blob_name=blob_name, overwrite=True
                            )
                            is_uploaded = 1
                        except Exception:
                            is_uploaded = 0

                        W.update_vector_store_file(company_guid, blob_name, file_name)

                        assistant_blob = AsisstantBlobs(
                            company_guid=company_guid,
                            assistant_id=assistant_id.assistant_id,
                            file_name=file,
                            blob_path=blob_name,
                            is_uploaded=is_uploaded,
                        )
                        session.add(assistant_blob)
                        session.commit()
                    except Exception:
                        print(traceback.format_exc())

        except Exception:
            print(traceback.format_exc())


async def process_folders(folders, session, parent_folder, azure, W, config):
    file_type = "company_filings"
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        for folder in folders:
            ticker = folder.split("(")[1].replace(")", "")

            try:
                company_guid = await loop.run_in_executor(
                    executor, get_company_guid, session, ticker
                )
                blobs = await loop.run_in_executor(
                    executor,
                    session.query(AsisstantBlobs)
                    .filter(AsisstantBlobs.company_guid == company_guid)
                    .all,
                )

                if not blobs:  # Better way to check if blobs is empty or None
                    subfolder_path = os.path.join(parent_folder, folder)
                    files = [f.name for f in os.scandir(subfolder_path) if f.is_file()]
                    model = config.models["OpenAI"]
                    W.get_assistant(company_guid, model.qualified_api_name)
                    assistant_id = await loop.run_in_executor(
                        executor,
                        session.query(OaiAssistants)
                        .filter(OaiAssistants.company_guid == company_guid)
                        .first,
                    )

                    for file in files:
                        try:
                            file_name = os.path.join(parent_folder, folder, file)

                            blob_name = f"uploads/{company_guid}/{file_type}/{file}"

                            print(blob_name)
                            try:
                                await loop.run_in_executor(
                                    executor,
                                    azure.upload_blob,
                                    file_name,
                                    blob_name,
                                    True,
                                )
                                is_uploaded = 1
                            except Exception:
                                is_uploaded = 0

                            await loop.run_in_executor(
                                executor,
                                W.update_vector_store_file,
                                company_guid,
                                blob_name,
                                file_name,
                            )

                            assistant_blob = AsisstantBlobs(
                                company_guid=company_guid,
                                assistant_id=assistant_id.assistant_id,
                                file_name=file,
                                blob_path=blob_name,
                                is_uploaded=is_uploaded,
                            )

                            await loop.run_in_executor(
                                executor, session.add, assistant_blob
                            )
                            await loop.run_in_executor(executor, session.commit)

                        except Exception:
                            print(traceback.format_exc())
            except Exception:
                print(traceback.format_exc())


def run_process_folders():
    session = db_session()
    parent_folder = ""
    folders = [f.name for f in os.scandir(parent_folder) if f.is_dir()]
    azure = Azure()
    W = model_wizards.OAI()
    asyncio.run(process_folders(folders, session, parent_folder, azure, W, config))
