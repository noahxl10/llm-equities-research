import os

from dba.data_models import (
    Companies,
    EngineBuildStatus,
    EvidenceSynthesisConditions,
    InvestmentThesis,
    PortfolioManagementParameters,
    SynthesisOutput,
)
from dba.db import db_session
from src.wizard import wizard
from src.wizard import model_wizards
import src.config as config
import src.excel_editor as ee
import src.emailer as emailer
from src.AZ import Azure


class Engine:
    def __init__(self, company: Companies):
        self.W = wizard.SuperWizard(model_wizards.OAI())
        self.model = config.models["OpenAI"]
        self.session = db_session
        self.azure = Azure()
        self.company = company
        self.EvidenceSynthesis = ee.EvidenceSynthesis(self.company)

    def run_base_synthesis_questions(self):
        # self.company = company
        self.session().commit()
        self.session().close()

    # UI and backend
    # Build table for tracking thesis stuff
    # build table for tracking base synthesis stuff
    # Download button for thesis report
    # Build button for building new one

    # VIEW Table for tracking thesis stuff
    # VIEW Table for tracking base synthesis stuff

    def build_thesis_report(self, thesis_config: dict):
        #  (a) an investment recommendation (based on our portfolio management parameters and our investment thesis for the company),
        #  (b) a one-line summarized rationale for that recommendation, and then
        #  (c) a more complete synthesis of everything material to our thesis that was disclosed in the earnings report.
        investment_recommendation = thesis_config["investment_recommendation"]
        one_line_summary = thesis_config["one_line_summary"]
        full_synthesis = thesis_config["full_synthesis"]
        pass

    def run(self, recipients=None, build_status_guid=None):
        # thesis = self.session().query(InvestmentThesis).filter(InvestmentThesis.company_guid == self.company.guid).all()
        # portfolio_management_parameters = self.session().query(PortfolioManagementParameters).filter(PortfolioManagementParameters.company_guid == self.company.guid).all()

        # self.session().commit()
        # self.session().close()

        thesis_config = {
            "investment_recommendation": "test",
            "one_line_summary": "test",
            "full_synthesis": "test",
        }

        export_path, blob_name = self.EvidenceSynthesis.build(thesis_config)

        self.azure.upload_blob(export_path, blob_name, overwrite=True)

        if recipients:
            for recipient in recipients:
                emailer.Emailer().send_evidence_synthesis(
                    recipient, self.company.ticker, export_path
                )

        os.remove(export_path)

        build_status = (
            self.session()
            .query(EngineBuildStatus)
            .filter(EngineBuildStatus.guid == build_status_guid)
            .first()
        )
        build_status.status = "completed"
        build_status.blob_path = blob_name
        self.session().commit()

        return blob_name
