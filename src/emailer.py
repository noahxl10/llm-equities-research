import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
from dotenv import load_dotenv

if os.getenv("ENV") is None:
    load_dotenv()


class Emailer:
    def __init__(self):
        self.server_name = os.environ.get("BREVO_SMTP_SERVER")
        self.port = os.environ.get("BREVO_SMTP_PORT")
        self.username = os.environ.get("BREVO_SMTP_USERNAME")
        self.password = os.environ.get("BREVO_SMTP_PASSWORD")
        self.sender_email = os.environ.get("BREVO_SENDER_EMAIL")

    def send_financial_model(self, recipient_email, ticker, model_path):
        subject = f"Your Financial Model for {ticker} is ready"
        body = "Here is a file: "
        self.send_email(recipient_email, subject, body, [model_path])

    def send_one_pager(self, recipient_email, ticker, model_path):
        subject = f"Your One Pager for {ticker} is ready"
        body = "Here is the file: "
        self.send_email(recipient_email, subject, body, [model_path])

    def send_evidence_synthesis(self, recipient_email, ticker, model_path):
        subject = f"Your Evidence Synthesis for {ticker} is ready"
        body = "Here is the file: "
        self.send_email(recipient_email, subject, body, [model_path])

    def send_email(self, recipient_email, subject, body, files=[]):
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        for f in files or []:
            with open(f, "rb") as fil:
                part = MIMEApplication(fil.read(), Name=os.path.basename(f))
            part["Content-Disposition"] = (
                'attachment; filename="%s"' % os.path.basename(f)
            )
            msg.attach(part)

        try:
            server = smtplib.SMTP(self.server_name, self.port)
            server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
            server.login(self.username, self.password)

            server.sendmail(self.sender_email, recipient_email, msg.as_string())

        except Exception as e:
            print(f"Failed to send email: {e}")

        finally:
            # Close the connection to the SMTP server
            server.quit()


def test_send_email():
    mailer = Emailer()
    mailer.send_financial_model(
        "noahcalex@gmail.com",
        "BURL",
        "static/engine/financial_model/llm_financial_model_base_altered_BURL.xlsm",
    )
