"""
Author: Noah Alex
Contact: noahcalex@gmail.com
Year: 2024
Company: Grandeur Peak Global Advisors
"""

from app import create_app
import os

app = create_app()

if __name__ == "__main__":
    if os.getenv("FLASK_ENV") == "PROD":
        app.run(debug=True, host="0.0.0.0")
    else:
        app.run(debug=True, host="0.0.0.0", port=8000)
