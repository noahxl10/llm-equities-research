import re
import traceback
from string import ascii_letters

# Translation table for removing letters A-Z/a-z.
translation_table = str.maketrans("", "", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


class Parser:
    """
    A utility class for parsing numerical amounts, time periods, and
    segment data from strings. Handles transformations like removing
    unwanted characters, converting units, and validating formats.
    """

    def __init__(self):
        pass

    def parse_amount(self, string_amount_value, unit):
        """
        Cleans and converts a string that may contain numerical values, currency symbols, and
        textual references (like 'million', 'billion', 'thousand', 'percent') into a float.

        :param string_amount_value: The raw string containing an amount.
        :param unit: A string indicating the unit or scale (e.g. 'million', 'billion', 'percent').
        :return: A float representing the amount parsed from the string.
        """
        try:
            # Remove commas, parentheses, currency symbols, etc.
            clean_value = (
                string_amount_value.replace(",", "")
                .replace("$", "")
                .replace("(", "")
                .replace(")", "")
                .replace("\\", "")
                .replace(" ", "")
                .rstrip(ascii_letters)
            )
            # Convert textual signals like "decrease" or "increase"
            if "decrease" in clean_value:
                clean_value = clean_value.replace("decrease", "-")
            elif "increase" in clean_value:
                clean_value = clean_value.replace("increase", "")

            # Remove all letters A-Z using the translation table
            clean_value = clean_value.translate(translation_table)

        except Exception:
            # If there's an error in cleaning, revert to the original
            clean_value = string_amount_value

        try:
            unit_lower = unit.lower()
            if "million" in unit_lower:
                int_amount = float(clean_value) * 1_000_000
            elif "thousand" in unit_lower:
                int_amount = float(clean_value) * 1_000
            elif "billion" in unit_lower:
                int_amount = float(clean_value) * 1_000_000_000
            elif "percent" in unit_lower or "%" in unit_lower:
                clean_value = clean_value.replace("%", "")
                int_amount = float(clean_value) / 100
            elif "dollar" in unit_lower:
                int_amount = float(clean_value)
            else:
                int_amount = 0.0
        except Exception as e:
            print(e)
            int_amount = 0.0

        return int_amount

    def is_valid_format(self, input_string):
        """
        Checks if a string is either a four-digit year (e.g. 2023)
        or a quarter string (e.g. q1mar23, q4dec25).

        :param input_string: The time-period or year string to validate.
        :return: True if the format matches either 'YYYY' or 'qXmmmYY'; False otherwise.
        """
        year_pattern = re.compile(r"^\d{4}$")
        quarter_pattern = re.compile(r"^q[1-4](sep|dec|jun|mar)\d{2}$")

        if year_pattern.match(input_string) or quarter_pattern.match(input_string):
            return True
        return False

    def parse_time_period(self, time_period):
        """
        Attempts to parse a time_period string into a normalized format.
        If it looks like a quarter (e.g. includes "q1", "q2", etc.), transform
        it into something like 'q1mar23'. If it looks like a year, keep it as is.
        Returns the transformed string or the original one if it can't be validated.

        :param time_period: The raw time_period string.
        :return: A validated/normalized time period string or the original if unable to parse.
        """
        original_time_period = time_period

        month_mapping = {
            1: "jan", 2: "feb", 3: "mar", 4: "apr",  5: "may",  6: "jun",
            7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"
        }
        quarter_start_month = {
            1: 3,   # Q1 starts in January
            2: 6,   # Q2 starts in April
            3: 9,   # Q3 starts in July
            4: 12,  # Q4 starts in October
        }
        search_dict = {
            1: [1, "first"],   # Q1
            2: [2, "second"],  # Q2
            3: [3, "third"],   # Q3
            4: [4, "fourth"],  # Q4
        }

        # Remove stray backslashes
        time_period = time_period.replace("\\", "")

        # Try to extract the year
        try:
            pattern = re.compile(r"\b\d{4}\b")
            matches = pattern.findall(time_period)
            year = matches[0].replace(" ", "")[2:]
        except Exception:
            year = time_period[-2:]

        # If we detect a 'q' in the string, guess it's a quarter format
        if "q" in time_period.lower():
            year = time_period[-2:]
            for i in range(1, 5):
                if any(str(ele) in time_period.lower()[0:2] for ele in search_dict[i]):
                    month = month_mapping[quarter_start_month[i]]
                    time_period = f"q{i}{month}{year}"
                    continue

        elif "f" in time_period.lower() or "year" in time_period.lower():
            # Basic cleaning in case of strings like 'FY2023' or 'f2023'
            time_period = (
                time_period.lower()
                .rstrip(ascii_letters)
                .lstrip(ascii_letters)
                .replace(" ", "")
            )

        # Validate final format; print debug if invalid
        if not self.is_valid_format(time_period):
            print("Original time period input:", original_time_period)
            print("parse attempt:", time_period)

        return time_period

    def parse_segments_2(self):
        """
        Reads a 'segments.txt' file, looks for bracketed data lines ([...]),
        and extracts segmented revenue & growth data keyed by year.

        :return: A dictionary keyed by segment, with sub-dicts keyed by year for
                 segment values like 'pct_of_revenue' and 'growth_rate'.
        """
        with open("static/engine/financial_model/segments.txt") as f:
            file_contents = f.read()

        matches = re.findall(r"\[[^\]]*\]", file_contents)
        segment_dict = {}
        for match in matches:
            entry = match.replace("[", "").replace("]", "")
            options = [
                option.replace("['", "").replace("']", "").replace("'", "")
                for option in entry.split(", ")
            ]
            year = self.parse_time_period(options[3])
            if options[0] not in segment_dict:
                segment_dict[options[0]] = {}
            segment_dict[options[0]][year] = {
                "pct_of_revenue": options[1].rstrip(ascii_letters),
                "growth_rate": options[2].rstrip(ascii_letters),
            }
        return segment_dict

    def test_parse(self, response):
        """
        A test parser that extracts bracketed values from a string, attempts
        to parse them into time periods and amounts, and prints debug info.

        :param response: A string potentially containing bracketed data.
        :return: A list of dictionaries with 'time_period' and parsed 'amount', etc.
        """
        data_list = []
        matches = re.findall(r"\[[^\]]*\]", response)

        for match in matches:
            try:
                # Filter out any '†' lines
                if "†" in match:
                    continue

                og_match = match
                cleaned = (
                    match.replace("['", "")
                    .replace("']", "")
                    .replace("[", "")
                    .replace("]", "")
                    .split(", ")
                )
                # Example: ['1000', 'USD', 'q1mar23']
                cleaned[0] = cleaned[0].replace(",", "")

                try:
                    tp = self.parse_time_period(cleaned[2])
                except Exception:
                    tp = self.parse_time_period(cleaned[1])

                if tp is not None:
                    parsed_amount = self.parse_amount(cleaned[0], cleaned[1])
                    print(tp)
                    print(parsed_amount)
                    data_list.append({})  # You could store the parsed data if needed

            except Exception:
                print(traceback.format_exc())
                print(og_match)

        return data_list

    def just_parse(self, response, column, format_type, measure_type, is_yoy, measure_prompt, number_format_type):
        """
        Parses bracketed segments from a response, attempting to interpret each segment
        as an amount, unit, and time period. Returns a list of parsed records.

        :param response: The raw string from which bracketed data is extracted.
        :param column: A column reference (e.g., 'AA' in an Excel sheet).
        :param format_type: The format (e.g., 'currency', 'percent') for the parsed amount.
        :param measure_type: Not used within the method, but included for function signature consistency.
        :param is_yoy: Indicates if the measure is year-over-year data.
        :param measure_prompt: Identifies the measure or prompt that led to this data extraction.
        :param number_format_type: The type of number formatting to apply (e.g., 'percent', 'currency').
        :return: A list of dicts containing the parsed segment data.
        """
        parsed_results = []
        matches = re.findall(r"\[[^\]]*\]", response)

        for match in matches:
            try:
                if "†" in match:
                    continue

                og_match = match
                cleaned = (
                    match.replace("['", "")
                    .replace("']", "")
                    .replace("[", "")
                    .replace("]", "")
                    .split(", ")
                )
                cleaned[0] = cleaned[0].replace(",", "")

                try:
                    tp = self.parse_time_period(cleaned[2])
                except Exception:
                    tp = self.parse_time_period(cleaned[1])

                if tp is not None:
                    parsed_dict = {
                        "core_measure_prompt": measure_prompt,
                        "time_period": str(tp.replace("\\", "")).replace("year", ""),
                        "amount": self.parse_amount(cleaned[0], cleaned[1]),
                        "column": column,
                        "format_type": format_type,
                        "is_yoy": is_yoy,
                        "number_format_type": number_format_type,
                    }
                    parsed_results.append(parsed_dict)

            except Exception:
                print(traceback.format_exc())
                print(og_match)

        return parsed_results

    def parse_responses(self, responses=None):
        """
        For a list of response strings, extracts bracketed segments [a, b, c, d],
        interprets them, and compiles them into a list of dicts with measure/time/amount/unit.

        :param responses: A list of strings to parse.
        :return: A list of dictionaries, each containing 'measure', 'time_period', 'amount', 'unit'.
        """
        if not responses:
            return []

        result_data = []
        for resp in responses:
            matches = re.findall(r"\[[^\]]*\]", resp)
            for match in matches:
                options = [
                    opt.replace("['", "").replace("']", "").replace("'", "")
                    for opt in match.split("', ")
                ]
                try:
                    parsed_item = {
                        "measure": options[0],
                        "time_period": str(self.parse_time_period(options[1])),
                        "amount": self.parse_amount(options[2], options[3]),
                        "unit": options[3],
                    }
                    result_data.append(parsed_item)
                except Exception:
                    print(traceback.format_exc())

        return result_data

    def parse_segments(self, response=None):
        """
        Similar to parse_responses but specialized for segments. Expects bracketed data
        with amount/unit/time data, e.g. [1000, USD, q1mar23]. Returns each as a dict.

        :param response: A string possibly containing bracketed segment definitions.
        :return: A list of dicts with 'amount', 'time_period', 'unit'.
        """
        if not response:
            return []

        result_data = []
        matches = re.findall(r"\[[^\]]*\]", response)

        for match in matches:
            cleaned = (
                match.replace("['", "")
                .replace("']", "")
                .replace("[", "")
                .replace("]", "")
                .split(", ")
            )
            cleaned[0] = cleaned[0].replace(",", "")

            try:
                segment_dict = {
                    "amount": self.parse_amount(cleaned[0], cleaned[1]),
                    "time_period": self.parse_time_period(cleaned[2]).split(" vs ")[0],
                    "unit": cleaned[1],
                }
                result_data.append(segment_dict)
            except Exception:
                print(traceback.format_exc())
                pass

        return result_data
