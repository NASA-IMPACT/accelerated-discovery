#!/usr/bin/env python3
"""
Convert Excel file with multiple sheets to JSON format.
Each sheet name becomes a key in the JSON structure.
"""

import json
import sys

import pandas as pd


def convert_excel_to_json(excel_file_path: str, output_file_path: str) -> None:
    """
    Convert Excel file with multiple sheets to JSON format.

    Args:
        excel_file_path: Path to the Excel file
        output_file_path: Path where JSON file will be saved
    """
    try:
        # Read all sheets from Excel file
        excel_file = pd.ExcelFile(excel_file_path)

        # Dictionary to store all sheets data
        all_sheets_data = {}

        # Process each sheet
        for sheet_name in excel_file.sheet_names:
            print(f"Processing sheet: {sheet_name}")

            # Read the sheet
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

            # Clean column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()

            # Convert DataFrame to list of dictionaries
            # Replace NaN values with None for better JSON representation
            sheet_data = df.where(pd.notnull(df), None).to_dict("records")

            # Add to main dictionary with sheet name as key
            all_sheets_data[sheet_name] = {
                "sheet_info": {
                    "name": sheet_name,
                    "total_journals": len(sheet_data),
                    "columns": list(df.columns),
                },
                "journals": sheet_data,
            }

        # Add metadata
        result = {
            "metadata": {
                "source_file": excel_file_path,
                "total_sheets": len(excel_file.sheet_names),
                "sheet_names": excel_file.sheet_names,
                "expected_columns": [
                    "Journal Name",
                    "Publisher / Society",
                    "Primary Focus Area",
                    "Open Access Status",
                    "Link to Journal Homepage",
                ],
            },
            "data": all_sheets_data,
        }

        # Write to JSON file
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\nConversion completed successfully!")
        print(f"JSON file saved to: {output_file_path}")
        print(f"Total sheets processed: {len(excel_file.sheet_names)}")

        # Print summary
        print("\nSummary:")
        for sheet_name, sheet_data in all_sheets_data.items():
            print(
                f"  - {sheet_name}: {sheet_data['sheet_info']['total_journals']} journals"
            )

    except Exception as e:
        print(f"Error processing Excel file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # File paths
    excel_path = "/Users/mramasub/work/accelerated-discovery/docs/Pubs-Whitelist.xlsx"
    json_path = "/Users/mramasub/work/accelerated-discovery/docs/pubs_whitelist.json"

    # Convert the file
    convert_excel_to_json(excel_path, json_path)
