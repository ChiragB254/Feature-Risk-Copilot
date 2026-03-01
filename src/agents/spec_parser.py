import json
from openai import OpenAI
from pydantic import ValidationError
from src.models import FeatureSpecification

class SpecParserAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
            
    def parse(self, description: str) -> FeatureSpecification:
        system_prompt = """You are a senior fintech analyst at Wealthsimple. Your task is to extract spec details from a product feature description and return ONLY valid JSON matching the exact schema provided. Do not include any markdown formatting like ```json, just the raw JSON object.

CRITICAL INSTRUCTION: You MUST extract the EXACT numerical values and conditions provided in the user's input. Do NOT copy the examples provided below.

Fields to extract:
- 'feature_name': A short, descriptive name for the feature.
- 'fee_type': Either 'fixed' or 'percentage'.
- 'fee_value': The exact numerical value of the fee extracted from the user text (e.g., if user says "1.5%", output 1.5).
- 'applies_to': Description of what the fee applies to (e.g., 'All withdrawals', 'crypto withdrawals').
- 'condition': A Pandas dataframe query string to filter impacted customers using these columns: account_balance, monthly_withdrawals, crypto_exposure, fx_transactions_mo, income_band, is_active_trader, account_age_months. (e.g., if user says "under $2000", output 'account_balance < 2000'). If no condition is specified, use 'index == index' to match all rows.
- 'assumptions': An array of strings with assumptions you made while parsing.

Ensure the condition string is valid pandas syntax because it will be passed directly to df.query().
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this feature description into JSON:\n\n{description}"}
                ]
            )
            
            raw_json = response.choices[0].message.content.strip()
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:-3].strip()
            elif raw_json.startswith("```"):
                raw_json = raw_json[3:-3].strip()
                
            parsed_dict = json.loads(raw_json)
            return FeatureSpecification(**parsed_dict)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM output as JSON: {str(e)}\nRaw Response: {raw_json}")
        except ValidationError as e:
            raise ValueError(f"LLM output did not match schema: {str(e)}\nRaw JSON: {raw_json}")
        except Exception as e:
            raise ValueError(f"An error occurred during API call: {str(e)}")
