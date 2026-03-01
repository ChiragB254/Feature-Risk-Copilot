import json
from openai import OpenAI
from pydantic import ValidationError
from src.models import FeatureSpecification, SimulationMetrics, ValidationReport

class SimulationValidatorAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
            
    def validate(self, spec: FeatureSpecification, metrics: SimulationMetrics) -> ValidationReport:
        system_prompt = """You are a senior data validator and financial analyst at Wealthsimple.
Your task is to review the results of a financial simulation against the original feature specification to identify any anomalies, data issues, or logical inconsistencies.
Return ONLY valid JSON matching the ValidationReport schema. Do not include any markdown formatting like ```json, just the raw JSON object.

Schema requires:
- is_valid: boolean
- anomalies: Array of strings
- notes: string
"""
        spec_dict = spec.model_dump()
        metrics_dict = metrics.model_dump()
        
        user_prompt = f"""Feature Specification:
{json.dumps(spec_dict, indent=2)}

Simulation Metrics:
{json.dumps(metrics_dict, indent=2)}

Please provide the ValidationReport in JSON format."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            raw_json = response.choices[0].message.content.strip()
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:-3].strip()
            elif raw_json.startswith("```"):
                raw_json = raw_json[3:-3].strip()
                
            parsed_dict = json.loads(raw_json)
            return ValidationReport(**parsed_dict)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM output as JSON: {str(e)}\nRaw Response: {raw_json}")
        except ValidationError as e:
            raise ValueError(f"LLM output did not match schema: {str(e)}\nRaw JSON: {raw_json}")
        except Exception as e:
            raise ValueError(f"An error occurred during API call: {str(e)}")
