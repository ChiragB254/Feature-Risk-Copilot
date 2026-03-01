import json
from openai import OpenAI
from pydantic import ValidationError
from src.models import RiskAnalysis, ComplianceChecklist

class ComplianceChecklistAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
            
    def generate_checklist(self, risk_analysis: RiskAnalysis) -> ComplianceChecklist:
        system_prompt = """You are a senior compliance and launch readiness manager at Wealthsimple.
Your task is to review a Risk Analysis report and generate a final tailored launch checklist, including critical actions, a readiness score, and an approval recommendation.
Return ONLY valid JSON matching the ComplianceChecklist schema. Do not include any markdown formatting like ```json, just the raw JSON object.

Schema requires:
- recommendation: "APPROVE", "REVIEW", or "BLOCK"
- critical_actions: Array of strings
- readiness_score: float (percentage 0 to 100 based on severity of risks)
"""
        risk_dict = risk_analysis.model_dump()
        
        user_prompt = f"""Risk Analysis Report:
{json.dumps(risk_dict, indent=2)}

Please provide the ComplianceChecklist in JSON format."""

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
            return ComplianceChecklist(**parsed_dict)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM output as JSON: {str(e)}\nRaw Response: {raw_json}")
        except ValidationError as e:
            raise ValueError(f"LLM output did not match schema: {str(e)}\nRaw JSON: {raw_json}")
        except Exception as e:
            raise ValueError(f"An error occurred during API call: {str(e)}")
