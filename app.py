import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from src.agents.spec_parser import SpecParserAgent
from src.simulator import FinancialSimulator
from src.agents.simulation_validator import SimulationValidatorAgent
from src.agents.risk_analyzer import RiskAnalysisAgent
from src.agents.compliance_checker import ComplianceChecklistAgent

load_dotenv()

st.set_page_config(page_title="Wealthsimple Feature Risk Copilot", layout="wide")

# Global CSS to scale down fonts for better density and readability
st.markdown("""
<style>
    /* Scale down main headers */
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.5rem !important; }
    h4 { font-size: 1.2rem !important; }
    h5 { font-size: 1.1rem !important; }
    
    /* Scale down standard text, metrics, and markdown */
    .stMarkdown, .stText, p, li { font-size: 0.95rem !important; }
    
    /* Scale down metric values (numbers) and labels */
    div[data-testid="stMetricValue"] > div { 
        font-size: 1.6rem !important; 
        text-overflow: unset !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    div[data-testid="stMetricLabel"] > label { 
        font-size: 0.85rem !important; 
        text-overflow: unset !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    
    /* Scale down expander titles and form labels */
    .streamlit-expanderHeader { font-size: 0.9rem !important; }
    .stTextArea label { font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Wealthsimple Feature Risk Copilot")
st.markdown("Analyze financial impact and fairness risks for new product features instantly.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🛠️ Configuration")
provider = st.sidebar.radio("Select AI Provider", ["Groq", "Ollama"])

api_client = None
model_name = None

if provider == "Groq":
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    model_name = st.sidebar.selectbox("Model", [
        "llama-3.3-70b-versatile", 
        "llama-3.1-8b-instant", 
        "mixtral-8x7b-32768", 
        "gemma2-9b-it"
    ])
    if not groq_api_key:
        st.sidebar.warning("Please enter your Groq API Key.")
    else:
        api_client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
else:
    model_name = st.sidebar.text_input("Ollama Model Name", value="ministral-3:3b")
    # Ollama uses a dummy API key due to OpenAI client requirements
    api_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Initialize components
@st.cache_resource
def get_simulator():
    return FinancialSimulator()

try:
    simulator = get_simulator()
except Exception as e:
    st.error(f"Failed to initialize simulator. Ensure data/customers.csv exists. Error: {str(e)}")
    st.stop()

if api_client is None:
    st.warning(f"Please configure the {provider} settings in the sidebar to proceed.")
    st.stop()

# Initialize agents with user-provided configurations
parser_agent = SpecParserAgent(client=api_client, model=model_name)
validator_agent = SimulationValidatorAgent(client=api_client, model=model_name)
risk_agent = RiskAnalysisAgent(client=api_client, model=model_name)
compliance_agent = ComplianceChecklistAgent(client=api_client, model=model_name)

if "user_feature_input" not in st.session_state:
    st.session_state.user_feature_input = "Introduce a 0.5% FX fee on crypto withdrawals under $1,000."

# --- SIDEBAR: FEATURE INPUT ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Feature Description")
with st.sidebar.form("feature_input_form"):
    feature_input = st.text_area(
        "Describe the product feature you want to evaluate:",
        value=st.session_state.user_feature_input,
        height=150
    )
    submit_pressed = st.form_submit_button("Analyze Feature", type="primary", use_container_width=True)

if submit_pressed:
    st.session_state.user_feature_input = feature_input
    with st.spinner("Parsing logic and running financial simulations..."):
        try:
            # Phase 1: Parse Spec
            feature_spec = parser_agent.parse(feature_input)
            
            # Phase 2: Simulate Financial Impact
            metrics = simulator.simulate(feature_spec)
            
            # Phase 3: Validate Simulation Data
            validation_report = validator_agent.validate(feature_spec, metrics)
            
            # Phase 4: Risk Analysis
            risk_report = risk_agent.analyze(feature_spec, metrics)
            
            # Phase 5: Compliance Checklist
            compliance_report = compliance_agent.generate_checklist(risk_report)
            
            # Save all outputs into session state so they persist across reruns
            st.session_state.results = {
                "feature_spec": feature_spec,
                "metrics": metrics,
                "validation_report": validation_report,
                "risk_report": risk_report,
                "compliance_report": compliance_report
            }
            
        except Exception as e:
            st.error(f"An error occurred during analysis:\n{str(e)}")
            st.info("Ensure to check your provider choice, provided API key, or your locally running models.")
            st.stop()
            
# --- MAIN CONTENT AREA ---
if "results" not in st.session_state:
    st.info("👈 Enter a feature description in the sidebar and click **Analyze Feature** to generate a risk dashboard.")
else:
    res = st.session_state.results
    
    # Safely handle corrupted or migrated session state caches
    try:
        feature_spec = res["feature_spec"]
        metrics = res["metrics"]
        validation_report = res["validation_report"]
        risk_report = res["risk_report"]
        compliance_report = res["compliance_report"]
        
        # Verify schema hasn't changed underneath
        if not hasattr(compliance_report, "critical_actions"):
            raise ValueError("Stale schema detected in st.session_state")
            
    except (KeyError, ValueError, AttributeError):
        st.session_state.clear()
        st.rerun()

    # --- HERO METRICS SECTION ---
    st.header(f"Feature: {feature_spec.feature_name}")
    score = compliance_report.readiness_score
    score_color = "🟢" if score >= 80 else "🟡" if score >= 50 else "🔴"
    rec_color = {"APPROVE": "🟢", "REVIEW": "🟡", "BLOCK": "🔴"}[compliance_report.recommendation.value]
    risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[risk_report.overall_risk_level.value]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System Recommendation", f"{rec_color} {compliance_report.recommendation.value}")
    with col2:
        st.metric("Readiness Score", f"{score_color} {score}/100")
    with col3:
        st.metric("Overall Risk Level", f"{risk_color} {risk_report.overall_risk_level.value}")
    with col4:
        st.metric("Impacted Customers", f"{metrics.impacted_customer_count:,} ({metrics.impacted_customer_percentage:.1f}%)")
    
    st.divider()
    
    # --- TABBED DEEP DIVES ---
    tab_spec, tab_finance, tab_risk, tab_launch = st.tabs([
        "📄 Parsed Spec",
        "💰 Financial Impact", 
        "⚖️ Risk Analysis",
        "📋 Launch Readiness"
    ])
    
    with tab_spec:
        colS1, colS2, colS3, colS4 = st.columns(4)
        colS1.metric("Feature Name", feature_spec.feature_name)
        colS2.metric("Fee Type", feature_spec.fee_type.value.capitalize())
        colS3.metric("Fee Value", f"{feature_spec.fee_value}{'%' if feature_spec.fee_type.value == 'percentage' else '$'}")
        colS4.metric("Applies To", feature_spec.applies_to)
        
        with st.expander("🛠️ View Database Query Logic"):
            st.code(f"Query: {feature_spec.condition}")
            if feature_spec.assumptions:
                st.markdown("**Assumptions made by LLM:**")
                for a in feature_spec.assumptions:
                    st.markdown(f"- {a}")

    with tab_finance:
        if not validation_report.is_valid:
            st.error(f"⚠️ **Simulation Validation Warning:** {validation_report.notes}")
            if validation_report.anomalies:
                for anom in validation_report.anomalies:
                    st.markdown(f"- *{anom}*")
        
        st.markdown("##### Revenue Projections")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Conservative", f"${metrics.total_revenue_estimate.conservative:,.0f}/yr")
        metrics_col2.metric("Realistic", f"${metrics.total_revenue_estimate.realistic:,.0f}/yr")
        metrics_col3.metric("Optimistic", f"${metrics.total_revenue_estimate.optimistic:,.0f}/yr")
        
        st.markdown("##### Customer Impact Dynamics")
        colA, colB = st.columns(2)
        colA.metric("Concentration Index", f"{metrics.concentration_index:.2f}",
                   help="Gini index measuring how concentrated the fee burden is across account balances.")
        
        if metrics.income_distribution_impacted:
            with colB:
                dist_df = pd.DataFrame(
                     list(metrics.income_distribution_impacted.items()), 
                     columns=['Income Band', 'Percentage']
                ).set_index('Income Band')
                st.bar_chart(dist_df, height=180)

    with tab_risk:
        tab_fairness, tab_reg, tab_rep, tab_conc = st.tabs(["⚖️ Fairness", "📜 Regulatory", "📰 Reputational", "🎯 Concentration"])
        
        def render_risk_items(items):
            if not items:
                st.success("No risks identified in this category.")
            for item in items:
                color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[item.severity.value]
                with st.container(border=True):
                    st.markdown(f"**{color} {item.title}**")
                    st.markdown(f"*{item.evidence}*")
                    st.markdown(f"**Mitigation:** {item.mitigation}")
                    
        with tab_fairness: render_risk_items(risk_report.fairness_risks)
        with tab_reg: render_risk_items(risk_report.regulatory_risks)
        with tab_rep: render_risk_items(risk_report.reputational_risks)
        with tab_conc: render_risk_items(risk_report.concentration_risks)
        
    with tab_launch:
        st.markdown(f"#### Readiness Score: {score}/100")
        st.progress(score / 100)
        
        if compliance_report.critical_actions:
            st.error("**Critical Actions Required Before Launch:**")
            for action in compliance_report.critical_actions:
                st.markdown(f"- 🚨 {action}")
            
        st.markdown("**Standard Operating Checklists:**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.checkbox("Legal/Compliance review")
            st.checkbox("Customer communications template created")
            st.checkbox("Risk mitigation plan formally approved")
            st.checkbox("Technical implementation validated")
        with col_c2:
            st.checkbox("Customer Support team trained")
            st.checkbox("Monitoring alerts & rollback ready")
            st.checkbox("Executive sign-off obtained")
            
        st.divider()
        st.markdown("### Launch Decision Engine")
        col_d1, col_d2, col_d3 = st.columns(3)
        
        if col_d1.button("✅ APPROVE LAUNCH", type="primary", use_container_width=True):
            if score < 100:
                st.warning("Cannot approve: Address critical risks and complete readiness checklists.")
            else:
                st.success("Feature approved for launch! 🚀")
                st.balloons()
                
        if col_d2.button("⏱️ REQUEST REVIEW", use_container_width=True):
            st.info("Feature sent to compliance committee for detailed review. Notifying team...")
            
        if col_d3.button("⛔ BLOCK FEATURE", type="primary", use_container_width=True):
            st.error("Feature blocked permanently. Return to ideation phase considering the severe risks.")
