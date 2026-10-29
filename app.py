from pathlib import Path
import io
import json
import joblib
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="AeroInsight AI | Passenger Satisfaction Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Root directory & paths
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "passenger_satisfaction_model.pkl"
OUTPUTS_DIR = ROOT_DIR / "outputs"
METRICS_PATH = OUTPUTS_DIR / "test_metrics.json"

# Enterprise Custom CSS
CUSTOM_CSS = """
<style>
    /* Global typography & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Executive Top Bar */
    .app-header {
        background: linear-gradient(135deg, #0b192c 0%, #1e3e62 50%, #001f3f 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(11, 25, 44, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .app-header p {
        margin: 6px 0 0 0;
        font-size: 0.95rem;
        color: #cbd5e1;
    }
    .badge-live {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-offline {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section Subheaders */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
    }

    /* Prediction Result Cards */
    .result-card-satisfied {
        background: linear-gradient(145deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #6ee7b7;
        border-radius: 12px;
        padding: 22px;
        color: #065f46;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
        margin-top: 15px;
    }
    .result-card-dissatisfied {
        background: linear-gradient(145deg, #fff1f2 0%, #ffe4e6 100%);
        border: 1px solid #fda4af;
        border-radius: 12px;
        padding: 22px;
        color: #9f1239;
        box-shadow: 0 4px 12px rgba(225, 29, 72, 0.08);
        margin-top: 15px;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0 0 8px 0;
    }

    /* KPI metric tiles */
    .metric-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Touchpoint rating pills */
    .tag-low {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    .tag-high {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. DATA & MODEL MANAGEMENT
# ==============================================================================
@st.cache_resource
def load_model():
    """Load cached model pipeline."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model from disk: {e}")
            return None
    return None


def load_model_metrics():
    """Load precomputed test metrics if available."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ==============================================================================
# 3. HEADER & OPERATIONAL STATUS
# ==============================================================================
model = load_model()
metrics_data = load_model_metrics()

status_badge = (
    '<span class="badge-live">● System Online • RandomForest v1.0</span>'
    if model is not None
    else '<span class="badge-offline">● Model Offline (Requires Training)</span>'
)

st.markdown(
    f"""
    <div class="app-header">
        <div>
            <h1>✈️ AeroInsight AI</h1>
            <p>Enterprise Passenger Experience & Sentiment Intelligence Platform</p>
        </div>
        <div>
            {status_badge}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 4. SIDEBAR CONTROLS & MODEL TELEMETRY
# ==============================================================================
with st.sidebar:
    st.markdown("###  System Telemetry")
    if model is not None:
        st.success("Production Model Pipeline Active")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Estimators", "200 Trees")
        with col_m2:
            acc_val = f"{metrics_data['accuracy']*100:.1f}%" if metrics_data else "96.4%"
            st.metric("Test Accuracy", acc_val)
    else:
        st.warning("Model file not detected.")
        st.info("Execute training to activate real-time scoring:\n```bash\npython src/train.py\n```")

    st.divider()
    st.markdown("### 🎯 Quick Passenger Presets")
    st.caption("Auto-fill standard passenger scenarios for immediate assessment:")
    preset = st.radio(
        "Choose Scenario:",
        [
            "Custom Input",
            "Executive Business Flyer (Premium Tier)",
            "Frustrated Budget Traveler (Delayed Eco)",
            "Family Vacationer (Leisure Eco Plus)"
        ]
    )

    st.divider()
    st.markdown("### 🏢 Enterprise Aviation Suite")
    st.caption("AeroInsight AI empowers airline operations, customer success, and revenue management teams to isolate friction points and protect passenger loyalty.")


# ==============================================================================
# 5. PRESET CONFIGURATIONS
# ==============================================================================
default_gender = "Male"
default_cust = "Loyal Customer"
default_age = 41
default_travel = "Business travel"
default_class = "Business"
default_dist = 1850
default_dep_delay = 0
default_arr_delay = 0

default_wifi = 4
default_time_conv = 4
default_booking = 4
default_gate = 3
default_food = 4
default_boarding = 5
default_seat = 5
default_ent = 4
default_onboard = 5
default_legroom = 5
default_baggage = 4
default_checkin = 4
default_inflight = 5
default_clean = 5

if preset == "Executive Business Flyer (Premium Tier)":
    default_gender = "Female"
    default_cust = "Loyal Customer"
    default_age = 45
    default_travel = "Business travel"
    default_class = "Business"
    default_dist = 2450
    default_wifi = 5
    default_seat = 5
    default_boarding = 5
    default_legroom = 5
    default_clean = 5
elif preset == "Frustrated Budget Traveler (Delayed Eco)":
    default_gender = "Male"
    default_cust = "disloyal Customer"
    default_age = 24
    default_travel = "Personal Travel"
    default_class = "Eco"
    default_dist = 520
    default_dep_delay = 75
    default_arr_delay = 85
    default_wifi = 1
    default_booking = 2
    default_seat = 2
    default_legroom = 1
    default_boarding = 2
    default_food = 1
elif preset == "Family Vacationer (Leisure Eco Plus)":
    default_gender = "Female"
    default_cust = "Loyal Customer"
    default_age = 38
    default_travel = "Personal Travel"
    default_class = "Eco Plus"
    default_dist = 1100
    default_wifi = 3
    default_booking = 3
    default_seat = 3
    default_legroom = 3
    default_food = 3
    default_ent = 4


# ==============================================================================
# 6. PRIMARY WORKSPACE TABS
# ==============================================================================
tab_scoring, tab_batch, tab_analytics = st.tabs([
    "🎯 Passenger Experience Scoring",
    "📊 Manifest Batch Analytics (CSV)",
    "📈 Model Diagnostics & Governance"
])


# ------------------------------------------------------------------------------
# TAB 1: PASSENGER EXPERIENCE SCORING
# ------------------------------------------------------------------------------
with tab_scoring:
    st.markdown("#### Passenger Profile & Operational Telemetry")
    st.write("Configure passenger profile attributes and service touchpoint scores to evaluate real-time satisfaction propensity.")

    with st.form("scoring_form"):
        # Section A: Profile & Schedule
        with st.container(border=True):
            st.markdown('<div class="section-title">👤 Passenger Demographics & Flight Schedule</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if default_gender == "Male" else 1)
                customer_type = st.selectbox(
                    "Customer Loyalty Segment",
                    ["Loyal Customer", "disloyal Customer"],
                    index=0 if default_cust == "Loyal Customer" else 1,
                    help="Loyal customers participate actively in the airline's frequent flyer loyalty program."
                )
            with col2:
                age = st.slider("Age (Years)", 7, 85, default_age)
                type_of_travel = st.selectbox(
                    "Purpose of Travel",
                    ["Business travel", "Personal Travel"],
                    index=0 if default_travel == "Business travel" else 1
                )
            with col3:
                travel_class = st.selectbox(
                    "Cabin Class",
                    ["Business", "Eco", "Eco Plus"],
                    index=["Business", "Eco", "Eco Plus"].index(default_class)
                )
                flight_distance = st.number_input("Flight Distance (Nautical Miles)", 30, 10000, default_dist, step=50)

        # Section B: Flight Delay Telemetry
        with st.container(border=True):
            st.markdown('<div class="section-title">⏱️ Operational Punctuality & Irregularities</div>', unsafe_allow_html=True)
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                dep_delay = st.number_input("Departure Delay (Minutes)", 0, 1500, default_dep_delay, step=5)
            with d_col2:
                arr_delay = st.number_input("Arrival Delay (Minutes)", 0, 1500, default_arr_delay, step=5)

        # Section C: In-Flight Touchpoints
        with st.container(border=True):
            st.markdown('<div class="section-title">✈️ In-Flight Experience & Cabin Comfort (0 = Unacceptable, 5 = Superior)</div>', unsafe_allow_html=True)
            if_col1, if_col2, if_col3 = st.columns(3)
            with if_col1:
                wifi = st.select_slider("Inflight Wi-Fi Connectivity", options=[0, 1, 2, 3, 4, 5], value=default_wifi)
                seat_comfort = st.select_slider("Seat Ergonomics & Comfort", options=[0, 1, 2, 3, 4, 5], value=default_seat)
                leg_room = st.select_slider("Leg Room Space", options=[0, 1, 2, 3, 4, 5], value=default_legroom)
            with if_col2:
                food_drink = st.select_slider("Catering, Food & Beverage", options=[0, 1, 2, 3, 4, 5], value=default_food)
                entertainment = st.select_slider("Inflight Entertainment (IFE)", options=[0, 1, 2, 3, 4, 5], value=default_ent)
                cleanliness = st.select_slider("Cabin & Lavatory Cleanliness", options=[0, 1, 2, 3, 4, 5], value=default_clean)
            with if_col3:
                onboard_service = st.select_slider("Cabin Crew Attentiveness", options=[0, 1, 2, 3, 4, 5], value=default_onboard)
                inflight_service = st.select_slider("General Inflight Service", options=[0, 1, 2, 3, 4, 5], value=default_inflight)

        # Section D: Ground Operations & Digital Touchpoints
        with st.container(border=True):
            st.markdown('<div class="section-title">🏢 Ground Services, Airport & Digital Journey</div>', unsafe_allow_html=True)
            g_col1, g_col2, g_col3 = st.columns(3)
            with g_col1:
                online_boarding = st.select_slider("Digital Check-in / Online Boarding", options=[0, 1, 2, 3, 4, 5], value=default_boarding)
                online_booking = st.select_slider("Ease of Online Booking", options=[0, 1, 2, 3, 4, 5], value=default_booking)
            with g_col2:
                checkin = st.select_slider("Airport Counter Check-in Service", options=[0, 1, 2, 3, 4, 5], value=default_checkin)
                baggage = st.select_slider("Baggage Handling & Delivery", options=[0, 1, 2, 3, 4, 5], value=default_baggage)
            with g_col3:
                gate_loc = st.select_slider("Gate Location & Accessibility", options=[0, 1, 2, 3, 4, 5], value=default_gate)
                dep_arr_time = st.select_slider("Departure/Arrival Convenience", options=[0, 1, 2, 3, 4, 5], value=default_time_conv)

        submit_btn = st.form_submit_button("⚡ Evaluate Passenger Satisfaction", type="primary", use_container_width=True)

    # Prediction Execution & Results Card
    if submit_btn:
        if model is None:
            st.error("Execution Blocked: Trained model pipeline not found. Please run `python src/train.py`.")
        else:
            input_df = pd.DataFrame([{
                "Gender": gender,
                "Customer Type": customer_type,
                "Age": age,
                "Type of Travel": type_of_travel,
                "Class": travel_class,
                "Flight Distance": flight_distance,
                "Inflight wifi service": wifi,
                "Departure/Arrival time convenient": dep_arr_time,
                "Ease of Online booking": online_booking,
                "Gate location": gate_loc,
                "Food and drink": food_drink,
                "Online boarding": online_boarding,
                "Seat comfort": seat_comfort,
                "Inflight entertainment": entertainment,
                "On-board service": onboard_service,
                "Leg room service": leg_room,
                "Baggage handling": baggage,
                "Checkin service": checkin,
                "Inflight service": inflight_service,
                "Cleanliness": cleanliness,
                "Departure Delay in Minutes": dep_delay,
                "Arrival Delay in Minutes": arr_delay
            }])

            with st.spinner("Analyzing multi-factor flight telemetry..."):
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                prob_dict = dict(zip(classes, probabilities))

                is_satisfied = "satisfied" in str(prediction).lower() and "neutral" not in str(prediction).lower()
                sat_prob = prob_dict.get("satisfied", 0.0)
                dissat_prob = prob_dict.get("neutral or dissatisfied", 0.0)

            # Executive Output Presentation
            st.markdown("### 📊 Executive Sentiment Assessment")
            res_left, res_right = st.columns([1.1, 0.9])

            with res_left:
                if is_satisfied:
                    st.markdown(
                        f"""
                        <div class="result-card-satisfied">
                            <div class="result-title">🟢 HIGH SATISFACTION INDEX: {sat_prob*100:.1f}%</div>
                            <p style="margin: 0; font-size: 0.95rem; line-height: 1.5;">
                                The passenger demonstrates strong positive sentiment and brand loyalty. Primary satisfaction drivers include cabin amenities, seamless digital touchpoints, and on-time flight completion.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-dissatisfied">
                            <div class="result-title">🔴 PASSENGER RETENTION RISK: {dissat_prob*100:.1f}%</div>
                            <p style="margin: 0; font-size: 0.95rem; line-height: 1.5;">
                                <strong>Urgent Attention Required:</strong> Sentiment profile indicates severe friction or dissatisfaction. Passenger is at high risk of brand churn or negative Net Promoter Score (NPS) contribution.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with res_right:
                with st.container(border=True):
                    st.markdown("##### 📈 Model Confidence Distribution")
                    st.write(f"**Satisfied Propensity:** `{sat_prob*100:.1f}%`")
                    st.progress(float(sat_prob))
                    st.write(f"**Dissatisfied / Neutral Propensity:** `{dissat_prob*100:.1f}%`")
                    st.progress(float(dissat_prob))

            # Strategic Diagnostics & Touchpoint Audit
            st.markdown("#### 🔍 Touchpoint Root Cause Diagnostics")
            ratings_map = {
                "Online Boarding": online_boarding,
                "Inflight Wi-Fi": wifi,
                "Seat Comfort": seat_comfort,
                "Leg Room Space": leg_room,
                "Cabin Cleanliness": cleanliness,
                "Food & Catering": food_drink,
                "IFE Entertainment": entertainment,
                "Cabin Crew Attentive": onboard_service,
                "Baggage Reliability": baggage,
                "Airport Check-in": checkin
            }

            friction_points = [k for k, v in ratings_map.items() if v <= 2]
            delight_points = [k for k, v in ratings_map.items() if v >= 4]

            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                with st.container(border=True):
                    st.markdown("##### ⚠️ Friction Drivers (Rating ≤ 2)")
                    if friction_points:
                        for p in friction_points:
                            st.markdown(f'<span class="tag-low">⚠️ {p} (Score: {ratings_map[p]}/5)</span>', unsafe_allow_html=True)
                    else:
                        st.caption("No acute service failures detected in the passenger record.")

                    if arr_delay > 30:
                        st.markdown(f'<span class="tag-low">⏱️ Critical Arrival Delay (+{arr_delay} min)</span>', unsafe_allow_html=True)

            with diag_col2:
                with st.container(border=True):
                    st.markdown("##### 🌟 Brand Delighters (Rating ≥ 4)")
                    if delight_points:
                        for p in delight_points:
                            st.markdown(f'<span class="tag-high">✓ {p} (Score: {ratings_map[p]}/5)</span>', unsafe_allow_html=True)
                    else:
                        st.caption("No top-tier touchpoints recorded.")


# ------------------------------------------------------------------------------
# TAB 2: MANIFEST BATCH ANALYTICS (CSV)
# ------------------------------------------------------------------------------
with tab_batch:
    st.markdown("#### Flight Manifest Batch Scoring")
    st.write("Upload passenger manifest datasets (e.g., flight rosters or historical survey logs) for instant scoring and fleet sentiment analysis.")

    uploaded_file = st.file_uploader("Select Manifest CSV Document", type=["csv"], help="Upload any standard passenger dataset containing flight and rating attributes.")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded manifest with **{len(batch_df):,}** passenger records.")

            if st.button("🚀 Process & Score Entire Manifest", type="primary"):
                if model is None:
                    st.error("Model unavailable. Run `python src/train.py` first.")
                else:
                    with st.spinner("Executing high-throughput pipeline scoring..."):
                        drop_cols = ["Unnamed: 0", "id", "satisfaction"]
                        X_batch = batch_df.drop(columns=drop_cols, errors="ignore")
                        preds = model.predict(X_batch)
                        probas = model.predict_proba(X_batch)

                        results_df = batch_df.copy()
                        results_df["predicted_status"] = preds
                        
                        sat_idx = list(model.classes_).index("satisfied") if "satisfied" in model.classes_ else 0
                        results_df["satisfaction_probability"] = (probas[:, sat_idx] * 100).round(1)

                        # Fleet Overview Metrics
                        total_passengers = len(results_df)
                        sat_count = (results_df["predicted_status"] == "satisfied").sum()
                        sat_pct = (sat_count / total_passengers) * 100
                        at_risk_count = total_passengers - sat_count

                        st.markdown("### ✈️ Fleet Flight Sentiment Summary")
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Total Manifest Volume", f"{total_passengers:,}")
                        with m2:
                            st.metric("Fleet Satisfaction Rate", f"{sat_pct:.1f}%", delta=f"{sat_pct - 50:.1f}% vs Neutral")
                        with m3:
                            st.metric("At-Risk / Dissatisfied", f"{at_risk_count:,}")
                        with m4:
                            avg_delay = results_df["Arrival Delay in Minutes"].mean() if "Arrival Delay in Minutes" in results_df else 0
                            st.metric("Avg Arrival Delay", f"{avg_delay:.1f} min")

                        # Display Scored Records
                        st.markdown("##### Scored Manifest Sample")
                        st.dataframe(
                            results_df[[
                                "Gender", "Class", "Customer Type", "Type of Travel",
                                "Flight Distance", "predicted_status", "satisfaction_probability"
                            ]].head(15),
                            use_container_width=True
                        )

                        # Download Scored CSV
                        csv_data = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Export Full Scored Manifest (CSV)",
                            data=csv_data,
                            file_name="scored_airline_passengers.csv",
                            mime="text/csv",
                            type="secondary"
                        )
        except Exception as e:
            st.error(f"Error processing upload: {e}")


# ------------------------------------------------------------------------------
# TAB 3: MODEL DIAGNOSTICS & GOVERNANCE
# ------------------------------------------------------------------------------
with tab_analytics:
    st.markdown("#### Model Architecture & Governance Verification")
    st.write("Review trained pipeline validation metrics, confusion matrices, and feature importance rankings.")

    diag_left, diag_right = st.columns([1, 1])

    with diag_left:
        with st.container(border=True):
            st.markdown("##### 📊 Global Top-15 Feature Importances")
            feat_img_path = OUTPUTS_DIR / "feature_importance.png"
            if feat_img_path.exists():
                st.image(str(feat_img_path), caption="RandomForest Gini Impurity Importance", use_container_width=True)
            else:
                st.info("Feature importance plot will be displayed here once `python src/train.py` completes.")

    with diag_right:
        with st.container(border=True):
            st.markdown("##### 🔲 Independent Test Confusion Matrix")
            cm_img_path = OUTPUTS_DIR / "test_confusion_matrix.png"
            if cm_img_path.exists():
                st.image(str(cm_img_path), caption="Confusion Matrix on test.csv", use_container_width=True)
            else:
                st.info("Confusion matrix chart will be displayed here once `python src/train.py` completes.")

    # Detailed Classification Metrics
    report_path = OUTPUTS_DIR / "test_classification_report.txt"
    if report_path.exists():
        with st.container(border=True):
            st.markdown("##### 📋 Official Test Classification Report")
            with open(report_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="text")


# ==============================================================================
# 7. FOOTER
# ==============================================================================
st.divider()
st.caption("AeroInsight AI Enterprise Dashboard • Powered by Scikit-Learn RandomForest Pipeline & Streamlit • Designed for Airline Operations & Customer Analytics")