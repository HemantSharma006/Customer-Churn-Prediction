from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("model/churn_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------
    # GET CUSTOMER INPUT
    # -----------------------------

    customer_data = {
        "gender": request.form["gender"],
        "SeniorCitizen": int(request.form["SeniorCitizen"]),
        "Partner": request.form["Partner"],
        "Dependents": request.form["Dependents"],
        "tenure": int(request.form["tenure"]),
        "PhoneService": request.form["PhoneService"],
        "MultipleLines": request.form["MultipleLines"],
        "InternetService": request.form["InternetService"],
        "OnlineSecurity": request.form["OnlineSecurity"],
        "OnlineBackup": request.form["OnlineBackup"],
        "DeviceProtection": request.form["DeviceProtection"],
        "TechSupport": request.form["TechSupport"],
        "StreamingTV": request.form["StreamingTV"],
        "StreamingMovies": request.form["StreamingMovies"],
        "Contract": request.form["Contract"],
        "PaperlessBilling": request.form["PaperlessBilling"],
        "PaymentMethod": request.form["PaymentMethod"],
        "MonthlyCharges": float(request.form["MonthlyCharges"]),
        "TotalCharges": float(request.form["TotalCharges"])
    }

    # Convert input into DataFrame
    input_data = pd.DataFrame([customer_data])


    # -----------------------------
    # MACHINE LEARNING PREDICTION
    # -----------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    probability_percentage = round(probability * 100, 2)


    # -----------------------------
    # RISK CLASSIFICATION
    # -----------------------------

    if probability_percentage < 30:

        risk_class = "low"
        risk_label = "Low Risk"

        result = "Customer Likely to Stay"

        message = (
            "This customer shows a relatively low likelihood "
            "of leaving the service based on the provided information."
        )

        recommendation = (
            "Customer appears relatively stable. Continue normal "
            "engagement and maintain good service quality."
        )

        recommendation_action = (
            "● Maintain regular customer engagement"
        )

    elif probability_percentage < 60:

        risk_class = "medium"
        risk_label = "Medium Risk"

        result = "Customer at Moderate Churn Risk"

        message = (
            "This customer shows a moderate likelihood of churn. "
            "Proactive engagement may help improve retention."
        )

        recommendation = (
            "Consider proactive engagement with this customer. "
            "Personalized communication, service improvements, "
            "or loyalty benefits may reduce churn risk."
        )

        recommendation_action = (
            "● Proactive retention recommended"
        )

    else:

        risk_class = "high"
        risk_label = "High Risk"

        result = "Customer Likely to Churn"

        message = (
            "This customer shows a high likelihood of churn. "
            "Retention efforts should be considered."
        )

        recommendation = (
            "High-priority retention action is recommended. "
            "Consider a personalized offer, loyalty benefit, "
            "service improvement, or direct customer support."
        )

        recommendation_action = (
            "● Immediate attention recommended"
        )


    # -----------------------------
    # KEY PREDICTION FACTORS
    # -----------------------------

    factors = []


    # Contract
    if customer_data["Contract"] == "Month-to-month":

        factors.append({
            "type": "risk",
            "icon": "↑",
            "title": "Month-to-Month Contract",
            "description": "Increases churn risk"
        })

    elif customer_data["Contract"] == "One year":

        factors.append({
            "type": "safe",
            "icon": "↓",
            "title": "One-Year Contract",
            "description": "Provides better retention"
        })

    elif customer_data["Contract"] == "Two year":

        factors.append({
            "type": "safe",
            "icon": "↓",
            "title": "Two-Year Contract",
            "description": "Strong retention indicator"
        })


    # Tenure
    if customer_data["tenure"] <= 6:

        factors.append({
            "type": "risk",
            "icon": "↑",
            "title": f"Short Tenure ({customer_data['tenure']} Months)",
            "description": "Increases churn risk"
        })

    elif customer_data["tenure"] >= 24:

        factors.append({
            "type": "safe",
            "icon": "↓",
            "title": f"Long Tenure ({customer_data['tenure']} Months)",
            "description": "Supports customer retention"
        })

    else:

        factors.append({
            "type": "safe",
            "icon": "↓",
            "title": f"Tenure ({customer_data['tenure']} Months)",
            "description": "Moderate customer relationship duration"
        })


    # Monthly charges
    if customer_data["MonthlyCharges"] >= 80:

        factors.append({
            "type": "risk",
            "icon": "↑",
            "title": f"High Monthly Charges (${customer_data['MonthlyCharges']:.2f})",
            "description": "May increase churn risk"
        })

    elif customer_data["MonthlyCharges"] <= 50:

        factors.append({
            "type": "safe",
            "icon": "↓",
            "title": f"Monthly Charges (${customer_data['MonthlyCharges']:.2f})",
            "description": "Lower pricing factor"
        })

    else:

        factors.append({
            "type": "safe",
            "icon": "↓",
            "title": f"Monthly Charges (${customer_data['MonthlyCharges']:.2f})",
            "description": "Moderate pricing factor"
        })


    # Internet service
    if customer_data["InternetService"] == "Fiber optic":

        factors.append({
            "type": "risk",
            "icon": "↑",
            "title": "Fiber Optic Internet",
            "description": "Can be associated with higher churn"
        })

    elif customer_data["InternetService"] == "DSL":

        factors.append({
            "type": "safe",
            "icon": "↓",
            "title": "DSL Internet",
            "description": "Lower-risk service category"
        })


    # Technical support
    if customer_data["TechSupport"] == "No":

        factors.append({
            "type": "risk",
            "icon": "↑",
            "title": "No Technical Support",
            "description": "May increase churn risk"
        })

    else:

        factors.append({
            "type": "safe",
            "icon": "↓",
            "title": "Technical Support Enabled",
            "description": "Supports customer retention"
        })


    # Online security
    if customer_data["OnlineSecurity"] == "No":

        factors.append({
            "type": "risk",
            "icon": "↑",
            "title": "No Online Security",
            "description": "May increase churn risk"
        })

    else:

        factors.append({
            "type": "safe",
            "icon": "↓",
            "title": "Online Security Enabled",
            "description": "Supports customer retention"
        })


    # -----------------------------
    # DISPLAY RESULT
    # -----------------------------

    return render_template(
        "result.html",

        result=result,

        probability=probability_percentage,

        risk_class=risk_class,

        risk_label=risk_label,

        message=message,

        factors=factors,

        recommendation=recommendation,

        recommendation_action=recommendation_action
    )


# -----------------------------
# RUN APPLICATION
# -----------------------------

if __name__ == "__main__":
 app.run(host="0.0.0.0", port=5000, debug=True)