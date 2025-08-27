from flask import Flask, redirect, request, render_template, make_response, session, url_for
import joblib
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from io import BytesIO
import mysql.connector
import hashlib
import os, json, re
from flask import Flask, jsonify
from pathlib import Path
from dotenv import load_dotenv
from flask_cors import CORS
from openai import OpenAI
from chatbot.tools.sales import SalesData
from chatbot.tools.forecast import Forecaster
from chatbot.tools.strategies import Strategies
from chatbot.tools import rag as ragtools

app = Flask(__name__, static_folder='static', static_url_path="/static")
app.secret_key = "my_secret_123"
CORS(app)

# --- External APIs ---
API_TOKEN = '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'
headers = {'Authorization': f'Bearer {API_TOKEN}'}

# --- DB ---
def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Narm819@2004",
        database="flask_login"
    )
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Models / Metadata (single source of truth) ---
MODEL_PATHS = {
    'M01AB': {
        'value': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_M01AB_value_forecast.joblib',
        'sales': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_M01AB.joblib',
        'uses': "treating pain and inflammation, particularly in conditions like arthritis",
    },
    'M01AE': {
        'value': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_M01AE_value_forecast.joblib',
        'sales': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_M01AE.joblib',
        'uses': "treating anxiety, seizures, and insomnia",
    },
    'N02BA': {
        'value': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_N02BA_value_forecast.joblib',
        'sales': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_N02BA.joblib',
        'uses': "relieving pain, reducing fever, and acting as an anti-inflammatory",
    },
    'N02BE': {
        'value': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_Paracetamol_value_new.joblib',
        'sales': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_N02BE.joblib',
        'uses': "relieving pain and reducing fever",
    },
    'N05B': {
        'value': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_N05B_value_forecast.joblib',
        'sales': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_N05B.joblib',
        'uses': "treating anxiety, seizures, and insomnia",
    },
    'N05C': {
        'value': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_N05C_value_forecast.joblib',
        'sales': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_N05C.joblib',
        'uses': "treating insomnia and helping with sleep initiation",
    },
    'R03': {
        'value': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_R03_value_forecast.joblib',
        'sales': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_R03.joblib',
        'uses': "treating anxiety, seizures, and insomnia",
    },
    'R06': {
        'value': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_R06_value_forecast.joblib',
        'sales': r'C:\Users\balan\OneDrive\Desktop\SalesForecasting\models\sarimax_model_R06.joblib',
        'uses': "relieving allergy symptoms such as hay fever, sneezing, runny nose, and itchy eyes",
    },
}





#-----------------chatbot-------------------------------
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
MODEL_ID = os.getenv("MODEL_ID","google/gemini-flash-1.5")

DATA_PATH  = Path(os.getenv("DATA_PATH","./chatbot/data/salesmonthly.csv"))
MODEL_DIR  = Path(os.getenv("MODEL_DIR","./models"))
CHROMA_DIR = os.getenv("CHROMA_DIR","./chatbot/knowledgebase/chroma")

sales       = SalesData(DATA_PATH)
forecaster  = Forecaster(MODEL_DIR, sales.df)
strategies  = Strategies(Path("./chatbot/knowledgebase/strategies_rules.json"))
ragtools.ingest("./chatbot/knowledgebase", CHROMA_DIR)

def system_prompt():
    return (
        "You are Pharma Sales Agent. Be brief and factual. "
        "Capabilities:\n"
        "1) lookup_sales(year, month, product)\n"
        "2) top_products(year, k)\n"
        "3) peak_month(product)\n"
        "4) yoy_mom(product, year, month)\n"
        "5) forecast_sales(product, periods)\n"
        "6) strategy_suggest(trend, mom_pct, yoy_pct)\n"
        "7) kb_search(query) for website/dashboard docs.\n"
        "Always use tools for numeric answers, and give currency/units if known."
    )

tools = [
    {"type":"function","function":{
        "name":"lookup_sales",
        "description":"Total sales filtered by optional year, month, product.",
        "parameters":{"type":"object","properties":{
            "year":{"type":"integer"},
            "month":{"type":"string","description":"e.g., May"},
            "product":{"type":"string"}
        }}
    }},
    {"type":"function","function":{
        "name":"top_products",
        "description":"Top products by total sales for a given year (optional).",
        "parameters":{"type":"object","properties":{
            "year":{"type":"integer"},
            "k":{"type":"integer","default":5}
        }}
    }},
    {"type":"function","function":{
        "name":"peak_month",
        "description":"Find peak month for a product.",
        "parameters":{"type":"object","properties":{"product":{"type":"string"}},"required":["product"]}
    }},
    {"type":"function","function":{
        "name":"yoy_mom",
        "description":"Return value and % change vs last month and last year.",
        "parameters":{"type":"object","properties":{
            "product":{"type":"string"},
            "year":{"type":"integer"},
            "month":{"type":"string"}
        },"required":["product","year","month"]}
    }},
    {"type":"function","function":{
        "name":"forecast_sales",
        "description":"Forecast with SARIMAX and create a Plotly HTML chart.",
        "parameters":{"type":"object","properties":{
            "product":{"type":"string"},
            "periods":{"type":"integer","default":12}
        },"required":["product"]}
    }},
    {"type":"function","function":{
        "name":"strategy_suggest",
        "description":"Short marketing tips using trend/MoM/YoY.",
        "parameters":{"type":"object","properties":{
            "trend":{"type":"string","enum":["up","flat","down"]},
            "mom_pct":{"type":"number"},
            "yoy_pct":{"type":"number"}
        }}
    }},
    {"type":"function","function":{
        "name":"kb_search",
        "description":"Search the knowledge base (website/dashboard).",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
    }}
]

def exec_tool(name, args):
    if name=="lookup_sales":
        return sales.sales_on(args.get("year"), args.get("month"), args.get("product"))
    if name=="top_products":
        return sales.top_products(args.get("year"), args.get("k",5))
    if name=="peak_month":
        return sales.peak_month(args["product"])
    if name=="yoy_mom":
        return sales.yoy_mom(args["product"], args["year"], args["month"])
    if name=="forecast_sales":
        return forecaster.forecast(args["product"], args.get("periods",12))
    if name=="strategy_suggest":
        tips = strategies.suggest(args.get("trend"), args.get("mom_pct"), args.get("yoy_pct"))
        return {"suggestions": tips}
    if name=="knowledgebase_search":
        return {"knowledgebase": ragtools.search(args["query"], CHROMA_DIR)}
    return {"error":"unknown tool"}



#------------------------------chatbot--------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    user_msg = body.get("message","").strip()
    messages = [
        {"role":"system","content":system_prompt()},
        {"role":"user","content":user_msg}
    ]
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2
    )
    msg = resp.choices[0].message

    # one tool-hop (extend to while-loop if you want multi-step)
    if getattr(msg, "tool_calls", None):
        call = msg.tool_calls[0]
        args = json.loads(call.function.arguments or "{}")
        result = exec_tool(call.function.name, args)
        messages.append({"role":"assistant","tool_calls":[call]})
        messages.append({"role":"tool","tool_call_id":call.id,"content":json.dumps(result)})

        final = client.chat.completions.create(
            model=MODEL_ID, messages=messages, temperature=0.2
        )
        return jsonify({"reply": final.choices[0].message.content, "tool_result": result})

    return jsonify({"reply": msg.content})




# --- Auth Routes ---

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip()
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM users WHERE email = %s AND password = %s",
            (email, hash_password(password))
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            session["user"] = {
                "name": user["name"],
                "email": user["email"]
            }
            return redirect(url_for("index"))
        else:
            return "❌ Invalid credentials! Try again."

    # If already logged in, go to /index
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                (name, email, hash_password(password))
            )
            conn.commit()
            return redirect(url_for("login"))
        except mysql.connector.IntegrityError:
            return "⚠️ Email already exists. Try logging in."
        finally:
            conn.close()

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# --- Home / Index / Input ---

@app.route("/")
def home():
    # Redirect to login if not authenticated, otherwise go to index
    if "user" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("index"))

@app.route("/index")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", name=session["user"]["name"])

@app.route("/input", methods=['GET'])
def input_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template('input.html')

# --- Forecast Utilities ------------------------------

def generate_forecast_explanation(forecast_df, product_name, product_uses):
    try:
        highest_sales_row = forecast_df.loc[forecast_df['Forecasted Sales'].idxmax()]
        lowest_sales_row = forecast_df.loc[forecast_df['Forecasted Sales'].idxmin()]

        start_date = forecast_df['Date'].min().strftime('%B %Y')
        end_date = forecast_df['Date'].max().strftime('%B %Y')

        highest_sales = highest_sales_row['Forecasted Sales']
        lowest_sales = lowest_sales_row['Forecasted Sales']
        highest_sales_date = highest_sales_row['Date'].strftime('%B %Y')
        lowest_sales_date = lowest_sales_row['Date'].strftime('%B %Y')

        lowest_possible_value = forecast_df['Lower Bound'].min()
        highest_possible_value = forecast_df['Upper Bound'].max()

        num_months = len(forecast_df['Date'].unique())

        explanation = (
            f"This forecast covers a total of {num_months} months of projected sales for {product_name}, "
            f"ranging from {start_date} to {end_date}. {product_name} is commonly used for {product_uses}. "
            f"Each entry shows the forecasted sales along with the lower and upper bounds of the prediction. "
            f"The highest forecasted sales of {highest_sales:.2f} are projected for {highest_sales_date}. "
            f"This could be due to increased demand during this period, possibly driven by seasonal factors, "
            f"promotional activities, or a rise in health issues that {product_name} addresses. "
            f"In contrast, the lowest forecasted sales of {lowest_sales:.2f} are expected on {lowest_sales_date}, "
            f"which may be attributed to reduced demand, competition, or a lesser prevalence of conditions that the product treats during this time. "
            f"Additionally, the lowest possible sales value during this period could go as low as {lowest_possible_value:.2f}, "
            f"while the highest possible sales value could reach up to {highest_possible_value:.2f}. "
            f"The range of lower and upper bounds indicates the potential variability in sales figures for each date. "
            f"Understanding these forecasts helps in effective planning by highlighting periods with potential peaks and troughs in sales."
        )
    except Exception as e:
        return f"Error generating the forecast explanation: {e}"
    return explanation

def get_summary(forecast_df, product_name, product_uses):
    explanation = generate_forecast_explanation(forecast_df, product_name, product_uses)
    if "Error" in explanation:
        return explanation

    payload = {
        'inputs': explanation,
        'parameters': {'max_length': 400, 'min_length': 150},
    }
    response = requests.post(
        'https://api-inference.huggingface.co/models/facebook/bart-large-cnn',
        headers=headers,
        json=payload
    )
    try:
        summary = response.json()
        if isinstance(summary, list) and 'summary_text' in summary[0]:
            return summary[0]['summary_text']
        else:
            return "No summary could be generated. Please try again."
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "There was an issue generating the summary. Please try again later."

def detect_trend(forecast_df):
    values = forecast_df['Forecasted Sales'].values
    slope = values[-1] - values[0]
    threshold = forecast_df['Forecasted Sales'].mean() * 0.05  # 5%

    if slope > threshold:
        return "Rising", "green"
    elif slope < -threshold:
        return "Falling", "red"
    else:
        return "Stable", "gray"

def create_plot(forecast_df, return_fig=False):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=forecast_df['Date'],
            y=forecast_df['Forecasted Sales'],
            name='Forecasted Sales (Bar)',
            marker=dict(color='lightblue'),
            opacity=0.6
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecasted Sales'],
            mode='lines+markers',
            name='Forecasted Sales (Line)',
            line=dict(color='blue', width=3),
            marker=dict(size=6, symbol='circle')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Lower Bound'],
            mode='lines+markers',
            name='Lower Bound',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6, symbol='triangle-down')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Upper Bound'],
            mode='lines+markers',
            name='Upper Bound',
            line=dict(color='green', width=2, dash='dot'),
            marker=dict(size=6, symbol='triangle-up')
        )
    )

    trend_text, trend_color = detect_trend(forecast_df)
    fig.add_annotation(
        text=f"Trend: {trend_text}",
        x=0.5, y=1.06, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color=trend_color, family="Arial Black"),
        bgcolor="white",
    )

    fig.update_layout(
        title=dict(text="Forecasted Sales with Confidence Intervals (Line and Bar Chart)", x=0.5, xanchor="center"),
        hovermode='x unified',
        yaxis_title="Sales",
        xaxis_title="Date",
        margin=dict(r=120, b=80, t=80),
        template='plotly_white',
        legend=dict(title="Legend", orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        barmode='overlay'
    )

    if return_fig:
        return fig
    return pio.to_html(fig, full_html=False)

def get_recommendations(forecast_df):
    recommendation = []
    max_sales_row = forecast_df.loc[forecast_df['Forecasted Sales'].idxmax()]
    min_sales_row = forecast_df.loc[forecast_df['Forecasted Sales'].idxmin()]

    recommendation.append("Recommendations:")
    recommendation.append(
        f" - Focus on increasing production for {max_sales_row['Date'].strftime('%B %Y')} "
        f"as it has the highest forecasted sales of {max_sales_row['Forecasted Sales']:.2f}."
    )
    recommendation.append(
        f" - Consider reducing inventory during {min_sales_row['Date'].strftime('%B %Y')}, "
        f"which has the lowest forecasted sales of {min_sales_row['Forecasted Sales']:.2f}."
    )
    recommendation.append(
        " - Monitor the sales closely during the months with high upper bound values, "
        "as these periods may require additional resources to meet potential demand."
    )
    return "\n".join(recommendation)

def run_forecast(product_name, from_date, to_date):
    if product_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported product '{product_name}'")

    value_model_path = MODEL_PATHS[product_name]['value']
    sales_model_path = MODEL_PATHS[product_name]['sales']

    value_model = joblib.load(value_model_path)
    sales_model = joblib.load(sales_model_path)

    forecast_start = pd.to_datetime(from_date, format='%Y-%m-%d')
    forecast_end = pd.to_datetime(to_date, format='%Y-%m-%d')
    forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='ME')

    value_forecast = value_model.get_forecast(steps=len(forecast_index))
    forecast_value_mean = value_forecast.predicted_mean.round().astype(int).values.reshape(-1, 1)

    sales_forecast = sales_model.get_forecast(steps=len(forecast_index), exog=forecast_value_mean)
    forecast_sales_mean = sales_forecast.predicted_mean
    forecast_sales_conf_int = sales_forecast.conf_int()

    forecast_sales_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted Sales': forecast_sales_mean.round(2),
        'Lower Bound': forecast_sales_conf_int.iloc[:, 0].clip(lower=0).round(2),
        'Upper Bound': forecast_sales_conf_int.iloc[:, 1].round(2)
    })
    return forecast_sales_df

# --- Forecast Routes ---

@app.route("/forecast", methods=["POST"])
def forecast():
    if "user" not in session:
        return redirect(url_for("login"))

    from_date = request.form['from_date']
    to_date = request.form['to_date']
    product_name = request.form['product']

    try:
        forecast_sales_df = run_forecast(product_name, from_date, to_date)
    except Exception as e:
        return f"Error while forecasting: {e}"

    # Build narrative + plot + recs
    product_uses = MODEL_PATHS[product_name]['uses']
    explanation = generate_forecast_explanation(forecast_sales_df, product_name, product_uses)
    summary = get_summary(forecast_sales_df, product_name, product_uses)
    sales_plot = create_plot(forecast_sales_df)
    recommendations = get_recommendations(forecast_sales_df)
    trend_text, trend_color = detect_trend(forecast_sales_df)

    # If you need row highlighting in the template, send min/max explicitly
    min_sales = float(forecast_sales_df['Forecasted Sales'].min())
    max_sales = float(forecast_sales_df['Forecasted Sales'].max())

    return render_template(
        'result.html',
        product_name=product_name,
        forecast_sales=forecast_sales_df.to_dict(orient='records'),
        explanation=explanation,
        summary=summary,
        recommendations=recommendations,
        sales_plot=sales_plot,
        trend_text=trend_text,
        trend_color=trend_color,
        min_sales=min_sales,
        max_sales=max_sales
    )

# --- Downloads ---

@app.route('/download-csv', methods=['POST'])
def download_csv():
    if "user" not in session:
        return redirect(url_for("login"))

    from_date = request.form['from_date']
    to_date = request.form['to_date']
    product_name = request.form['product']

    try:
        forecast_sales_df = run_forecast(product_name, from_date, to_date)
    except Exception as e:
        return f"Error while generating CSV: {e}"

    csv_data = forecast_sales_df.to_csv(index=False)
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = f"attachment; filename={product_name}_forecast.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    if "user" not in session:
        return redirect(url_for("login"))

    from_date = request.form['from_date']
    to_date = request.form['to_date']
    product_name = request.form['product']

    try:
        forecast_sales_df = run_forecast(product_name, from_date, to_date)
    except Exception as e:
        return f"Error while generating PDF: {e}"

    product_uses = MODEL_PATHS[product_name]['uses']
    explanation = generate_forecast_explanation(forecast_sales_df, product_name, product_uses)
    summary = get_summary(forecast_sales_df, product_name, product_uses)

    # Plot image for PDF (gracefully handle missing kaleido)
    fig = create_plot(forecast_sales_df, return_fig=True)
    img_bytes = BytesIO()
    chart_available = True
    try:
        pio.write_image(fig, img_bytes, format="png", width=800, height=500)
        img_bytes.seek(0)
    except Exception as e:
        print(f"Chart image generation failed (kaleido likely missing): {e}")
        chart_available = False

    # Recommendations
    recommendations = get_recommendations(forecast_sales_df)

    # Add Serial Number column (clean and consistent)
    tbl_df = forecast_sales_df.reset_index(drop=True)
    tbl_df.index = tbl_df.index + 1
    tbl_df.index.name = 'Serial Number'
    tbl_df = tbl_df.reset_index()

    # Build PDF
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph(f"{product_name} Forecast Report", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))

    if chart_available:
        elements.append(Paragraph("Forecasted Sales Chart:", styles['Heading2']))
        elements.append(Image(img_bytes, width=6*inch, height=4*inch))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Summary:", styles['Heading2']))
    elements.append(Paragraph(summary, styles['BodyText']))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Explanation:", styles['Heading2']))
    elements.append(Paragraph(explanation, styles['BodyText']))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Recommendations:", styles['Heading2']))
    elements.append(Paragraph(recommendations, styles['BodyText']))
    elements.append(Spacer(1, 0.2 * inch))

    table_data = [tbl_df.columns.to_list()] + tbl_df.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(table)
    elements.append(Spacer(1, 0.2 * inch))

    pdf.build(elements)
    buffer.seek(0)

    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={product_name}_forecast.pdf'
    return response

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)
