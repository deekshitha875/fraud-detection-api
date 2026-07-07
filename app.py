from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('fraud_model.pkl')

FEATURES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
            'V28', 'Amount']

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection API</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 40px 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { font-size: 2rem; color: #7c3aed; margin-bottom: 8px; }
        p.subtitle { color: #94a3b8; margin-bottom: 32px; }
        .card { background: #1e293b; border-radius: 12px; padding: 28px; margin-bottom: 24px; }
        h2 { font-size: 1.1rem; color: #a78bfa; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        label { display: block; font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px; }
        input { width: 100%; padding: 8px 10px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; color: #e2e8f0; font-size: 0.9rem; }
        input:focus { outline: none; border-color: #7c3aed; }
        .btn { width: 100%; padding: 14px; background: #7c3aed; color: white; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; margin-top: 20px; }
        .btn:hover { background: #6d28d9; }
        .result { display: none; padding: 20px; border-radius: 8px; text-align: center; font-size: 1.2rem; font-weight: 600; margin-top: 20px; }
        .fraud { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
        .legit { background: #052e16; color: #4ade80; border: 1px solid #166534; }
        .hint { font-size: 0.78rem; color: #64748b; margin-top: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Fraud Detection API</h1>
        <p class="subtitle">Enter transaction features to predict whether it's fraudulent.</p>

        <div class="card">
            <h2>Transaction Features</h2>
            <div class="grid" id="featureGrid"></div>
            <button class="btn" onclick="predict()">Run Prediction</button>
            <div class="result" id="result"></div>
            <p class="hint">💡 Leave all fields as 0 for a sample prediction. V1–V28 are PCA-transformed features.</p>
        </div>

        <div class="card">
            <h2>API Usage</h2>
            <code style="font-size:0.82rem; color:#94a3b8; line-height:1.8;">
                POST /predict<br>
                Content-Type: application/json<br><br>
                { "Time": 0, "V1": -1.36, ..., "Amount": 149.62 }
            </code>
        </div>
    </div>

    <script>
        const features = {{ features | tojson }};

        const grid = document.getElementById('featureGrid');
        features.forEach(f => {
            const div = document.createElement('div');
            div.innerHTML = `<label>${f}</label><input type="number" step="any" id="${f}" value="0">`;
            grid.appendChild(div);
        });

        async function predict() {
            const payload = {};
            features.forEach(f => {
                payload[f] = parseFloat(document.getElementById(f).value) || 0;
            });

            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = 'result';
            resultDiv.textContent = 'Analyzing...';

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.fraud) {
                    resultDiv.className = 'result fraud';
                    resultDiv.textContent = '🚨 FRAUDULENT Transaction Detected';
                } else {
                    resultDiv.className = 'result legit';
                    resultDiv.textContent = '✅ Legitimate Transaction';
                }
            } catch (e) {
                resultDiv.textContent = 'Error: ' + e.message;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE, features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    result = bool(pred == -1)
    return jsonify({'fraud': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
