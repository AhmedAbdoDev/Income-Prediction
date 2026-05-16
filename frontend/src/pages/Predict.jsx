import { useState } from "react";

const DEFAULTS = {
  age: 35,
  sex: "Male",
  education_num: 13,
  hours_per_week: 40,
  capital_gain: 0,
  capital_loss: 0,
  workclass: "Private",
  marital_status: "Never-married",
  occupation: "Prof-specialty",
  relationship: "Not-in-family",
};

const NUM_FIELDS = [
  "age",
  "education_num",
  "hours_per_week",
  "capital_gain",
  "capital_loss",
];

export default function Predict({ online }) {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handle = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: NUM_FIELDS.includes(name) ? Number(value) : value,
    }));
  };

  const predict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Connection failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const isHigh = result?.prediction === ">50K";

  return (
    <div>
      <div className="card">
        <div className="section-title">Personal Info</div>
        <div className="form-grid">
          <div className="form-group">
            <label>Age</label>
            <input
              type="number"
              name="age"
              value={form.age}
              onChange={handle}
              min={17}
              max={90}
            />
          </div>
          <div className="form-group">
            <label>Sex</label>
            <select name="sex" value={form.sex} onChange={handle}>
              <option>Male</option>
              <option>Female</option>
            </select>
          </div>
          <div className="form-group">
            <label>Education Years</label>
            <input
              type="number"
              name="education_num"
              value={form.education_num}
              onChange={handle}
              min={1}
              max={16}
            />
          </div>
          <div className="form-group">
            <label>Marital Status</label>
            <select
              name="marital_status"
              value={form.marital_status}
              onChange={handle}
            >
              {[
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-spouse-absent",
                "Separated",
                "Married-AF-spouse",
                "Widowed",
              ].map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Relationship</label>
            <select
              name="relationship"
              value={form.relationship}
              onChange={handle}
            >
              {[
                "Not-in-family",
                "Husband",
                "Wife",
                "Own-child",
                "Unmarried",
                "Other-relative",
              ].map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="section-title">Work</div>
        <div className="form-grid">
          <div className="form-group">
            <label>Workclass</label>
            <select name="workclass" value={form.workclass} onChange={handle}>
              {[
                "Private",
                "Self-emp-not-inc",
                "Self-emp-inc",
                "Federal-gov",
                "Local-gov",
                "State-gov",
                "Without-pay",
                "Never-worked",
              ].map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Occupation</label>
            <select name="occupation" value={form.occupation} onChange={handle}>
              {[
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
                "Prof-specialty",
                "Other-service",
                "Sales",
                "Craft-repair",
                "Transport-moving",
                "Farming-fishing",
                "Machine-op-inspct",
                "Tech-support",
                "Protective-serv",
                "Armed-Forces",
                "Priv-house-serv",
              ].map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Hours / Week</label>
            <input
              type="number"
              name="hours_per_week"
              value={form.hours_per_week}
              onChange={handle}
              min={1}
              max={99}
            />
          </div>
        </div>
      </div>

      <div className="card">
        <div className="section-title">Capital</div>
        <div className="form-grid">
          <div className="form-group">
            <label>Capital Gain</label>
            <input
              type="number"
              name="capital_gain"
              value={form.capital_gain}
              onChange={handle}
              min={0}
            />
          </div>
          <div className="form-group">
            <label>Capital Loss</label>
            <input
              type="number"
              name="capital_loss"
              value={form.capital_loss}
              onChange={handle}
              min={0}
            />
          </div>
        </div>

        <button
          className="predict-btn"
          onClick={predict}
          disabled={loading || !online}
        >
          {loading && <span className="loading" />}
          {loading ? "Running..." : "▶ Run Prediction"}
        </button>

        {error && <div className="error-msg">{error}</div>}

        {result && (
          <div className={`toast ${isHigh ? "" : "negative"}`}>
            <div>
              <div className="toast-label">Income Class</div>
              <div className="toast-result">{result.prediction}</div>
              <div className="toast-meta">
                Model: {result.model_name} · F1: {result.f1}%
              </div>
            </div>
            <div className="toast-right">
              <div className="toast-conf-label">Confidence</div>
              <div className="toast-conf-val">{result.confidence}%</div>
              <div className="toast-bar">
                <div
                  className="toast-bar-fill"
                  style={{ width: `${result.confidence}%` }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
