"""
server/app.py - FastAPI server for ICU Bed Allocation Environment
"""

import os
import sys

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.responses import JSONResponse, HTMLResponse
from openenv.core.env_server import create_fastapi_app

from models import ICUAction, ICUObservation
from server.upd_env import ICUEnvironment


# ---------------- AUTO ENV ROUTES ----------------
# creates /reset /step /state /health /docs
app = create_fastapi_app(ICUEnvironment, ICUAction, ICUObservation)


# ---------------- ROOT ----------------

@app.get("/")
def root():
    return JSONResponse(content={
        "name":        "ICU Bed Allocation Environment",
        "version":     "1.0.0",
        "status":      "running",
        "description": "RL environment for ICU triage optimization",
        "endpoints": {
            "health":   "/health",
            "docs":     "/docs",
            "tasks":    "/tasks",
            "grader":   "/grader",
            "baseline": "/baseline",
            "web":      "/web",
        }
    })


# ---------------- WEB UI ----------------

@app.get("/web", tags=["UI"], response_class=HTMLResponse)
def web_ui():
        html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <title>ICU Env Web UI</title>
    <style>
        body { font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 960px; }
        h1 { margin-bottom: 0.25rem; }
        h2 { margin-top: 1.5rem; }
        section { border: 1px solid #e5e7eb; padding: 0.75rem 1rem; border-radius: 0.5rem; margin-top: 0.75rem; }
        label { display: inline-block; min-width: 120px; }
        input { padding: 0.2rem 0.4rem; }
        button { margin-top: 0.5rem; padding: 0.25rem 0.75rem; cursor: pointer; }
        pre { background: #111827; color: #e5e7eb; padding: 0.75rem; border-radius: 0.5rem; max-height: 320px; overflow: auto; }
        code { background: #f3f4f6; padding: 0 0.2rem; border-radius: 0.25rem; }
    </style>
</head>
<body>
    <h1>ICU Bed Allocation Environment</h1>
    <p>Interactive web UI to exercise all HTTP endpoints of the ICU environment.</p>

    <section>
        <h2>Core environment endpoints</h2>
        <div>
            <button onclick=\"callHealth()\">GET /health</button>
            <button onclick=\"callReset()\">POST /reset</button>
            <br />
            <label>Action value (for /step):</label>
            <input id=\"step-value\" type=\"number\" min=\"0\" value=\"0\" />
            <button onclick=\"callStep()\">POST /step</button>
            <button onclick=\"callState()\">GET /state</button>
        </div>
    </section>

    <section>
        <h2>Competition endpoints</h2>
        <div>
            <button onclick=\"callTasks()\">GET /tasks</button>
            <br />
            <label>Grader steps:</label>
            <input id=\"grader-steps\" type=\"number\" min=\"1\" value=\"100\" />
            <button onclick=\"callGrader()\">POST /grader?steps=...</button>
            <br />
            <button onclick=\"callBaseline()\">GET /baseline</button>
        </div>
    </section>

    <section>
        <h2>Docs</h2>
        <p>
            OpenAPI docs are available at <code>/docs</code>.
            <button onclick=\"openDocs()\">Open /docs in new tab</button>
        </p>
    </section>

    <h2>Last response</h2>
    <pre id=\"output\">(no requests yet)</pre>

    <script>
        function showResponse(label, res, bodyText) {
            document.getElementById('output').textContent =
                label + '\nStatus: ' + res.status + '\n\n' + bodyText;
        }

        async function callHealth() {
            const res = await fetch('/health');
            const txt = await res.text();
            showResponse('GET /health', res, txt);
        }

        async function callReset() {
            const res = await fetch('/reset', { method: 'POST' });
            const txt = await res.text();
            showResponse('POST /reset', res, txt);
        }

        async function callStep() {
            const valueInput = document.getElementById('step-value');
            const value = parseInt(valueInput.value || '0', 10);
            const res = await fetch('/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ value: value })
            });
            const txt = await res.text();
            showResponse('POST /step {"value": ' + value + '}', res, txt);
        }

        async function callState() {
            const res = await fetch('/state');
            const txt = await res.text();
            showResponse('GET /state', res, txt);
        }

        async function callTasks() {
            const res = await fetch('/tasks');
            const txt = await res.text();
            showResponse('GET /tasks', res, txt);
        }

        async function callGrader() {
            const stepsInput = document.getElementById('grader-steps');
            const steps = parseInt(stepsInput.value || '100', 10);
            const res = await fetch('/grader?steps=' + encodeURIComponent(steps), {
                method: 'POST'
            });
            const txt = await res.text();
            showResponse('POST /grader?steps=' + steps, res, txt);
        }

        async function callBaseline() {
            const res = await fetch('/baseline');
            const txt = await res.text();
            showResponse('GET /baseline', res, txt);
        }

        function openDocs() {
            window.open('/docs', '_blank');
        }
    </script>
</body>
</html>"""
        return HTMLResponse(content=html)


# ---------------- TASKS ----------------

@app.get("/tasks", tags=["Competition"])
def get_tasks():
    return JSONResponse(content={
        "tasks": [
            {
                "task_id": "easy",
                "description": "Low patient load, basic prioritization",
                "difficulty": "easy",
            },
            {
                "task_id": "medium",
                "description": "Mixed severity, moderate load",
                "difficulty": "medium",
            },
            {
                "task_id": "hard",
                "description": "High load, critical prioritization required",
                "difficulty": "hard",
            },
        ],
        "total": 3,
        "action_schema": {
            "action": "int (0 = wait, 1..k = admit patient index)"
        },
    })


# ---------------- GRADER ----------------

@app.post("/grader", tags=["Competition"])
def run_grader(steps: int = 100):
    """
    Runs a random agent for evaluation
    (since ICU env is sequential, not single-shot like SQL)
    """

    import random

    env = ICUEnvironment()
    obs = env.reset()

    total_reward = 0.0

    for _ in range(steps):
        if obs.done:
            break

        # random baseline
        if obs.patients and obs.available_beds > 0:
            action = random.randint(1, len(obs.patients))
        else:
            action = 0

        obs = env.step(ICUAction(value=action))
        total_reward += obs.reward

    score = obs.metadata.get("score", 0.0)

    return JSONResponse(content={
        "score": score,
        "total_reward": total_reward,
        "steps": obs.time_step,
        "task_level": obs.metadata.get("task_level"),
        "feedback": obs.metadata.get("feedback"),
        "hint": obs.metadata.get("hint"),
    })


# ---------------- BASELINE ----------------

@app.get("/baseline", tags=["Competition"])
def run_baseline():
    """
    Greedy baseline: always admit highest severity
    """

    env = ICUEnvironment()
    obs = env.reset()

    total_reward = 0.0

    while not obs.done:
        if obs.available_beds > 0 and obs.patients:
            idx = max(range(len(obs.patients)),
                      key=lambda i: obs.patients[i].severity)
            action = idx + 1
        else:
            action = 0

        obs = env.step(ICUAction(value=action))
        total_reward += obs.reward

    score = obs.metadata.get("score", 0.0)

    return JSONResponse(content={
        "baseline_agent": "greedy (highest severity first)",
        "score": score,
        "total_reward": total_reward,
        "steps": obs.time_step,
        "task_level": obs.metadata.get("task_level"),
        "feedback": obs.metadata.get("feedback"),
    })


# ---------------- RUN SERVER ----------------

def main():
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run("server.app:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    main()