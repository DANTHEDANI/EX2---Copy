import os

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .sdk import TradingSDK
from .services.inference import InferenceService

app = FastAPI(title="Trading SDK GUI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sdk = TradingSDK()

# Simple global state to track training
task_state = {
    "training": {
        "status": "idle",
        "message": "",
        "progress": 0,
        "episodes": 0,
        "reward": 0.0,
        "loss": 0.0,
        "epsilon": 0.0,
    },
    "backtest": {"status": "idle", "message": ""},
}


class ActionRequest(BaseModel):
    ticker: str
    start_date: str | None = None
    end_date: str | None = None
    episodes: int | None = None


def progress_callback(ep, total_ep, reward, loss, epsilon):
    task_state["training"]["progress"] = ep
    task_state["training"]["episodes"] = total_ep
    task_state["training"]["reward"] = round(reward, 2)
    task_state["training"]["loss"] = round(loss, 4)
    task_state["training"]["epsilon"] = round(epsilon, 4)


def run_training_task(req: ActionRequest):
    task_state["training"]["status"] = "running"
    task_state["training"]["message"] = f"Training started for {req.ticker}..."
    try:
        # Override params if provided
        if req.start_date:
            sdk.config_manager.setup["data"]["start_date"] = req.start_date
        if req.end_date:
            sdk.config_manager.setup["data"]["end_date"] = req.end_date
        if req.episodes:
            sdk.config_manager.setup["hyperparameters"]["episodes"] = req.episodes

        sdk.run_training_pipeline(req.ticker, progress_callback=progress_callback)
        task_state["training"]["status"] = "completed"
        task_state["training"]["message"] = f"Training successfully completed for {req.ticker}."
    except Exception as e:
        task_state["training"]["status"] = "error"
        task_state["training"]["message"] = str(e)


def run_backtest_task(req: ActionRequest):
    task_state["backtest"]["status"] = "running"
    task_state["backtest"]["message"] = f"Backtesting started for {req.ticker}..."
    try:
        if req.start_date:
            sdk.config_manager.setup["data"]["start_date"] = req.start_date
        if req.end_date:
            sdk.config_manager.setup["data"]["end_date"] = req.end_date

        sdk.evaluate_strategy(req.ticker)
        task_state["backtest"]["status"] = "completed"
        task_state["backtest"]["message"] = f"Backtesting successfully completed for {req.ticker}."
    except Exception as e:
        task_state["backtest"]["status"] = "error"
        task_state["backtest"]["message"] = str(e)


@app.post("/api/train")
def start_training(req: ActionRequest, background_tasks: BackgroundTasks):
    if task_state["training"]["status"] == "running":
        return JSONResponse(
            {"status": "error", "message": "Training is already in progress!"}, status_code=400
        )
    background_tasks.add_task(run_training_task, req)
    return {"status": "started", "message": f"Training initiated for {req.ticker}"}


@app.get("/api/train/status")
def get_training_status():
    return task_state["training"]


@app.post("/api/backtest")
def start_backtest(req: ActionRequest, background_tasks: BackgroundTasks):
    if task_state["backtest"]["status"] == "running":
        return JSONResponse(
            {"status": "error", "message": "Backtest is already in progress!"}, status_code=400
        )
    background_tasks.add_task(run_backtest_task, req)
    return {"status": "started", "message": f"Backtest initiated for {req.ticker}"}


@app.get("/api/backtest/status")
def get_backtest_status():
    return task_state["backtest"]


@app.post("/api/data/chart")
def get_chart_data(req: ActionRequest):
    try:
        if req.start_date:
            sdk.config_manager.setup["data"]["start_date"] = req.start_date
        if req.end_date:
            sdk.config_manager.setup["data"]["end_date"] = req.end_date

        cfg = sdk.config_manager.setup["data"]
        raw_df = sdk.data_client.download_ticker(
            req.ticker, cfg["start_date"], cfg["end_date"], cfg["interval"]
        )

        if raw_df is None or raw_df.empty:
            return JSONResponse({"status": "error", "message": "No data found."}, status_code=400)

        # Prepare data for Plotly Candlestick
        df = raw_df.tail(100).copy()  # return last 100 days for chart

        dates = df.index.strftime("%Y-%m-%d").tolist()
        opens = df["Open"].tolist()
        highs = df["High"].tolist()
        lows = df["Low"].tolist()
        closes = df["Close"].tolist()

        return {"dates": dates, "open": opens, "high": highs, "low": lows, "close": closes}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/inference")
def get_inference(req: ActionRequest):
    try:
        cfg = sdk.config_manager.setup["data"]
        raw_df = sdk.data_client.download_ticker(
            req.ticker, cfg["start_date"], cfg["end_date"], cfg["interval"]
        )
        if raw_df is None or len(raw_df) < 30:
            return JSONResponse(
                {"status": "error", "message": "Not enough data for inference."}, status_code=400
            )

        states, _ = sdk._build_states_and_prices(raw_df)
        if len(states) == 0:
            return JSONResponse(
                {"status": "error", "message": "Could not engineer features."}, status_code=400
            )

        latest_state = states[-1]

        inf_service = InferenceService(sdk.config_manager)
        res = inf_service.predict_detailed(latest_state)

        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        action_str = action_map[res["action"]]

        # Build explanation
        rsi = latest_state[-1, 1] * 100  # Approx un-normalize
        macd_hist = latest_state[-1, 4]

        explanation = f"Model chose {action_str}."
        if action_str == "BUY":
            explanation += " Q-Value indicates profitable entry."
        elif action_str == "SELL":
            explanation += " Q-Value indicates profitable exit or risk."
        else:
            explanation += " Market is currently neutral."

        explanation += f" Features used: RSI(norm) ~{rsi:.1f}, MACD_Hist(norm) ~{macd_hist:.2f}."

        return {"action": action_str, "q_values": res["q_values"], "explanation": explanation}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# Resolve absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gui_dir = os.path.join(project_root, "gui")
results_dir = os.path.join(project_root, "data", "results")

os.makedirs(results_dir, exist_ok=True)
app.mount("/results", StaticFiles(directory=results_dir), name="results")
if os.path.exists(gui_dir):
    app.mount("/", StaticFiles(directory=gui_dir, html=True), name="gui")
