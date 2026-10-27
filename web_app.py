"""Browser version: python web_app.py - opens http://127.0.0.1:8000

She uploads one image at a time; it is saved to inbox/ and the same agent (agent.py) handles it,
showing its steps live and asking its questions in the page. The dashboard reads the same data/
files as the terminal version, so both versions share one record of transactions.
Runs only on this computer (127.0.0.1): there are no users or passwords.
"""

import asyncio
import json
import re
import sys
import threading
import webbrowser
from datetime import date as calendar_date

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import main
import storage
import tools
from ui.web import WebUI

HOST, PORT = "127.0.0.1", 8000
WEB_DIR = config.BASE_DIR / "web"
# Months shown in the trend chart (the selected month and the ones before it)
TREND_MONTHS = 6
# Characters Windows does not allow in file names
_BAD_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

app = FastAPI(title="Receipt agent")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
# One agent run at a time: two runs would handle the same inbox files
_run_lock = threading.Lock()


@app.exception_handler(storage.StorageError)
async def _storage_error(request: Request, error: storage.StorageError) -> JSONResponse:
    """A damaged data file: show the message (it says how to restore the backup)."""
    return JSONResponse({"error": str(error)}, status_code=500)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def _months_back(month: str, count: int) -> list[str]:
    """`count` months ending with `month` ("YYYY-MM"), oldest first."""
    year, number = map(int, month.split("-"))
    months = []
    for back in range(count - 1, -1, -1):
        y, m = divmod(year * 12 + number - 1 - back, 12)
        months.append(f"{y}-{m + 1:02d}")
    return months


@app.get("/api/dashboard")
def dashboard(month: str | None = None) -> dict:
    """Everything the dashboard shows for one month. All numbers come from tools.py, as in the terminal."""
    month = month or calendar_date.today().strftime("%Y-%m")
    summary = tools.get_monthly_summary(month)
    if not summary["ok"]:
        raise HTTPException(400, summary["error"])

    transactions = storage.load_transactions()
    data_months = {t["date"][:7] for t in transactions if isinstance(t.get("date"), str)}
    months = data_months | {calendar_date.today().strftime("%Y-%m")}
    in_month = [dict(t, has_image=(config.INBOX_DIR / t.get("source_file", "")).is_file())
                for t in transactions if str(t.get("date", "")).startswith(month + "-")]
    in_month.sort(key=lambda t: (t["date"], t.get("id", "")), reverse=True)

    trend = []
    for m in _months_back(month, TREND_MONTHS):
        s = tools.get_monthly_summary(m)
        # A month that cannot be summarized (a test recorded the year 1000; six months back is 999)
        # is shown as empty instead of failing the whole dashboard
        trend.append({"month": m, "income": s.get("income_total", 0), "expense": s.get("expense_total", 0)})

    return {
        "summary": summary,
        "transactions": in_month,
        "months": sorted(months, reverse=True),
        "latest_month": max(data_months, default=None),  # the page opens here if this month is empty
        "trend": trend,
        "categories": config.EXPENSE_CATEGORIES,
        "income_categories": config.INCOME_CATEGORIES,
        "inbox_pending": tools.list_documents().get("new_count", 0),
    }


def _free_name(name: str) -> str:
    """A safe file name for inbox/ that does not overwrite an existing file."""
    name = _BAD_NAME_CHARS.sub("_", name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]).strip(" .") or "image"
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    candidate, n = name, 2
    while (config.INBOX_DIR / candidate).exists():
        candidate, n = f"{stem}-{n}.{ext}", n + 1
    return candidate


@app.post("/api/upload")
async def upload(request: Request, name: str) -> dict:
    """Save one image (the request body is the file itself) to inbox/."""
    extension = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if extension not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"אפשר להעלות רק תמונות או PDF ({', '.join(sorted(config.SUPPORTED_EXTENSIONS))})")
    body = await request.body()
    if not body:
        raise HTTPException(400, "הקובץ ריק")
    if len(body) > config.MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(400, f"הקובץ גדול מ-{config.MAX_IMAGE_MB} MB")
    config.INBOX_DIR.mkdir(exist_ok=True)
    file = _free_name(name)
    (config.INBOX_DIR / file).write_bytes(body)
    return {"file": file}


@app.get("/api/files/{name}")
def inbox_file(name: str) -> FileResponse:
    """The original image of a transaction (only files directly inside inbox/)."""
    path = config.INBOX_DIR / name
    if path.name != name or not path.is_file():
        raise HTTPException(404, "הקובץ לא נמצא")
    return FileResponse(path)


@app.patch("/api/transactions/{transaction_id}")
async def correct(transaction_id: str, request: Request) -> dict:
    """Her correction of a business name / category / date / amount; it also corrects what the agent remembers."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "בקשה לא תקינה")
    if not isinstance(body, dict):
        raise HTTPException(400, "בקשה לא תקינה")
    # The agent writes the same files during a run
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(409, "האג'נט עובד כרגע. אפשר לתקן כשיסיים.")
    try:
        result = tools.correct_transaction(transaction_id, body.get("business"), body.get("category"),
                                           body.get("date"), body.get("amount"))
    finally:
        _run_lock.release()
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result


@app.websocket("/ws/run")
async def run_agent(ws: WebSocket) -> None:
    """Run the agent once; stream its events to the page and pass her answers back to it."""
    await ws.accept()
    if not _run_lock.acquire(blocking=False):
        await ws.send_json({"type": "error", "text": "האג'נט כבר עובד בחלון אחר. אפשר לנסות שוב כשיסיים."})
        await ws.send_json({"type": "done"})
        await ws.close()
        return

    loop = asyncio.get_running_loop()
    events: asyncio.Queue[dict] = asyncio.Queue()
    ui = WebUI(lambda event: loop.call_soon_threadsafe(events.put_nowait, event))

    def work() -> None:
        try:
            main.run_with(ui)
        except Exception as e:  # a bug must not leave the page waiting forever
            ui.error(f"שגיאה לא צפויה: {e}")
        finally:
            loop.call_soon_threadsafe(events.put_nowait, {"type": "done"})

    async def receive_answers() -> None:
        try:
            while True:
                try:
                    message = json.loads(await ws.receive_text())
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict) and message.get("type") == "answer":
                    ui.answer(str(message.get("text", "")))
        except WebSocketDisconnect:
            ui.disconnect()

    # A daemon thread: a run waiting for an answer must not block stopping the server (Ctrl+C)
    threading.Thread(target=work, daemon=True).start()
    receiver = asyncio.create_task(receive_answers())
    connected = True
    try:
        while True:  # until "done" - also after the tab was closed, so the next run waits for this one
            event = await events.get()
            if connected:
                try:
                    await ws.send_json(event)
                except (WebSocketDisconnect, RuntimeError):
                    connected = False
                    ui.disconnect()
            if event["type"] == "done":
                break
    finally:
        receiver.cancel()
        _run_lock.release()
    if connected:
        await ws.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # Windows: server log lines may contain Hebrew
        sys.stdout.reconfigure(encoding="utf-8")
    url = f"http://{HOST}:{PORT}"
    print(f"פותח את {url} (לעצירה: Ctrl+C)")
    threading.Timer(1.5, webbrowser.open, args=[url]).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
