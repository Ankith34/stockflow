from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import re
from dotenv import load_dotenv
import json

load_dotenv()                    # must be first

from core.orchestrator import run_analysis
from core.database import init_db

load_dotenv()
init_db()

app = FastAPI()

class AnalysisRequest(BaseModel):
    ticker: str

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """Run the full analysis pipeline and stream the result."""
    
    def generate():
        try:
            # Yield status updates
            yield f"data: 🔍 Fetching financial data for **{request.ticker.upper()}**...\n\n"
            
            import time
            # Run the full pipeline (blocking, but we'll stream updates)
            memo = run_analysis(request.ticker)
            
            yield f"data: {json.dumps(memo)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: ❌ Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.get("/")
async def serve_frontend():
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)