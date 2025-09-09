from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.security.api_key import APIKeyHeader
from data_optimizer import process_data_pipeline, load_results
from utils import rate_limiter

API_KEY = "secret123"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(title="Data Optimizer Microservice")

def validate_api_key(api_key: str = Depends(api_key_header)):
    # APIKeyHeader.auto_error=False means api_key may be None
    if api_key is None or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/optimize")
async def optimize(request: Request):
    """
    Trigger the full pipeline. No API key required by default (public trigger).
    If you want to protect /optimize, add Depends(validate_api_key) here.
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    return process_data_pipeline(data)

@app.get("/retrieve")
@rate_limiter(max_calls=5, period=60)
async def retrieve(api_key: str = Depends(validate_api_key)):
    """
    Retrieve refined results. Protected by API key + rate limiter.
    Rate limiter returns a proper HTTPException on limit exceed.
    """
    try:
        return load_results()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Refined results not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
