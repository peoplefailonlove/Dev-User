# app.py
import asyncio
import json
import os
import tempfile
import shutil
import logging
import time
from urllib.parse import urlparse, unquote
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import requests
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
    BlobSasPermissions,
)

# Import your parser (json_ext.py must be next to this app.py or in PYTHONPATH)
try:
    from json_ext import extract_document_to_json
except Exception as e:
    # Fail fast with helpful message if import fails
    raise ImportError(
        f"Failed to import extract_document_to_json from json_ext.py: {e}"
    )

try:
    from questionnaire_parser import extract_questions_labels, extract_sections
except Exception as e:
    # Fail fast with helpful message if import fails
    raise ImportError(
        f"Failed to import from questionnaire_parser.py: {e}"
    )

try:
    from generate_audience_characteristics import (
        _create_azure_client,
        generate_audience_characteristics,
    )
except Exception as e:
    raise ImportError(
        f"Failed to import from generate_audience_characteristics.py: {e}"
    )

try:
    from run_survey import run_survey_for_all_members
except Exception as e:
    raise ImportError(
        f"Failed to import run_survey_for_all_members from run_survey.py: {e}"
    )

try:
    from answer_mapper import create_excel_from_responses
except Exception as e:
    raise ImportError(
        f"Failed to import create_excel_from_responses from answer_mapper.py: {e}"
    )

# Vaibhav changed - import answer_distribution module so we can compute distributions
try:
    import answer_distribution  # Vaibhav changed
except Exception as e:
    raise ImportError(f"Failed to import answer_distribution.py: {e}")  # Vaibhav changed

# Load .env
load_dotenv()

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("blob-doc-parser")

# FastAPI App
app = FastAPI(
    title="Document Parser & Audience Generator API",
    description="API for document parsing and synthetic audience generation",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Trailing slash redirect middleware
class TrailingSlashMiddleware(BaseHTTPMiddleware):
    """Redirect requests without trailing slash to include one."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        # Skip if already has trailing slash, is root, or has file extension
        if not path.endswith("/") and "." not in path.split("/")[-1] and path != "/":
            return RedirectResponse(
                url=str(request.url).replace(path, path + "/", 1),
                status_code=307,  # Preserve HTTP method
            )
        return await call_next(request)


app.add_middleware(TrailingSlashMiddleware)

# Blob Container Names
AUDIENCE_OUTPUT_CONTAINER = "generated-synthetic-audience"
SURVEY_RESULTS_CONTAINER = "survey-answer"
FINAL_ANS_JSON_CONTAINER = "final-ans-json"
FINAL_ANS_EXCEL_CONTAINER = "final-ans-excel"
FINAL_ANS_DIST_JSON_CONTAINER = "final-ans-dist-json"   # Vaibhav changed

UPDATE_PERSONAS_API_URL = os.getenv(
    "UPDATE_PERSONAS_API_URL",
    "https://sample-agument-middleware-dev.azurewebsites.net/sample-enrichment/api/projects/update-persona-file-url",
)
UPDATE_SURVEY_API_URL = os.getenv(
    "UPDATE_SURVEY_API_URL",
    "https://sample-agument-middleware-dev.azurewebsites.net/sample-enrichment/api/projects/update-survey-file-url",
)

UPDATE_ANSWER_JSON_EXCEL_API_URL = os.getenv(
    "UPDATE_ANSWER_JSON_EXCEL_API_URL",
    "https://sample-agument-middleware-dev.azurewebsites.net/sample-enrichment/api/projects/update-excel-file-url",
)

# ============== Request/Response Models ==============


# Document Parser Request
class ProcessRequest(BaseModel):
    input_blob_url: str  # full blob URL to DOCX/PDF
    output_container: str = "samplefiles"  # where output JSON will be saved
    output_blob_prefix: str = (
        "output_file"  # prefix for output JSON (timestamp appended)
    )


# Audience Generation Request/Response
class GenerateAudienceRequest(BaseModel):
    """Request body for audience generation with inline JSON data."""

    input_data: dict  # Direct JSON data containing audiences
    output_blob_prefix: str = "audience_output"  # Prefix for output JSON
    max_concurrent: int = Field(default=10, ge=1, le=50)
    selected_sections: list[str] | None = Field(
        default=None,
        description="List of section names user selected to answer. If None, all sections are used."
    )
    quota_enabled: bool = Field(
        default=False,
        description="If True (quota='YES'), age and income are mandatory for response generation."
    )
    sample_survey_summary: str = Field(
        default="",
        description="Summarized sample survey dataset for context in generation."
    )


class GenerateAudienceResponse(BaseModel):
    """Response for audience generation (full result)."""

    task_id: str
    status: str
    output_blob: str  # SAS URL for output
    total_audiences: int
    total_members_processed: int
    total_successfully_generated: int
    total_failed: int
    processing_time_seconds: float


class GenerateAudienceAsyncResponse(BaseModel):
    """Immediate response for async audience generation."""

    task_id: str
    status: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


# Survey Simulation request/response models
class RunSurveyRequest(BaseModel):
    """Request body for survey simulation."""

    questionnaire_blob_url: str
    audience_blob_url: str
    output_blob_prefix: str = "survey_llm_results"
    task_id: str | None = None  # Task ID from generate_audience for correlation


class RunSurveyResponse(BaseModel):
    """Response body for survey simulation."""

    status: str
    output_blob: str
    total_members: int
    total_questions: int
    processing_time_seconds: float
    task_id: str | None = None  # Echoed back for correlation


# Helper: parse container + blob name from full blob URL
def parse_blob_url(blob_url: str):
    parsed = urlparse(blob_url)
    path = parsed.path.lstrip("/")  # samplefiles/input_file.docx

    if "/" not in path:
        raise ValueError("Invalid blob URL format. Expected /container/blobname")

    container, blob_name = path.split("/", 1)
    return unquote(container), unquote(blob_name)


# Helper: create timestamped output blob name (now includes short uuid for uniqueness)
def generate_output_blob_name(prefix: str):
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short_uuid = uuid4().hex[:8]
    safe_prefix = prefix.replace(" ", "_")
    return f"{safe_prefix}_{stamp}_{short_uuid}.json"


# Helper: parse AccountName and AccountKey from connection string
def parse_account_from_connection_string(conn_str: str):
    parts = {}
    for segment in conn_str.split(";"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.strip()] = v.strip()
    account_name = parts.get("AccountName")
    account_key = parts.get("AccountKey")
    return account_name, account_key


@app.post("/process")
@app.post("/process/", include_in_schema=False)
def process_file(req: ProcessRequest):
    logger.info(f"Received request for blob: {req.input_blob_url}")

    # Get connection string from env
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        return {
            "status": "Error",
            "message": "AZURE_STORAGE_CONNECTION_STRING not set",
        }

    # Parse account info (needed for SAS)
    account_name, account_key = parse_account_from_connection_string(conn_str)
    if not account_name or not account_key:
        logger.error(
            "AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING. "
            "SAS URL generation requires a connection string with AccountKey."
        )
        return {
            "status": "Error",
            "message": "AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING (cannot generate SAS).",
        }

    # Create BlobServiceClient
    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
    except Exception as e:
        logger.exception("Failed to create BlobServiceClient")
        return {
            "status": "Error",
            "message": f"Failed to create BlobServiceClient: {e}",
        }

    # Parse input blob URL
    try:
        container, blob_name = parse_blob_url(req.input_blob_url)
        logger.info(f"Parsed container={container}, blob={blob_name}")
    except Exception as e:
        return {
            "status": "Error",
            "message": f"Invalid input blob URL: {e}",
        }

    tmp_dir = None

    try:
        # Create a temp dir (unique per request)
        tmp_dir = tempfile.mkdtemp(prefix="doc_parse_")

        # ---------- Download input blob ----------
        input_uuid = uuid4().hex[:8]
        original_filename = Path(blob_name).name
        local_input_path = os.path.join(tmp_dir, f"{input_uuid}_{original_filename}")

        try:
            blob_client = blob_service.get_blob_client(
                container=container,
                blob=blob_name,
            )
            logger.info(f"Downloading blob to {local_input_path}")
            data = blob_client.download_blob().readall()
            with open(local_input_path, "wb") as f:
                f.write(data)
            logger.info("Download complete")
        except Exception as e:
            logger.exception("Blob download failed")
            return {
                "status": "Error",
                "message": f"Failed to download input blob: {e}",
            }

        # ---------- Run parser ----------
        output_blob_name = generate_output_blob_name(req.output_blob_prefix)
        local_output_path = os.path.join(tmp_dir, output_blob_name)

        try:
            logger.info("Running parser...")
            # New extractor returns dict grouped by category
            result = extract_document_to_json(
                file_path=local_input_path,
                output_path=local_output_path,
            )
            logger.info("Parsing completed")
            if not result:
                raise RuntimeError("Parser returned no result")
        except Exception as e:
            logger.exception("Parser failed")
            return {
                "status": "Error",
                "message": f"Parser failed: {e}",
            }

        # Ensure output file exists
        if not os.path.exists(local_output_path):
            return {
                "status": "Error",
                "message": "Parser did not create output file",
            }

        # ---------- Upload output JSON ----------
        try:
            output_blob_client = blob_service.get_blob_client(
                container=req.output_container,
                blob=output_blob_name,
            )
            logger.info(
                f"Uploading output JSON to container={req.output_container}, "
                f"blob={output_blob_name}"
            )
            with open(local_output_path, "rb") as f:
                output_blob_client.upload_blob(
                    f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="application/json"),
                )
            logger.info("Upload complete")
        except Exception as e:
            logger.exception("Output upload failed")
            return {
                "status": "Error",
                "message": f"Failed to upload output JSON: {e}",
            }

        # ---------- Generate SAS URL for the output blob (NO FALLBACK) ----------
        try:
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=req.output_container,
                blob_name=output_blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=365),  # 1 year validity
            )

            if not sas_token:
                raise RuntimeError("generate_blob_sas returned empty token")

            output_url = f"{output_blob_client.url}?{sas_token}"
            logger.info("Generated SAS URL for output blob")
        except Exception as e:
            logger.exception("Failed to generate SAS URL")
            return {
                "status": "Error",
                "message": f"Failed to generate SAS URL: {e}",
            }

        # ---------- Post-process: question labels & counts ----------
        # result is expected to be: { "Screener": [...], "Main Survey": [...], ... }
        if isinstance(result, dict):
            questions_extracted = sum(
                len(v) for v in result.values() if isinstance(v, list)
            )
        else:
            questions_extracted = None

        questions_labels = (
            extract_questions_labels(result) if isinstance(result, dict) else []
        )
        
        # Extract sections for quota selection screen
        sections = (
            extract_sections(result) if isinstance(result, dict) else []
        )

        # ---------- SUCCESS RESPONSE ----------
        return {
            "status": "success",
            "input_blob": req.input_blob_url,
            "output_blob": output_url,  # ALWAYS SAS URL on success
            "questions_extracted": questions_extracted,
            "questions_labels": questions_labels,
            "sections": sections,  # All sections for user selection in quota screen
        }

    except Exception as e:
        logger.exception("Unexpected error during processing")
        return {
            "status": "Error",
            "message": f"Unexpected error: {e}",
        }

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Temporary folder cleaned up")


# ============== Survey Simulation Helpers ==============


def _run_simulate_survey_internal(
    questionnaire_blob_url: str,
    audience_blob_url: str,
    output_blob_prefix: str,
    task_id: str,
    conn_str: str,
    account_name: str,
    account_key: str,
) -> dict:
    """
    Internal function to run survey simulation. Can be called from background tasks.
    """
    logger.info(
        f"_run_simulate_survey_internal: questionnaire={questionnaire_blob_url}, "
        f"audience={audience_blob_url}, task_id={task_id}"
    )

    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
    except Exception as e:
        logger.exception("Failed to create BlobServiceClient")
        raise RuntimeError(f"Failed to create BlobServiceClient: {e}")

    # Parse input blob URLs
    try:
        q_container, q_blob_name = parse_blob_url(questionnaire_blob_url)
        a_container, a_blob_name = parse_blob_url(audience_blob_url)
    except Exception as e:
        raise ValueError(f"Invalid input blob URL: {e}")

    tmp_dir = tempfile.mkdtemp(prefix="survey_run_internal_")
    try:
        # ---------- Download questionnaire JSON ----------
        questionnaire_local = os.path.join(tmp_dir, "questionnaire.json")
        try:
            q_blob_client = blob_service.get_blob_client(
                container=q_container,
                blob=q_blob_name,
            )
            data = q_blob_client.download_blob().readall()
            with open(questionnaire_local, "wb") as f:
                f.write(data)
        except Exception as e:
            logger.exception("Failed to download questionnaire blob")
            raise RuntimeError(f"Failed to download questionnaire blob: {e}")

        # ---------- Download audience/persona JSON ----------
        audience_local = os.path.join(tmp_dir, "audience.json")
        try:
            a_blob_client = blob_service.get_blob_client(
                container=a_container,
                blob=a_blob_name,
            )
            data = a_blob_client.download_blob().readall()
            with open(audience_local, "wb") as f:
                f.write(data)
        except Exception as e:
            logger.exception("Failed to download audience blob")
            raise RuntimeError(f"Failed to download audience blob: {e}")

        # ---------- Run survey simulation ----------
        start_time = time.time()
        # Vaibhav changed: Wrap in try-except to handle partial failures gracefully
        try:
            results = run_survey_for_all_members(
                questionnaire_path=questionnaire_local,
                project_path=audience_local,
            )
        except Exception as survey_err:
            # Vaibhav changed: Log the error but check if we can still proceed
            logger.error(f"Task {task_id}: run_survey_for_all_members raised exception: {survey_err}")
            logger.exception("Full traceback for survey simulation error")
            # Re-raise to be caught by outer handler, but log that we're failing
            raise RuntimeError(f"Survey simulation failed: {survey_err}") from survey_err
        
        processing_seconds = time.time() - start_time

        # Vaibhav changed: Filter out any None or invalid results before proceeding
        if not results:
            raise RuntimeError("Survey simulation returned no results")
        
        # Filter out None results (in case batch returned partial failures)
        valid_results = [r for r in results if r is not None and isinstance(r, dict) and "answers" in r]
        if not valid_results:
            raise RuntimeError("Survey simulation returned no valid results")
        
        # Use valid results, log if some were filtered
        if len(valid_results) < len(results):
            logger.warning(
                f"Task {task_id}: Filtered out {len(results) - len(valid_results)} invalid results "
                f"from survey simulation"
            )
        results = valid_results

        total_members = len(results)
        total_questions = len(results[0]["answers"]) if results else 0

        # ---------- Save results locally ----------
        output_blob_name = generate_output_blob_name(output_blob_prefix)
        local_output_path = os.path.join(tmp_dir, output_blob_name)
        with open(local_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # ---------- Upload results to survey-answer container ----------
        try:
            output_blob_client = blob_service.get_blob_client(
                container=SURVEY_RESULTS_CONTAINER,
                blob=output_blob_name,
            )
            with open(local_output_path, "rb") as f:
                output_blob_client.upload_blob(
                    f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="application/json"),
                )
        except Exception as e:
            logger.exception("Failed to upload survey results JSON")
            raise RuntimeError(f"Failed to upload survey results JSON: {e}")

        # ---------- Generate SAS URL ----------
        try:
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=SURVEY_RESULTS_CONTAINER,
                blob_name=output_blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=365),
            )
            if not sas_token:
                raise RuntimeError("generate_blob_sas returned empty token")

            output_url = f"{output_blob_client.url}?{sas_token}"
        except Exception as e:
            logger.exception("Failed to generate SAS URL for survey results")
            raise RuntimeError(f"Failed to generate SAS URL: {e}")

        return {
            "status": "success",
            "output_blob": output_url,
            "total_members": total_members,
            "total_questions": total_questions,
            "processing_time_seconds": processing_seconds,
            "task_id": task_id,
        }

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Temporary folder for survey_run_internal cleaned up")


# ============== Survey Simulation Endpoint ==============


@app.post("/simulate-survey", response_model=RunSurveyResponse, tags=["Survey"])
@app.post("/simulate-survey/", include_in_schema=False)
def simulate_survey(req: RunSurveyRequest) -> RunSurveyResponse:
    """
    Run LLM-based survey simulation for all personas in the given audience JSON.
    """
    logger.info(
        f"simulate-survey: questionnaire_blob_url={req.questionnaire_blob_url}, "
        f"audience_blob_url={req.audience_blob_url}, "
        f"task_id={req.task_id}"
    )

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise HTTPException(
            status_code=500,
            detail="AZURE_STORAGE_CONNECTION_STRING not set",
        )

    account_name, account_key = parse_account_from_connection_string(conn_str)
    if not account_name or not account_key:
        logger.error(
            "AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING. "
            "SAS URL generation requires a connection string with AccountKey."
        )
        raise HTTPException(
            status_code=500,
            detail="AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING (cannot generate SAS).",
        )

    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
    except Exception as e:
        logger.exception("Failed to create BlobServiceClient")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create BlobServiceClient: {e}",
        )

    # Parse input blob URLs
    try:
        q_container, q_blob_name = parse_blob_url(req.questionnaire_blob_url)
        a_container, a_blob_name = parse_blob_url(req.audience_blob_url)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input blob URL: {e}",
        )

    tmp_dir = tempfile.mkdtemp(prefix="survey_run_")
    try:
        # ---------- Download questionnaire JSON ----------
        questionnaire_local = os.path.join(tmp_dir, "questionnaire.json")
        try:
            q_blob_client = blob_service.get_blob_client(
                container=q_container,
                blob=q_blob_name,
            )
            data = q_blob_client.download_blob().readall()
            with open(questionnaire_local, "wb") as f:
                f.write(data)
        except Exception as e:
            logger.exception("Failed to download questionnaire blob")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download questionnaire blob: {e}",
            )

        # ---------- Download audience/persona JSON ----------
        audience_local = os.path.join(tmp_dir, "audience.json")
        try:
            a_blob_client = blob_service.get_blob_client(
                container=a_container,
                blob=a_blob_name,
            )
            data = a_blob_client.download_blob().readall()
            with open(audience_local, "wb") as f:
                f.write(data)
        except Exception as e:
            logger.exception("Failed to download audience blob")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download audience blob: {e}",
            )

        # ---------- Run survey simulation ----------
        start_time = time.time()
        results = run_survey_for_all_members(
            questionnaire_path=questionnaire_local,
            project_path=audience_local,
            batch_size=30,
        )
        processing_seconds = time.time() - start_time

        total_members = len(results)
        total_questions = len(results[0]["answers"]) if results else 0

        # ---------- Save results locally ----------
        output_blob_name = generate_output_blob_name(req.output_blob_prefix)
        local_output_path = os.path.join(tmp_dir, output_blob_name)
        with open(local_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # ---------- Upload results to survey-answer container ----------
        try:
            output_blob_client = blob_service.get_blob_client(
                container=SURVEY_RESULTS_CONTAINER,
                blob=output_blob_name,
            )
            with open(local_output_path, "rb") as f:
                output_blob_client.upload_blob(
                    f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="application/json"),
                )
        except Exception as e:
            logger.exception("Failed to upload survey results JSON")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload survey results JSON: {e}",
            )

        # ---------- Generate SAS URL ----------
        try:
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=SURVEY_RESULTS_CONTAINER,
                blob_name=output_blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=365),
            )
            if not sas_token:
                raise RuntimeError("generate_blob_sas returned empty token")

            output_url = f"{output_blob_client.url}?{sas_token}"
        except Exception as e:
            logger.exception("Failed to generate SAS URL for survey results")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate SAS URL: {e}",
            )

        return RunSurveyResponse(
            status="success",
            output_blob=output_url,
            total_members=total_members,
            total_questions=total_questions,
            processing_time_seconds=processing_seconds,
            task_id=req.task_id,
        )

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Temporary folder for survey_run cleaned up")


# ============== Health Check Endpoint ==============


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get(
    "/health/", include_in_schema=False
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="document-parser-audience-generator",
        version="1.0.0",
    )


# ============== Audience Generation Helpers ==============


def call_update_survey_api(
    project_id: int, task_id: str, survey_file_url: str, status_text: str = "Survey File Generated"
) -> bool:
    """
    Calls the update-survey-file-url API after survey is completed and uploaded to blob.
    Returns True on success, False on failure.
    """
    if not UPDATE_SURVEY_API_URL:
        logger.warning("UPDATE_SURVEY_API_URL not configured, skipping survey update API call")
        return False

    payload = {
        "projectId": project_id,
        "taskId": task_id,
        "surveyFileUrl": survey_file_url,
        "status": status_text,
    }

    try:
        logger.info(
            f"Calling update-survey-file-url API → project {project_id}, task {task_id}"
        )

        response = requests.post(
            UPDATE_SURVEY_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()
        api_status = data.get("status", "").lower()

        if api_status == "success":
            logger.info(
                f"Survey file URL update SUCCESS for project {project_id}, task {task_id}"
            )
            return True
        else:
            logger.error(
                f"Survey file URL update FAILED for project {project_id}, task {task_id} | "
                f"API response: {data}"
            )
            return False

    except requests.exceptions.RequestException as e:
        logger.error(
            f"HTTP error calling update-survey-file-url API (project {project_id}, task {task_id}): {e}"
        )
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response body: {e.response.text}")
        return False
    except (ValueError, KeyError, TypeError) as e:
        logger.error(
            f"Invalid JSON response from update-survey-file-url API (project {project_id}, task {task_id}): {e}"
        )
        logger.error(f"Raw response: {response.text if 'response' in locals() else 'N/A'}")
        return False


def call_update_personas_api(project_id: int, sas_url: str) -> bool:
    """
    Calls the update-personas API and returns True only if JSON response has "status": "success".
    Returns True on success, False on failure.
    """
    payload = {"projectId": project_id, "sasUrl": sas_url}

    try:
        logger.info(f"Calling update-personas API → project {project_id}")

        response = requests.post(
            UPDATE_PERSONAS_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        # 1. Check HTTP status first
        response.raise_for_status()

        # 2. Parse JSON body
        data = response.json()

        # 3. Check the actual business status
        api_status = data.get("status", "").lower()

        if api_status == "success":
            logger.info(f"Persona update SUCCESS for project {project_id}")
            return True
        else:
            logger.error(
                f"Persona update FAILED for project {project_id} | "
                f"API response: {data}"
            )
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error calling update-personas API (project {project_id}): {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response body: {e.response.text}")
        return False
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Invalid JSON response from update-personas API (project {project_id}): {e}")
        logger.error(f"Raw response: {response.text if 'response' in locals() else 'N/A'}")
        return False


def call_update_answer_json_excel_api(
    project_id: int,
    task_id: str,
    survey_file_json_url: str,
    survey_output_json_url: str,
    excel_file_url: str,
    status_text: str = "Complete",
) -> bool:
    """
    Calls the update-excel-file-url API after answer mapping is completed and
    the Excel + JSON files are uploaded to blob.

    Returns True on success, False on failure.
    """
    if not UPDATE_ANSWER_JSON_EXCEL_API_URL:
        logger.warning(
            "UPDATE_ANSWER_JSON_EXCEL_API_URL not configured, "
            "skipping answer JSON+Excel update API call"
        )
        return False

    payload = {
        "projectId": project_id,
        "taskId": task_id,
        "surveyFileJsonUrl": survey_file_json_url,
        "surveyOutputJsonUrl": survey_output_json_url,
        "excelFileUrl": excel_file_url,
        "status": status_text,
    }

    try:
        logger.info(
            "Calling update-excel-file-url API → project %s, task %s",
            project_id,
            task_id,
        )

        response = requests.post(
            UPDATE_ANSWER_JSON_EXCEL_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()
        api_status = data.get("status", "").lower()

        if api_status == "success":
            logger.info(
                "Answer JSON+Excel URL update SUCCESS for project %s, task %s",
                project_id,
                task_id,
            )
            return True
        else:
            logger.error(
                "Answer JSON+Excel URL update FAILED for project %s, task %s | "
                "API response: %s",
                project_id,
                task_id,
                data,
            )
            return False

    except requests.exceptions.RequestException as e:
        logger.error(
            "HTTP error calling update-excel-file-url API "
            "(project %s, task %s): %s",
            project_id,
            task_id,
            e,
        )
        if hasattr(e, "response") and e.response is not None:
            logger.error("Response body: %s", e.response.text)
        return False
    except (ValueError, KeyError, TypeError) as e:
        logger.error(
            "Invalid JSON response from update-excel-file-url API "
            "(project %s, task %s): %s",
            project_id,
            task_id,
            e,
        )
        logger.error(
            "Raw response: %s",
            response.text if "response" in locals() else "N/A",
        )
        return False


def process_audience_generation_background(
    task_id: str,
    req: GenerateAudienceRequest,
    conn_str: str,
    account_name: str,
    account_key: str,
) -> None:
    """
    Background task to process audience generation and call update personas API.

    Args:
        task_id: Unique task identifier
        req: The original request
        conn_str: Azure storage connection string
        account_name: Azure storage account name
        account_key: Azure storage account key
    """
    start_time = time.time()
    input_data = req.input_data
    tmp_dir = None

    try:
        # Create BlobServiceClient
        blob_service = BlobServiceClient.from_connection_string(conn_str)

        # Create a temp dir (unique per request)
        tmp_dir = tempfile.mkdtemp(prefix="audience_gen_")

        # Initialize Azure OpenAI client
        try:
            client, deployment = _create_azure_client()
        except ValueError as e:
            logger.error(f"Task {task_id}: Azure OpenAI not configured: {e}")
            return

        # Get audiences from input data
        audiences = input_data.get("audiences", [])
        if not audiences:
            logger.error(f"Task {task_id}: No audiences found in input data")
            return

        # Run generation
        normalized_audiences = [
            {
                "persona": aud.get("persona", {}),
                "screenerQuestions": aud.get("screenerQuestions", []),
                "sampleSize": aud.get("sampleSize", 1),
                "quotas": aud.get("quotas", {}),
            }
            for aud in audiences
        ]

        async def run_generation():
            tasks = [
                generate_audience_characteristics(
                    client=client,
                    deployment=deployment,
                    audience_data=aud,
                    audience_index=idx,
                    max_concurrent=req.max_concurrent,
                    selected_sections=req.selected_sections,
                    quota_enabled=req.quota_enabled,
                    sample_survey_summary=req.sample_survey_summary,
                )
                for idx, aud in enumerate(normalized_audiences)
            ]
            return await asyncio.gather(*tasks)

        enriched_audiences = asyncio.run(run_generation())

        total_generated = sum(
            aud["metadata"]["generation_stats"]["successfully_generated"]
            for aud in enriched_audiences
        )
        total_failed = sum(
            aud["metadata"]["generation_stats"]["failed"] for aud in enriched_audiences
        )

        processing_time = time.time() - start_time

        # Build output response
        project_id = input_data.get("projectId")
        user_id = input_data.get("userId")
        json_file_url = input_data.get("json_file_url")
        output_data = {
            "project_name": input_data.get("projectName"),
            "project_description": input_data.get("projectDescription"),
            "project_id": str(project_id) if project_id is not None else None,
            "user_id": str(user_id) if user_id is not None else None,
            "request_id": input_data.get("requestId"),
            "json_file_url": json_file_url,
            "generation_model": f"azure:{deployment}",
            "provider": "azure",
            "total_audiences": len(enriched_audiences),
            "total_members_processed": total_generated + total_failed,
            "total_successfully_generated": total_generated,
            "total_failed": total_failed,
            "processing_time_seconds": round(processing_time, 2),
            "selected_sections": req.selected_sections,
            "quota_enabled": req.quota_enabled,
            "audiences": enriched_audiences,
        }

        # Write output to temp file
        output_blob_name = generate_output_blob_name(req.output_blob_prefix)
        local_output_path = os.path.join(tmp_dir, output_blob_name)

        with open(local_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Task {task_id}: Generation completed, output written to temp file"
        )

        # Ensure output container exists
        try:
            container_client = blob_service.get_container_client(
                AUDIENCE_OUTPUT_CONTAINER
            )
            try:
                container_client.get_container_properties()
            except Exception:
                container_client.create_container()
                logger.info(f"Created container '{AUDIENCE_OUTPUT_CONTAINER}'")
        except Exception as e:
            logger.error(f"Task {task_id}: Failed to ensure container exists: {e}")
            return

        # Upload output JSON
        try:
            output_blob_client = blob_service.get_blob_client(
                container=AUDIENCE_OUTPUT_CONTAINER,
                blob=output_blob_name,
            )
            with open(local_output_path, "rb") as f:
                output_blob_client.upload_blob(
                    f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="application/json"),
                )
            logger.info(f"Task {task_id}: Upload complete")
        except Exception as e:
            logger.error(f"Task {task_id}: Output upload failed: {e}")
            return

        # Generate SAS URL for the output blob
        try:
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=AUDIENCE_OUTPUT_CONTAINER,
                blob_name=output_blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=365),
            )

            if not sas_token:
                raise RuntimeError("generate_blob_sas returned empty token")

            output_url = f"{output_blob_client.url}?{sas_token}"
            logger.info(f"Task {task_id}: Generated SAS URL for output blob")
        except Exception as e:
            logger.error(f"Task {task_id}: Failed to generate SAS URL: {e}")
            return

        # Call update personas API if projectId is provided
        update_success = False
        if project_id is not None:
            update_success = call_update_personas_api(int(project_id), output_url)
        else:
            logger.warning(
                f"Task {task_id}: No projectId provided, skipping update personas API call"
            )

        # Chain simulate-survey after successful update_personas_api call
        if update_success and json_file_url:
            logger.info(
                f"Task {task_id}: update_personas_api succeeded, chaining simulate-survey"
            )
            try:
                survey_result = _run_simulate_survey_internal(
                    questionnaire_blob_url=json_file_url,
                    audience_blob_url=output_url,
                    output_blob_prefix="survey_results",
                    task_id=task_id,
                    conn_str=conn_str,
                    account_name=account_name,
                    account_key=account_key,
                )
                survey_output_url = survey_result.get("output_blob")
                logger.info(
                    f"Task {task_id}: simulate-survey completed successfully\n"
                    f"  survey_output_blob: {survey_output_url}\n"
                    f"  total_members: {survey_result.get('total_members')}\n"
                    f"  total_questions: {survey_result.get('total_questions')}"
                )

                # 1) Call update-survey-file-url API AFTER survey is complete
                if survey_output_url and project_id is not None:
                    survey_update_success = call_update_survey_api(
                        project_id=int(project_id),
                        task_id=task_id,
                        survey_file_url=survey_output_url,
                        status_text="Survey File Generated",
                    )
                    if survey_update_success:
                        logger.info(
                            f"Task {task_id}: update-survey-file-url API call succeeded"
                        )
                    else:
                        logger.warning(
                            f"Task {task_id}: update-survey-file-url API call failed"
                        )
                else:
                    logger.warning(
                        f"Task {task_id}: survey_output_url or project_id missing, "
                        f"skipping update-survey-file-url API call"
                    )

                if survey_output_url and json_file_url and project_id is not None:
                    logger.info(
                        f"Task {task_id}: Starting answer_mapper to generate Excel + JSON"
                    )

                    try:
                        # 2.a) Download questionnaire JSON locally for answer_mapper
                        q_container, q_blob_name = parse_blob_url(json_file_url)
                        questionnaire_local = os.path.join(tmp_dir, "questionnaire_for_mapper.json")
                        try:
                            q_blob_client = blob_service.get_blob_client(
                                container=q_container,
                                blob=q_blob_name,
                            )
                            q_data = q_blob_client.download_blob().readall()
                            with open(questionnaire_local, "wb") as f:
                                f.write(q_data)
                            logger.info(
                                f"Task {task_id}: Downloaded questionnaire JSON for answer_mapper "
                                f"to {questionnaire_local}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Task {task_id}: Failed to download questionnaire JSON for "
                                f"answer_mapper: {e}"
                            )
                            raise

                        s_container, s_blob_name = parse_blob_url(survey_output_url)
                        survey_local = os.path.join(tmp_dir, "survey_results_for_mapper.json")
                        try:
                            s_blob_client = blob_service.get_blob_client(
                                container=s_container,
                                blob=s_blob_name,
                            )
                            s_data = s_blob_client.download_blob().readall()
                            with open(survey_local, "wb") as f:
                                f.write(s_data)
                            logger.info(
                                f"Task {task_id}: Downloaded survey results JSON for answer_mapper "
                                f"to {survey_local}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Task {task_id}: Failed to download survey results JSON for "
                                f"answer_mapper: {e}"
                            )
                            raise

                        excel_output_path = os.path.join(tmp_dir, f"{task_id}_answer_mapper.xlsx")

                        mapper_result = create_excel_from_responses(
                            questionnaire_path=questionnaire_local,
                            persona_responses_path=survey_local,
                            output_path=excel_output_path,
                        )

                        local_excel_path = mapper_result.get("excel_path", excel_output_path)
                        local_json_path = mapper_result.get("json_path")

                        logger.info(
                            f"Task {task_id}: answer_mapper completed. "
                            f"Excel: {local_excel_path}, JSON: {local_json_path}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Task {task_id}: answer_mapper failed to generate Excel/JSON: {e}"
                        )
                        local_excel_path = None
                        local_json_path = None

                    excel_url = None
                    survey_output_json_url = None
                    mapped_json_sas_url = None  # Vaibhav changed: Initialize mapped_json_sas_url for answer_mapper JSON

                    # ---------- START: upload mapped Excel/JSON and compute distribution ----------
                    if local_excel_path and local_json_path:
                        try:
                            # Create simple blob names based on task_id
                            excel_blob_name = f"{task_id}_final_answers.xlsx"
                            json_blob_name = f"{task_id}_final_answers.json"

                            # Ensure final-ans-excel container exists
                            try:
                                excel_container_client = blob_service.get_container_client(
                                    FINAL_ANS_EXCEL_CONTAINER
                                )
                                try:
                                    excel_container_client.get_container_properties()
                                except Exception:
                                    excel_container_client.create_container()
                                    logger.info(
                                        f"Created container '{FINAL_ANS_EXCEL_CONTAINER}'"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Task {task_id}: Failed to ensure "
                                    f"container '{FINAL_ANS_EXCEL_CONTAINER}' exists: {e}"
                                )
                                raise

                            # Ensure final-ans-json container exists
                            try:
                                json_container_client = blob_service.get_container_client(
                                    FINAL_ANS_JSON_CONTAINER
                                )
                                try:
                                    json_container_client.get_container_properties()
                                except Exception:
                                    json_container_client.create_container()
                                    logger.info(
                                        f"Created container '{FINAL_ANS_JSON_CONTAINER}'"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Task {task_id}: Failed to ensure "
                                    f"container '{FINAL_ANS_JSON_CONTAINER}' exists: {e}"
                                )
                                raise

                            # Upload Excel to final-ans-excel
                            excel_blob_client = blob_service.get_blob_client(
                                container=FINAL_ANS_EXCEL_CONTAINER,
                                blob=excel_blob_name,
                            )
                            with open(local_excel_path, "rb") as f:
                                excel_blob_client.upload_blob(
                                    f,
                                    overwrite=True,
                                    content_settings=ContentSettings(
                                        content_type=(
                                            "application/vnd.openxmlformats-"
                                            "officedocument.spreadsheetml.sheet"
                                        )
                                    ),
                                )

                            # Upload JSON to final-ans-json
                            json_blob_client = blob_service.get_blob_client(
                                container=FINAL_ANS_JSON_CONTAINER,
                                blob=json_blob_name,
                            )
                            with open(local_json_path, "rb") as f:
                                json_blob_client.upload_blob(
                                    f,
                                    overwrite=True,
                                    content_settings=ContentSettings(
                                        content_type="application/json"
                                    ),
                                )

                            logger.info(
                                f"Task {task_id}: Uploaded mapped Excel/JSON to "
                                f"'{FINAL_ANS_EXCEL_CONTAINER}' and '{FINAL_ANS_JSON_CONTAINER}'"
                            )

                            # Generate SAS URLs for excel and mapped JSON
                            excel_sas = generate_blob_sas(
                                account_name=account_name,
                                container_name=FINAL_ANS_EXCEL_CONTAINER,
                                blob_name=excel_blob_name,
                                account_key=account_key,
                                permission=BlobSasPermissions(read=True),
                                expiry=datetime.utcnow() + timedelta(days=365),
                            )
                            json_sas = generate_blob_sas(
                                account_name=account_name,
                                container_name=FINAL_ANS_JSON_CONTAINER,
                                blob_name=json_blob_name,
                                account_key=account_key,
                                permission=BlobSasPermissions(read=True),
                                expiry=datetime.utcnow() + timedelta(days=365),
                            )

                            if not excel_sas or not json_sas:
                                raise RuntimeError(
                                    "generate_blob_sas returned empty token for "
                                    "Excel and/or JSON"
                                )

                            excel_url = f"{excel_blob_client.url}?{excel_sas}"
                            mapped_json_sas_url = f"{json_blob_client.url}?{json_sas}"

                            logger.info(
                                f"Task {task_id}: Generated SAS URLs for mapped "
                                f"Excel/JSON"
                            )

                            # ---------------------------
                            # Compute distribution from mapped JSON (local_json_path)
                            # and upload distribution JSON to FINAL_ANS_DIST_JSON_CONTAINER
                            # ---------------------------
                            try:
                                # Local distribution file path
                                local_dist_path = os.path.join(tmp_dir, f"{task_id}_final_answers_distribution.json")

                                # Use answer_distribution.save_distribution_to_file which reads the input JSON and writes distribution
                                dist_output_file = answer_distribution.save_distribution_to_file(
                                    input_file=local_json_path,
                                    output_file=local_dist_path,
                                )

                                # Ensure FINAL_ANS_DIST_JSON_CONTAINER exists
                                try:
                                    dist_container_client = blob_service.get_container_client(
                                        FINAL_ANS_DIST_JSON_CONTAINER
                                    )
                                    try:
                                        dist_container_client.get_container_properties()
                                    except Exception:
                                        dist_container_client.create_container()
                                        logger.info(f"Created container '{FINAL_ANS_DIST_JSON_CONTAINER}'")
                                except Exception as e:
                                    logger.error(
                                        f"Task {task_id}: Failed to ensure container '{FINAL_ANS_DIST_JSON_CONTAINER}' exists: {e}"
                                    )
                                    raise

                                # Upload distribution JSON to FINAL_ANS_DIST_JSON_CONTAINER
                                dist_blob_name = f"{task_id}_final_answers_distribution.json"
                                dist_blob_client = blob_service.get_blob_client(
                                    container=FINAL_ANS_DIST_JSON_CONTAINER,
                                    blob=dist_blob_name,
                                )
                                with open(local_dist_path, "rb") as f:
                                    dist_blob_client.upload_blob(
                                        f,
                                        overwrite=True,
                                        content_settings=ContentSettings(content_type="application/json"),
                                    )

                                # Generate SAS for distribution JSON
                                dist_sas = generate_blob_sas(
                                    account_name=account_name,
                                    container_name=FINAL_ANS_DIST_JSON_CONTAINER,
                                    blob_name=dist_blob_name,
                                    account_key=account_key,
                                    permission=BlobSasPermissions(read=True),
                                    expiry=datetime.utcnow() + timedelta(days=365),
                                )
                                if not dist_sas:
                                    raise RuntimeError("generate_blob_sas returned empty token for distribution JSON")

                                dist_sas_url = f"{dist_blob_client.url}?{dist_sas}"

                                logger.info(
                                    f"Task {task_id}: Uploaded distribution JSON and generated SAS URL "
                                    f"(container={FINAL_ANS_DIST_JSON_CONTAINER}, blob={dist_blob_name})"
                                )

                                # According to requirement: set survey_output_json_url to the distribution JSON SAS link
                                survey_output_json_url = dist_sas_url  # Vaibhav changed

                            except Exception as e:
                                # Vaibhav changed: Add detailed error logging with traceback
                                logger.exception(f"Task {task_id}: Failed to compute/upload distribution JSON: {e}")
                                # If distribution calculation fails, fallback to mapped JSON SAS url
                                survey_output_json_url = mapped_json_sas_url
                                logger.warning(f"Task {task_id}: Falling back to mapped JSON SAS URL for survey_output_json_url")

                        except Exception as e:
                            logger.error(
                                f"Task {task_id}: Failed to upload mapped Excel/JSON "
                                f"or generate SAS URLs: {e}"
                            )
                    else:
                        logger.warning(
                            f"Task {task_id}: Missing local Excel/JSON from answer_mapper; skipping upload + distribution"
                        )
                    # ---------- END: upload mapped Excel/JSON and compute distribution ----------

                    # 3) Call update-excel-file-url API with all URLs
                    if (
                        excel_url
                        and survey_output_json_url
                        and mapped_json_sas_url  # Vaibhav changed: Check for mapped_json_sas_url instead of survey_output_url
                        and project_id is not None
                    ):
                        try:
                            # Mapping 
                            # surveyFileJsonUrl    -> mapped_json_sas_url (answer_mapper JSON) - Vaibhav changed
                            # surveyOutputJsonUrl  -> survey_output_json_url (distribution JSON) - Vaibhav changed
                            # excelFileUrl         -> excel_url (answer_mapper Excel)
                            answer_update_success = call_update_answer_json_excel_api(
                                project_id=int(project_id),
                                task_id=task_id,
                                survey_file_json_url=mapped_json_sas_url,  # Vaibhav changed: Use answer_mapper JSON instead of survey_output_url
                                survey_output_json_url=survey_output_json_url,  # Vaibhav changed: This is the distribution JSON
                                excel_file_url=excel_url,
                                status_text="Complete",
                            )
                            if answer_update_success:
                                logger.info(
                                    f"Task {task_id}: update-excel-file-url API "
                                    f"call succeeded"
                                )
                            else:
                                logger.warning(
                                    f"Task {task_id}: update-excel-file-url API "
                                    f"call failed"
                                )
                        except Exception as e:
                            logger.error(
                                f"Task {task_id}: Exception while calling "
                                f"update-excel-file-url API: {e}"
                            )
                    else:
                        logger.warning(
                            f"Task {task_id}: Missing URLs for Excel/JSON or "
                            f"survey_output_url/project_id; skipping "
                            f"update-excel-file-url API call"
                        )

                else:
                    logger.warning(
                        f"Task {task_id}: Missing survey_output_url/json_file_url/"
                        f"project_id, skipping answer_mapper + update-excel-file-url"
                    )

            except Exception as survey_err:
                logger.error(
                    f"Task {task_id}: simulate-survey failed: {survey_err}"
                )
        elif update_success and not json_file_url:
            logger.warning(
                f"Task {task_id}: update_personas_api succeeded but no json_file_url provided, "
                "skipping simulate-survey"
            )
        elif not update_success:
            logger.warning(
                f"Task {task_id}: update_personas_api failed, skipping simulate-survey"
            )

        # Print completion summary
        logger.info(
            f"\n{'='*60}\n"
            f"TASK COMPLETED SUCCESSFULLY\n"
            f"{'='*60}\n"
            f"task_id: {task_id}\n"
            f"output_blob: {output_url}\n"
            f"json_file_url: {json_file_url}\n"
            f"project_id: {project_id}\n"
            f"{'='*60}"
        )

    except Exception as e:
        logger.exception(f"Task {task_id}: Unexpected error during processing: {e}")

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"Task {task_id}: Temporary folder cleaned up")


# ============== Audience Generation Endpoint ==============


@app.post(
    "/generate_audience/",
    response_model=GenerateAudienceAsyncResponse,
    tags=["Audience Generation"],
    summary="Generate audience characteristics from JSON data (async)",
    description="Accepts JSON data, returns task_id immediately, processes in background, and calls update-personas API when complete.",
)
@app.post(
    "/generate_audience",
    response_model=GenerateAudienceAsyncResponse,
    tags=["Audience Generation"],
    include_in_schema=False,  # Hide duplicate from docs
)
async def generate_audience(
    req: GenerateAudienceRequest,
    background_tasks: BackgroundTasks,
) -> GenerateAudienceAsyncResponse:
    """
    Generate audience characteristics from inline JSON data (async).

    - Accepts JSON data directly in request body (must include projectId and userId)
    - Returns task_id and status immediately
    - Processes audiences in background using Azure OpenAI
    - Uploads output to 'generated-synthetic-audience' container
    - Calls update-personas API with projectId and sasUrl when complete
    """
    task_id = str(uuid4())
    logger.info(f"Received audience generation request, task_id: {task_id}")

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise HTTPException(
            status_code=500,
            detail="AZURE_STORAGE_CONNECTION_STRING not set",
        )

    account_name, account_key = parse_account_from_connection_string(conn_str)
    if not account_name or not account_key:
        logger.error(
            "AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING. "
            "SAS URL generation requires a connection string with AccountKey."
        )
        raise HTTPException(
            status_code=500,
            detail="AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING (cannot generate SAS).",
        )

    # Start background task
    background_tasks.add_task(
        process_audience_generation_background,
        task_id,
        req,
        conn_str,
        account_name,
        account_key,
    )

    return GenerateAudienceAsyncResponse(task_id=task_id, status="started")
