
# Survey JSON Extraction Service

A FastAPI service that downloads a document from Azure Blob Storage, extracts survey questions using Azure OpenAI, and uploads the resulting JSON back to a blob container.

## Setup

### Install dependencies:

pip install -r requirements.txt

## Environment Variables

AZURE_STORAGE_CONNECTION_STRING=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=gpt-4o
OPENAI_API_VERSION=2025-01-01-preview
LOG_LEVEL=INFO

## Run the **API**

### Start the server (development):

fastapi run app.py

Open the interactive **API** docs (Swagger):

[http://localhost:**8000**/docs](http://localhost:**8000**/docs)

### Example Request

**POST** /process

Request body (**JSON**):

{
    *input_blob_url*: "[https://`<account>`.blob.core.windows.net/samplefiles/survey.docx&#34;,](https://<account>.blob.core.windows.net/samplefiles/survey.docx*,)
    *output_container*: *json-output-container*,
    *output_blob_prefix*: *survey_extracted*
}

### Example Response

{
    *status*: *success*,
    *input_blob*: *[https://`<account>`.blob.core.windows.net/samplefiles/survey.docx&#34;,](https://<account>.blob.core.windows.net/samplefiles/survey.docx*,)
    *output_blob*: *[https://`<account>`.blob.core.windows.net/json-output-container/survey_extracted_20251119T073453Z_ab12cd34.json&#34;,](https://<account>.blob.core.windows.net/json-output-container/survey_extracted_20251119T073453Z_ab12cd34.json*,)
    *questions_extracted": **148**
}

## Quick test

**POST** [http://localhost:**8000**/process](http://localhost:**8000**/process) Content-Type: application/json

{
    *input_blob_url*: "[https://`<account>`.blob.core.windows.net/samplefiles/survey.docx&#34;,](https://<account>.blob.core.windows.net/samplefiles/survey.docx*,)
    *output_container*: *json-output-container*,
    *output_blob_prefix*: *survey_extracted"
}
