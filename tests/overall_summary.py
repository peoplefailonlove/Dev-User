import asyncio
from generate_audience_characteristics import (
    _create_azure_client,
    generate_overall_summary_from_sample_data,
)

sample_data = {
    "data": [
        {"role": "BR", "summary": "BR respondent prioritizes automation..."},
        {"role": "pvt_vendor", "summary": "Private vendor values ERP integration..."},
    ]
}

async def test():
    client, _ = _create_azure_client()
    result = await generate_overall_summary_from_sample_data(client, sample_data)
    print(result)

asyncio.run(test())