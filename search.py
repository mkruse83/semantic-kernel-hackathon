# Copyright (c) Microsoft. All rights reserved.
# <defineClass>
import math
from typing import Annotated

from semantic_kernel.functions.kernel_function_decorator import kernel_function
import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from promptflow.connections import CustomConnection

import logging

logger = logging.getLogger(__name__)


class SearchPlugin:
    client: SearchClient

    def __init__(self, con: CustomConnection):
        endpoint = con.secrets["AZURE_SEARCH_ENDPOINT"]
        api_key = con.secrets["AZURE_SEARCH_API_KEY"]
        index_name = con.secrets["AZURE_SEARCH_INDEX_NAME"]
        if not endpoint or not api_key:
            raise ValueError(
                "Azure Search endpoint or API key not set in environment variables."
            )

        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key),
        )

    @kernel_function(
        description="Search for service offerings of PRODYNA SE.",
        name="SearchServiceOfferings",
    )
    def search(
        self,
        query: Annotated[
            str, "A natural language query to search for service offerings."
        ],
    ) -> Annotated[float, "The output is a float"]:
        # use Azure AI Search to search for service offerings
        logger.debug(f"Searching for query: {query}")
        results = self.client.search(
            search_text=query,
            top=5,
            query_type="semantic",
            semantic_configuration_name="default",
            include_total_count=True,
        )
        if not results:
            logger.warning("No results found.")
            return []
        logger.info(f"Found total {results.get_count()} results.")
        results = list(results)
        logger.debug(f"Results: {results}")

        # For demonstration, return the number of results as a float
        return results
