import asyncio
import os
from promptflow.core import tool
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.contents import ChatHistory
from math_plugin import MathPlugin

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

from lights import LightsPlugin
from promptflow.connections import CustomConnection
from search import SearchPlugin


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need

# In Python tool you can do things like calling external services or
# pre/post processing of data, pretty much anything you want

import logging

# Set the logging level for  semantic_kernel.kernel to DEBUG.
logging.basicConfig(
    format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("kernel").setLevel(logging.DEBUG)
logging.getLogger("echo").setLevel(logging.DEBUG)
logging.getLogger("search").setLevel(logging.INFO)


@tool
async def echo(userInput: str, con: CustomConnection) -> str:
    print("####### " + __name__)
    # Initialize the kernel6
    kernel = Kernel()

    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name="gpt-4.1",
        api_key=con.secrets["OPENAI_41_KEY"],
        endpoint=con.secrets["OPENAI_41__ENDPOINT"],
    )
    kernel.add_service(chat_completion)

    # Add a plugin (the LightsPlugin class is defined below)
    # kernel.add_plugin(
    #     LightsPlugin(),
    #     plugin_name="Lights",
    # )
    # kernel.add_plugin(MathPlugin(), plugin_name="MathPlugin")
    kernel.add_plugin(SearchPlugin(con), plugin_name="SearchPlugin")

    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Create a history of the conversation
    history = ChatHistory()
    history.add_system_message(
        "You are a helpful assistant. You can answer questions and execute tools."
    )

    # Initiate a back-and-forth chat
    # Add user input to the history
    history.add_user_message(userInput)

    # Get the response from the AI
    result = await chat_completion.get_chat_message_content(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel,
    )

    # Print the results
    print("Assistant > " + str(result))

    # Add the message from the agent to the chat history
    history.add_message(result)
    return str(result)


async def execute_tool(user_message: str) -> str:
    """
    This function is the entry point for the tool.
    It calls the echo function with the provided input.
    """
    load_dotenv()
    # Create a CustomConnection instance with the environment variables
    con = CustomConnection(
        secrets={
            "OPENAI_41_KEY": os.getenv("OPENAI_41_KEY"),
            "OPENAI_41__ENDPOINT": os.getenv("OPENAI_41__ENDPOINT"),
            "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
            "AZURE_SEARCH_API_KEY": os.getenv("AZURE_SEARCH_API_KEY"),
            "AZURE_SEARCH_INDEX_NAME": os.getenv(
                "AZURE_SEARCH_INDEX_NAME", "service-offerings"
            ),
        }
    )
    return await echo(user_message, con)
