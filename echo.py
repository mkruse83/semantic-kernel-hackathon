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
from semantic_kernel.agents import Agent, ChatCompletionAgent
from semantic_kernel.agents import GroupChatOrchestration, RoundRobinGroupChatManager
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.agents.runtime import InProcessRuntime
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

logger = logging.getLogger(__name__)


def agent_response_callback(message: ChatMessageContent) -> None:
    logger.debug(f"+++++{message.name}+++++\n{message.content}")


@tool
async def echo(userInput: str, con: CustomConnection) -> str:
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

    salesAgent = ChatCompletionAgent(
        name="JuniorSalesAgent",
        description="An agent that can answer questions and execute tools.",
        instructions="You are a sales agent for PRODYNA SE. You can answer questions about our service offerings and execute tools to provide information.",
        kernel=kernel,
    )

    seniorSalesAgent = ChatCompletionAgent(
        name="SeniorSalesAgent",
        description="A senior sales agent that refines the best offers.",
        instructions="You are a senior sales agent for PRODYNA SE. Your junior staff provides offers to you based on customer requests. You review these offers and produce follow-up questions for the customer to come to a single fitting offer. e.g. if you have two offers about Cloud platforms (Strategy offering and Data offering), then ask the customer if he is more interested in Strategy or data. If there is only one offer left, then provide this offer to the customer.",
        service=AzureChatCompletion(
            deployment_name="gpt-4.1",
            api_key=con.secrets["OPENAI_41_KEY"],
            endpoint=con.secrets["OPENAI_41__ENDPOINT"],
        ),
    )

    # customerCommunicationAgent = ChatCompletionAgent(
    #     name="CustomerCommunicationAgent",
    #     description="An agent that communicates with the customer.",
    #     instructions="You are a customer communication agent for PRODYNA SE. You manage the communication between the sales department and the customer. You may refer the customer request back to the Junior Sales Agent, based on the Senior Sales Agent feedback.",
    #     service=AzureChatCompletion(
    #         deployment_name="gpt-4.1",
    #         api_key=con.secrets["OPENAI_41_KEY"],
    #         endpoint=con.secrets["OPENAI_41__ENDPOINT"],
    #     ),
    # )
    group_chat_orchestration = GroupChatOrchestration(
        members=[
            salesAgent,
            seniorSalesAgent,
            # customerCommunicationAgent,
        ],
        manager=RoundRobinGroupChatManager(max_rounds=9),
        agent_response_callback=agent_response_callback,
    )
    runtime = InProcessRuntime()
    runtime.start()
    orchestration_result = await group_chat_orchestration.invoke(
        task=userInput, runtime=runtime
    )
    result = await orchestration_result.get()
    # # Enable planning
    # execution_settings = AzureChatPromptExecutionSettings()
    # execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # # Create a history of the conversation
    # history = ChatHistory()
    # history.add_system_message(
    #     "You are a helpful assistant. You can answer questions and execute tools."
    # )

    # # Initiate a back-and-forth chat
    # # Add user input to the history
    # history.add_user_message(userInput)

    # # Get the response from the AI
    # result = await chat_completion.get_chat_message_content(
    #     chat_history=history,
    #     settings=execution_settings,
    #     kernel=kernel,
    # )

    # # Print the results
    # print("Assistant > " + str(result))

    # # Add the message from the agent to the chat history
    # history.add_message(result)
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
