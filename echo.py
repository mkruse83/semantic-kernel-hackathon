import os
import logging

from dotenv import load_dotenv
from promptflow.core import tool
from promptflow.connections import CustomConnection

from pydantic import BaseModel, Field
from typing import Union, Optional

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents import (
    ChatCompletionAgent,
    GroupChatOrchestration,
    RoundRobinGroupChatManager,
    GroupChatManager,
    BooleanResult,
    StringResult,
    MessageResult,
)
from semantic_kernel.agents.runtime import InProcessRuntime

from search import SearchPlugin
import json

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


class JuniorSalesResult(BaseModel):
    search_results: Optional[str] = Field(
        description="The raw search results from the Azure AI Search service.",
    )
    number_search_results: int = Field(
        description="The number of raw search results from the Azure AI Search service.",
    )


class SeniorSalesResult(BaseModel):
    junior_sales_query: Optional[str] = Field(
        description="A new query for the junior sales agent to refine the search results based on the senior sales agent's review. None if no further request is needed.",
    )
    questions_to_user: Optional[str] = Field(
        description="Questions to the user to refine the best offer based on the junior sales agent's results. None if no further questions are needed.",
    )
    final_offer: Optional[str] = Field(
        description="The final offer for the customer based on the junior sales agent's results and the questions answered by the user. None if there are still multiple offers to choose from.",
    )


class CustomGroupChatManager(GroupChatManager):
    async def filter_results(self, chat_history: ChatHistory) -> MessageResult:
        content = chat_history.messages[-1].content
        if isinstance(content, str):
            content = json.loads(content)
        senior_res = SeniorSalesResult.model_validate(content)
        return MessageResult(
            result=ChatMessageContent(
                role="assistant", content=senior_res.questions_to_user
            ),
            reason="Adding questions to user based on Senior Sales Agent's review.",
        )

    async def select_next_agent(
        self, chat_history: ChatHistory, participant_descriptions: dict[str, str]
    ) -> StringResult:
        if len(chat_history.messages) == 0:
            # If no messages, return the first agent
            res = StringResult(
                result="JuniorSalesAgent", reason="Starting with Junior Sales Agent."
            )
        elif chat_history.messages[-1].name == "SeniorSalesAgent":
            # If the last message was from the Senior Sales Agent, switch to Junior Sales Agent
            res = StringResult(
                result="JuniorSalesAgent",
                reason="Switching to Junior Sales Agent after Senior Sales Agent.",
            )
        else:
            # If the last message was from Junior Sales Agent, switch to Senior Sales Agent
            res = StringResult(
                result="SeniorSalesAgent",
                reason="Switching to Senior Sales Agent after Junior Sales Agent.",
            )
        return res

    async def should_request_user_input(
        self, chat_history: ChatHistory
    ) -> BooleanResult:
        res = None
        if len(chat_history.messages) == 0:
            res = BooleanResult(result=False, reason="Agent need to perform a task.")
        elif chat_history.messages[-1].name != "SeniorSalesAgent":
            res = BooleanResult(
                result=False, reason="Only Senior Agent can delegate to user."
            )
        else:
            content = chat_history.messages[-1].content
            if isinstance(content, str):
                content = json.loads(content)
            senior_res = SeniorSalesResult.model_validate(content)
            senior_res.questions_to_user = senior_res.questions_to_user or "None"
            if senior_res.questions_to_user != "None":
                res = BooleanResult(
                    result=True,
                    reason="Senior Sales Agent needs to ask the user for more information.",
                )
            else:
                res = BooleanResult(
                    result=False,
                    reason="Senior Sales Agent does not need to ask the user for more information.",
                )
        return res

    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        # Optionally call the base implementation to check for default termination logic
        base_result = await super().should_terminate(chat_history)
        if base_result.result:
            return base_result
        # Custom logic to determine if the chat should terminate
        should_end = len(chat_history.messages) > 10  # Example: end after 10 messages
        return BooleanResult(result=should_end, reason="Custom termination logic.")


@tool
async def echo(userInput: str, con: CustomConnection) -> str:
    junior_settings = AzureChatPromptExecutionSettings()
    junior_settings.response_format = JuniorSalesResult
    senior_settings = AzureChatPromptExecutionSettings()
    senior_settings.response_format = SeniorSalesResult

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
        arguments=KernelArguments(settings=junior_settings),
    )

    seniorSalesAgent = ChatCompletionAgent(
        name="SeniorSalesAgent",
        description="A senior sales agent that refines the best offers.",
        instructions="""
You are a senior sales agent for PRODYNA SE.
Your junior staff provided offers from the PRODYNA portfolio based on customer requests.
Your task is to review these offers and see if there is a single best fit for the user request. You can do either of the following:
- provide a new query for the junior sales agent to refine the search results based on your review
- ask the user questions to refine the best offer based on the junior sales agent's results
- provide a final offer for the customer based on the junior sales agent's results and the questions answered by the user.
Provide None for the not applicable fields.
Remember your task is to refine the offers until only one offer is left. Ask the user questions until only one offer is left.
        """,
        service=AzureChatCompletion(
            deployment_name="gpt-4.1",
            api_key=con.secrets["OPENAI_41_KEY"],
            endpoint=con.secrets["OPENAI_41__ENDPOINT"],
        ),
        arguments=KernelArguments(settings=senior_settings),
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
        manager=CustomGroupChatManager(),
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
    # execution_settings.response_format = {
    #     "type": "text",
    #     "text": {
    #         "max_tokens": 1000,
    #         "stop_sequences": ["\n"],
    #     },
    # }

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
    print("Assistant > " + str(result))

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
