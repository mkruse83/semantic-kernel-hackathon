# execute the tool function from echo.py
from echo import execute_tool
import asyncio
import logging

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting the tool execution...")

    # Example input to the tool
    user_message = "What service offerings are available for the topic 'AI'?"

    # Execute the tool function
    asyncio.run(execute_tool(user_message=user_message))


if __name__ == "__main__":
    main()
