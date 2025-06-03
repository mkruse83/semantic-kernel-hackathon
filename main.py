# execute the tool function from echo.py
from echo import execute_tool
import asyncio


def main():
    # Example input to the tool
    user_message = "What is the squareroot of 8*2?"

    # Execute the tool function
    result = asyncio.run(execute_tool(user_message=user_message))

    # Print the result
    print("Tool Output:", result)


if __name__ == "__main__":
    main()
