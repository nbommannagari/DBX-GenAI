from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from databricks.sdk import WorkspaceClient
import json
import mlflow
import os

# Point the OpenAI Agents SDK at Databricks
set_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))
set_default_openai_api("chat_completions")

mlflow.set_tracking_uri("databricks")

# Read environment variables injected by the DAB
SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME")
experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")

if experiment_id:
    mlflow.set_experiment(experiment_id=experiment_id)
elif experiment_name:
    mlflow.set_experiment(experiment_name)

# MCP configuration
catalog_name = os.getenv("CATALOG_NAME")
function_name = os.getenv("FUNCTION_1_NAME")

ws = WorkspaceClient()

mcp_server = McpServer.from_uc_function(
    catalog=catalog_name,
    schema="agent_deployment",
    workspace_client=ws,
    timeout=60.0,
)

# Define the agent with MCP tool access
agent = Agent(
    name="Simple Chatbot",
    instructions="You are a helpful assistant. Answer the user's questions clearly and concisely.",
    model=SERVING_ENDPOINT_NAME,
    mcp_servers=[mcp_server],
)


@mlflow.trace(
    name="tool_calling_chatbot_run",
    span_type="AGENT",
    attributes={"agent_name": "Tool-calling Chatbot"},
)
async def run_agent(messages: list[dict]) -> str:
    with mlflow.start_run(nested=True):
        mlflow.log_param("agent_name", agent.name)
        mlflow.log_param("model_endpoint", agent.model)
        mlflow.log_param("num_messages", len(messages))
        mlflow.log_text(json.dumps(messages, indent=2), "inputs/messages.json")
    # McpServer is an async context manager: Opens the connection before the run
    async with mcp_server:
        result = await Runner.run(agent, messages)
    final_output = result.final_output
    mlflow.log_text(final_output, "outputs/final_output.txt")
    return final_output