import argparse
import asyncio
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from type.tools import *
from lib.tools import discover_tools

# Configure logging
import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    #global logger
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--voice", action='store_true',
                        help="Enable voice input for the agent")
    parser.add_argument("--speaker_only", action='store_true',
                        help="Enable speaker only mode")
    parser.add_argument("--verbose", action='store_true',
                        help="Enable verbose logging")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Maximum number of retries for the critique session")
    parser.add_argument("--theme", type=str, default=None,
                        help="Background color for the console")
    # use memory
    parser.add_argument("--use_memory", action='store_true',
                        help="Enable memory usage for the agent")
    # clear memory
    parser.add_argument("--clear_memory", action='store_true',
                        help="Clear memory before starting the agent")
    
    # initialize memory
    parser.add_argument("--load_memory", type=str, default=None,
                        help="Load memory before starting the agent")
    # user_id
    parser.add_argument("--user_id", type=str, default="default_user",
                        help="User ID for memory operations")
    
    # persona
    parser.add_argument("--persona", type=str, default="assistant",
                        help="Persona for the agent")
    
    # use critique
    parser.add_argument("--use_critique", action='store_true',
                        help="Enable critique usage for the agent")
    
    # use dspy
    parser.add_argument("--use_dspy", action='store_true',
                        help="Use dspy model serving")
    
    # use openai api
    parser.add_argument("--use_openai", action='store_true',
                        help="Use OpenAI API for model serving")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        args.is_vllm = config['serving'].get('vllm', {}).get('enabled', False)
        if args.is_vllm:
            args.model = config['serving']['vllm']['model']
            args.host = config['serving']['vllm']['host']
            args.critique = config['serving']['vllm']['critique']
            args.think = config['serving']['vllm'].get('think', True)
        else:
            args.model = config['serving']['ollama']['model']
            args.host = config['serving']['ollama']['host']
            args.critique = config['serving']['ollama']['critique']
            args.think = config['serving']['ollama'].get('think', True)

        args.memory_config = config.get('memory', {})

        mcp_servers = config['mcp']['servers'] if 'mcp' in config and 'servers' in config['mcp'] else []
        tool_registry = asyncio.run(discover_tools(mcp_servers))
        
        args.pipelines = config.get('pipelines', {})
        # set up pipeline options
        if args.pipelines and tool_registry:
            for pipeline_name, pipeline_tools in args.pipelines.items():
                if pipeline_name in tool_registry.tools:
                    options = {}
                    for pipeline_tool in pipeline_tools:
                        if pipeline_tool in tool_registry.tools:
                            url = tool_registry.get_url(pipeline_tool)
                            tool_item = tool_registry[pipeline_tool].get_tool()
                            options[pipeline_tool] = {'url': url, 'tool_item': tool_item}
                
                    if 'set_options' in tool_registry.tools:
                        url = tool_registry.get_url(pipeline_name)
                        tool_handler = tool_registry.get_handler_by(tool_name='set_options', url=url)
                        if tool_handler:
                            async def run_tool_handler():
                                try:
                                    return await tool_handler(options=options)
                                except Exception as e:
                                    logger.error(f"Error running pipeline tool {pipeline_name}: {e}")
                                    return None
                            _ = asyncio.run(run_tool_handler())
        
        args.tool_registry = tool_registry
        
        # print all tools

        tools = tool_registry.get_tool_items() if tool_registry else []  
        for tool in tools:
            logger.info(f"Tool: {tool['function']['name']}")
        
    return args