from fastmcp import FastMCP

try:
    from ..servers.tasks.shared import uvicorn_config
except ImportError:
    from mcp.servers.tasks.shared import uvicorn_config


import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

vlm_tools_mcp = FastMCP("VLM-Tools-MCP")

@vlm_tools_mcp.tool
def view_image_from_url(url: str):
    """
    Analyzes an image from a given URL.
    
    Args:
        url (str): The URL of the image to analyze.
    
    Returns:
        json: base64 encoded image
    """

    try:
        import requests
        from PIL import Image
        from io import BytesIO
        import base64
        import uuid

        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        image = Image.open(BytesIO(response.content))
        
        # Convert RGBA/palette images to RGB for JPEG compatibility
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")

        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # You can change the format if needed
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        file_name = f"{uuid.uuid4()}.jpg"  # Generate a unique file name

        mime_type = "image/jpeg"  # Adjust this if you are using a different image format

        return {
            "file_data_base64": img_str,
            "mime_type": mime_type,
            "file_name": file_name
        }
    except Exception as e:
        logger.error(f"Error in view_image_from_url: {e}")
        return {"error": str(e)}

def run(
    transport = "sse",
    host = "0.0.0.0",
    port = 18222,
    path = "/sse",
    options = {}
) -> None:
    if 'verbose' in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    quiet = 'verbose' not in options
    if quiet:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    logger.info(f"Starting VLM Tools MCP server on {host}:{port}{path} with transport {transport}")
    if transport == 'stdio':
        # stdout is the protocol channel on stdio: no socket, no banner.
        vlm_tools_mcp.run(transport='stdio', show_banner=False)
        return

    vlm_tools_mcp.run(transport=transport, host=host, port=port, path=path,
                      uvicorn_config=uvicorn_config(quiet=quiet))