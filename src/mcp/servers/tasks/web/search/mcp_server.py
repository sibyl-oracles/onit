'''
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Web Search and Content MCP Server - Consolidated API

Search the web, fetch content, extract media, and get weather information.

Requires:
    pip install ddgs geopy requests beautifulsoup4 PyPDF2 PyMuPDF python-dateutil

    Environment variables:
    - OPENWEATHER_API_KEY or OPENWEATHERMAP_API_KEY for weather data

4 Core Tools:
1. search - Unified web search (news or general)
2. fetch_content - Extract content and media URLs from any webpage
3. get_weather - Weather with automatic location detection
4. extract_pdf_images - Extract images from PDF files
'''

import os
import re
import json
import time
import hashlib
import tempfile
import requests
from io import BytesIO
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fastmcp import FastMCP

try:
    from .web_search import WebSearch
except ImportError:
    from web_search import WebSearch

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
    logging.getLogger('primp').setLevel(logging.WARNING)
except ImportError:
    raise ImportError("`ddgs` not installed. Please install using `pip install ddgs`")

try:
    from geopy.geocoders import Nominatim
except ImportError:
    raise ImportError("`geopy` not installed. Please install using `pip install geopy`")

# Get OpenWeather API key from environment
openweather_api_key = os.environ.get('OPENWEATHER_API_KEY') or os.environ.get('OPENWEATHERMAP_API_KEY')

mcp = FastMCP("Web Search MCP Server")

# Constants
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60
MAX_SIZE = 100 * 1024 * 1024  # 100MB limit

# Data path for file creation (set via options['data_path'] in run())
# All file writes are confined to this directory. Never use home folder.
DATA_PATH = os.path.join(tempfile.gettempdir(), "onit", "data")
DEFAULT_MEDIA_DIR = None  # Set in run() based on DATA_PATH

# Common image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico'}
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.ogg', '.mov', '.avi'}


def _secure_makedirs(dir_path: str) -> None:
    """Create directory with owner-only permissions (0o700)."""
    os.makedirs(dir_path, mode=0o700, exist_ok=True)


def _get_media_dir() -> str:
    """Get the media directory path within DATA_PATH."""
    return os.path.join(os.path.abspath(os.path.expanduser(DATA_PATH)), "media")


def _get_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    adapter = HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5))
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _read_pdf(url: str) -> str:
    """Extract text content from a PDF URL."""
    try:
        response = requests.get(url, timeout=READ_TIMEOUT)
        from PyPDF2 import PdfReader
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def _get_location_from_ip():
    """Get location from IP address."""
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()
        if data.get('status') == 'success':
            return data.get('lat'), data.get('lon'), data.get('city'), data.get('country')
    except Exception:
        pass
    return None, None, None, None


def _get_coordinates(place_name: str):
    """Get coordinates from place name."""
    try:
        geolocator = Nominatim(user_agent="weather_app")
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude, location.address
    except Exception:
        pass
    return None, None, None


def _extract_media_urls(soup: BeautifulSoup, base_url: str) -> dict:
    """Extract image and video URLs from parsed HTML."""
    images = []
    videos = []

    # Extract images
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
        if src:
            full_url = urljoin(base_url, src)
            # Filter out tiny images, icons, trackers
            width = img.get('width', '999')
            height = img.get('height', '999')
            try:
                if int(width) < 50 or int(height) < 50:
                    continue
            except (ValueError, TypeError):
                pass

            # Skip common tracking/icon patterns
            if any(x in full_url.lower() for x in ['pixel', 'tracker', 'beacon', '1x1', 'spacer', 'blank']):
                continue

            if full_url not in images:
                images.append(full_url)

    # Extract from srcset
    for img in soup.find_all(['img', 'source']):
        srcset = img.get('srcset')
        if srcset:
            for part in srcset.split(','):
                url_part = part.strip().split()[0]
                if url_part:
                    full_url = urljoin(base_url, url_part)
                    if full_url not in images:
                        images.append(full_url)

    # Extract background images from style attributes
    for tag in soup.find_all(style=True):
        style = tag.get('style', '')
        urls = re.findall(r'url\(["\']?([^"\'()]+)["\']?\)', style)
        for url in urls:
            full_url = urljoin(base_url, url)
            if any(full_url.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                if full_url not in images:
                    images.append(full_url)

    # Extract videos
    for video in soup.find_all('video'):
        src = video.get('src')
        if src:
            videos.append(urljoin(base_url, src))
        for source in video.find_all('source'):
            src = source.get('src')
            if src:
                full_url = urljoin(base_url, src)
                if full_url not in videos:
                    videos.append(full_url)

    # Extract YouTube/Vimeo embeds
    for iframe in soup.find_all('iframe'):
        src = iframe.get('src', '')
        if 'youtube.com' in src or 'youtu.be' in src:
            videos.append(src)
        elif 'vimeo.com' in src:
            videos.append(src)
        elif 'player' in src and any(x in src for x in ['video', 'embed']):
            videos.append(src)

    return {"images": images[:50], "videos": videos[:20]}  # Limit results


def _download_file(url: str, output_dir: str, timeout: int = 30) -> dict:
    """Download a file and return info about it."""
    try:
        _secure_makedirs(output_dir)

        # Generate filename from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "download"
        if not os.path.splitext(filename)[1]:
            # No extension, try to determine from content-type later
            filename += ".bin"

        # Make filename unique using hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{url_hash}{ext}"
        filepath = os.path.join(output_dir, filename)

        # Download
        response = requests.get(url, timeout=timeout, stream=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        response.raise_for_status()

        # Check content type and size
        content_type = response.headers.get('Content-Type', '')
        content_length = response.headers.get('Content-Length')

        if content_length and int(content_length) > MAX_SIZE:
            return {"error": f"File too large: {content_length} bytes"}

        # Write to file with owner-only permissions
        fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return {
            "url": url,
            "path": filepath,
            "size_bytes": os.path.getsize(filepath),
            "content_type": content_type
        }
    except Exception as e:
        return {"url": url, "error": str(e)}


# =============================================================================
# TOOL 1: UNIFIED SEARCH
# =============================================================================

@mcp.tool(
    title="Search the Web",
    description="""Search the web for news or general information using DuckDuckGo.

Args:
- query: Search terms (e.g., "AI regulations 2024", "how to bake bread")
- type: "news" for recent news, "web" for general search (default: "web")
- max_results: Number of results (default: 5, max: 10)

Returns JSON: [{title, snippet, url, source, date}]"""
)
def search(
    query: str,
    type: str = "web",
    max_results: int = 5
) -> str:
    try:
        max_results = min(max_results, 10)
        ddgs = DDGS(timeout=10)

        if type == "news":
            results = ddgs.news(query=query, max_results=max_results)
            formatted = []
            for r in results:
                date_pub = r.get('date', '')
                try:
                    dt = parser.parse(date_pub)
                    date_pub = dt.strftime('%B %d, %Y')
                except Exception:
                    pass
                formatted.append({
                    "title": r.get('title', ''),
                    "snippet": r.get('body', ''),
                    "date": date_pub,
                    "source": r.get('source', ''),
                    "url": r.get('url', '')
                })
            return json.dumps(formatted, indent=2)
        else:
            # Use the existing WebSearch for general queries
            search_tool = WebSearch()
            return search_tool.search(query)

    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


# =============================================================================
# TOOL 2: FETCH CONTENT WITH MEDIA EXTRACTION
# =============================================================================

@mcp.tool(
    title="Fetch Web Content",
    description="""Fetch content from a URL. Extracts text, images, and video links. Handles PDFs.

Args:
- url: Webpage URL to fetch (e.g., "https://example.com/article")
- extract_media: Extract image/video URLs (default: True)
- download_media: Download media files locally (default: False)
- output_dir: Save location for downloads (default: server data directory/media)
- media_limit: Max files to download (default: 10)

Returns JSON: {title, url, content, images, videos, downloaded}"""
)
def fetch_content(
    url: str,
    extract_media: bool = True,
    download_media: bool = False,
    output_dir: str = "",
    media_limit: int = 10
) -> str:
    try:
        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Handle unsupported extensions
        unsupported = (".docx", ".xlsx", ".pptx", ".zip", ".tar", ".gz")
        if url.lower().endswith(unsupported):
            return json.dumps({"error": f"Unsupported file type", "url": url})

        # Handle PDFs
        if url.lower().endswith(".pdf"):
            pdf_text = _read_pdf(url)
            return json.dumps({
                "title": "PDF Document",
                "url": url,
                "content": pdf_text,
                "content_type": "application/pdf"
            }, indent=2)

        # Fetch the page
        session = _get_session()
        try:
            response = session.get(
                url,
                headers={"Connection": "close"},
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return json.dumps({"error": "Request timed out", "url": url})
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"Request failed: {str(e)}", "url": url})
        finally:
            session.close()

        # Check size
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > MAX_SIZE:
            return json.dumps({"error": "File too large", "url": url})

        # Read content with timeout protection
        start_time = time.time()
        chunks = []
        total_size = 0

        for chunk in response.iter_content(chunk_size=8192):
            if time.time() - start_time > READ_TIMEOUT:
                break
            if chunk:
                total_size += len(chunk)
                if total_size > MAX_SIZE:
                    break
                chunks.append(chunk)

        # Decode bytes to string
        raw_content = b"".join(chunks)
        # Try to detect encoding from response, fallback to utf-8
        encoding = response.encoding or 'utf-8'
        try:
            html = raw_content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            html = raw_content.decode('utf-8', errors='replace')

        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = "No title"
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Extract media URLs
        media = {"images": [], "videos": []}
        if extract_media:
            media = _extract_media_urls(soup, url)

        # Clean and extract text content
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        # Try to find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': re.compile(r'content|article|post', re.I)})
        if main_content:
            text = main_content.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")

        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        content = "\n".join(line for line in lines if line)

        # Build result
        result = {
            "title": title,
            "url": url,
            "content": content[:50000],  # Limit content length
        }

        if extract_media:
            result["images"] = media["images"]
            result["videos"] = media["videos"]
            result["image_count"] = len(media["images"])
            result["video_count"] = len(media["videos"])

        # Download media if requested
        if download_media and media["images"]:
            output_path = os.path.abspath(os.path.expanduser(output_dir)) if output_dir else _get_media_dir()
            downloaded = []
            for img_url in media["images"][:media_limit]:
                dl_result = _download_file(img_url, output_path)
                if "path" in dl_result:
                    downloaded.append(dl_result)
            result["downloaded"] = downloaded
            result["download_dir"] = output_path

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "url": url})


# =============================================================================
# TOOL 3: WEATHER (WITH BUILT-IN LOCATION)
# =============================================================================

@mcp.tool(
    title="Get Weather",
    description="""Get current weather and optional 5-day forecast. Auto-detects location if not specified.

Args:
- place: City or location (e.g., "Tokyo, Japan"). Auto-detects from IP if omitted
- forecast: Include 5-day forecast (default: False)

Returns JSON: {location, current: {description, temperature_c, humidity_percent, wind_speed_ms, sunrise, sunset}, forecast_5day}

Requires: OPENWEATHER_API_KEY environment variable."""
)
def get_weather(
    place: str = None,
    forecast: bool = False
) -> str:
    global openweather_api_key

    if not openweather_api_key:
        return json.dumps({
            "error": "OpenWeather API key not set",
            "help": "Set OPENWEATHER_API_KEY or OPENWEATHERMAP_API_KEY environment variable"
        })

    try:
        # Get coordinates
        if place and place.strip():
            lat, lon, address = _get_coordinates(place)
            city = address or place
            country = None
        else:
            lat, lon, city, country = _get_location_from_ip()

        if lat is None or lon is None:
            return json.dumps({"error": "Could not determine location coordinates"})

        location_name = f"{city}, {country}" if (city and country) else (city or "Unknown")

        # Fetch current weather
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={openweather_api_key}&units=metric"
        weather = requests.get(url, timeout=10).json()

        if weather.get('cod') != 200:
            return json.dumps({"error": f"Weather API error: {weather.get('message', 'Unknown error')}"})

        # Format sunrise/sunset
        sunrise = datetime.fromtimestamp(weather['sys']['sunrise']).strftime('%I:%M %p')
        sunset = datetime.fromtimestamp(weather['sys']['sunset']).strftime('%I:%M %p')

        result = {
            "location": location_name,
            "coordinates": {"lat": lat, "lon": lon},
            "current": {
                "description": weather['weather'][0]['description'],
                "temperature_c": weather['main']['temp'],
                "feels_like_c": weather['main']['feels_like'],
                "humidity_percent": weather['main']['humidity'],
                "wind_speed_ms": weather['wind']['speed'],
                "sunrise": sunrise,
                "sunset": sunset
            },
            "source": f"https://openweathermap.org/city/{weather['id']}"
        }

        # Fetch forecast if requested
        if forecast:
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={openweather_api_key}&units=metric"
            forecast_data = requests.get(forecast_url, timeout=10).json()

            if forecast_data.get('cod') == '200':
                forecast_list = []
                for entry in forecast_data['list']:
                    dt = datetime.fromtimestamp(entry['dt'])
                    forecast_list.append({
                        "datetime": dt.strftime('%Y-%m-%d %H:%M'),
                        "day": dt.strftime('%A'),
                        "description": entry['weather'][0]['description'],
                        "temperature_c": entry['main']['temp']
                    })
                result["forecast_5day"] = forecast_list

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Weather fetch failed: {str(e)}"})


# =============================================================================
# TOOL 4: EXTRACT IMAGES FROM PDF
# =============================================================================

@mcp.tool(
    title="Extract PDF Images",
    description="""Extract all images from a PDF file and save them locally.

Args:
- pdf_path: Path to PDF file or URL (required)
- output_dir: Directory to save extracted images (default: server data directory/pdf_images)
- min_size: Minimum image dimension in pixels to extract (default: 100)

Returns JSON: {pdf_path, output_dir, images: [{path, width, height, format}], image_count, status}"""
)
def extract_pdf_images(
    pdf_path: str,
    output_dir: str = "",
    min_size: int = 100
) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF not installed. Run: pip install PyMuPDF",
            "pdf_path": pdf_path
        })

    try:
        # Normalize output directory (default to DATA_PATH/pdf_images)
        if output_dir:
            output_path = os.path.abspath(os.path.expanduser(output_dir))
        else:
            output_path = os.path.join(os.path.abspath(os.path.expanduser(DATA_PATH)), "pdf_images")
        _secure_makedirs(output_path)

        # Handle URL or local path
        if pdf_path.startswith(('http://', 'https://')):
            # Download PDF first
            response = requests.get(pdf_path, timeout=30)
            response.raise_for_status()
            pdf_data = response.content
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            pdf_name = os.path.basename(urlparse(pdf_path).path) or "document"
        else:
            # Local file
            local_path = os.path.abspath(os.path.expanduser(pdf_path))
            if not os.path.exists(local_path):
                return json.dumps({"error": f"PDF not found: {local_path}"})
            doc = fitz.open(local_path)
            pdf_name = os.path.splitext(os.path.basename(local_path))[0]

        extracted_images = []
        image_count = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image["width"]
                    height = base_image["height"]

                    # Skip small images (likely icons/artifacts)
                    if width < min_size or height < min_size:
                        continue

                    # Save image with owner-only permissions
                    image_count += 1
                    image_filename = f"{pdf_name}_p{page_num + 1}_img{img_index + 1}.{image_ext}"
                    image_path = os.path.join(output_path, image_filename)

                    fd = os.open(image_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "wb") as f:
                        f.write(image_bytes)

                    extracted_images.append({
                        "path": image_path,
                        "width": width,
                        "height": height,
                        "format": image_ext,
                        "page": page_num + 1
                    })

                except Exception:
                    # Skip images that can't be extracted
                    continue

        doc.close()

        return json.dumps({
            "pdf_path": pdf_path,
            "output_dir": output_path,
            "images": extracted_images,
            "image_count": len(extracted_images),
            "status": "success" if extracted_images else "no images found"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "pdf_path": pdf_path,
            "status": "failed"
        })


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 9003,
    path: str = "/search",
    options: dict = {}
) -> None:
    """Run the MCP server."""
    global DATA_PATH, DEFAULT_MEDIA_DIR

    if 'verbose' in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if 'data_path' in options:
        DATA_PATH = options['data_path']
    abs_data = os.path.abspath(os.path.expanduser(DATA_PATH))
    DEFAULT_MEDIA_DIR = os.path.join(abs_data, "media")
    _secure_makedirs(abs_data)

    logger.info(f"Starting Web Search MCP Server at {host}:{port}{path}")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info("4 Core Tools: search, fetch_content, get_weather, extract_pdf_images")

    mcp.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
