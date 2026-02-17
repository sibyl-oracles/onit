'''
Google Workspace MCP Server

Tools for Google Drive, Docs, Sheets, Slides, and Gmail.

Requirements:
    pip install google-api-python-client google-auth-oauthlib google-auth-httplib2

Authentication:
    1. Create service account at Google Cloud Console > IAM & Admin > Service Accounts
    2. Enable APIs: Docs, Sheets, Slides, Drive, Gmail
    3. Download JSON key and configure via:
       - Server config: credentials_file: /path/to/service-account.json
       - Env var: GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
       - Default: ~/.config/gcloud/credentials.json
    4. Share existing files with service account email
    5. For Gmail: Enable domain-wide delegation and configure in Google Admin Console

IMPORTANT - Google Workspace (Organization) Accounts:
    For Google Workspace accounts, service accounts require domain-wide delegation
    to create files. Use the user_email parameter in create tools to impersonate
    a user. Setup:
    1. In Google Cloud Console: Enable domain-wide delegation for service account
    2. In Google Admin Console (admin.google.com):
       - Security > API Controls > Domain-wide delegation
       - Add service account client ID with scopes:
         https://www.googleapis.com/auth/documents
         https://www.googleapis.com/auth/spreadsheets
         https://www.googleapis.com/auth/presentations
         https://www.googleapis.com/auth/drive
         https://www.googleapis.com/auth/gmail.modify

    Without delegation, files created by service accounts are owned by the service
    account and not visible to users unless explicitly shared via share_with parameter.

Tools (22):
  Auth:      google_auth
  Drive:     drive_list, drive_create_folder, drive_move, drive_delete, drive_share, drive_download, drive_upload
  Docs:      doc_create, doc_read, doc_write
  Sheets:    sheet_create, sheet_read, sheet_write
  Slides:    slides_create, slides_edit, slides_read
  Gmail:     gmail_list, gmail_read, gmail_modify, gmail_send, gmail_attachment, gmail_create_label
'''
import os
import json
import uuid
import base64
import tempfile
import mimetypes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional

from fastmcp import FastMCP

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("Google Workspace MCP Server")

# Data path for file creation (set via options['data_path'] in run())
# All file writes are confined to this directory. Never use home folder.
DATA_PATH = os.path.join(tempfile.gettempdir(), "onit", "data")

SCOPES = [
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.compose',
]

_credentials = None
_docs_service = None
_sheets_service = None
_slides_service = None
_drive_service = None
_gmail_service = None
_credentials_file = None
_delegated_user = None  # For Gmail domain-wide delegation

DEFAULT_TOKEN_PATH = os.path.expanduser('~/.config/gcloud/token.pickle')
DEFAULT_CREDS_PATH = os.path.expanduser('~/.config/gcloud/credentials.json')


def _get_credentials():
    """Get Google API credentials."""
    global _credentials
    if _credentials and _credentials.valid:
        return _credentials

    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
        import pickle
    except ImportError:
        raise ImportError("Install: pip install google-api-python-client google-auth-oauthlib")

    # Try credential sources in priority order
    for path in [_credentials_file, os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'), DEFAULT_CREDS_PATH]:
        if path and os.path.exists(path):
            _credentials = service_account.Credentials.from_service_account_file(path, scopes=SCOPES)
            return _credentials

    # Try OAuth2 token
    if os.path.exists(DEFAULT_TOKEN_PATH):
        with open(DEFAULT_TOKEN_PATH, 'rb') as f:
            _credentials = pickle.load(f)
        if _credentials and _credentials.expired and _credentials.refresh_token:
            _credentials.refresh(Request())
        if _credentials and _credentials.valid:
            return _credentials

    raise PermissionError(
        "No credentials found. Set credentials_file in config, "
        f"GOOGLE_APPLICATION_CREDENTIALS env var, or place JSON at {DEFAULT_CREDS_PATH}"
    )


def _get_service(service_type: str, delegated_user: Optional[str] = None):
    """Get Google API service.

    For Google Workspace accounts, domain-wide delegation may be required.
    Pass delegated_user to impersonate a user when creating files.
    """
    global _docs_service, _sheets_service, _slides_service, _drive_service, _gmail_service
    from googleapiclient.discovery import build

    services = {
        'docs': ('docs', 'v1', '_docs_service'),
        'sheets': ('sheets', 'v4', '_sheets_service'),
        'slides': ('slides', 'v1', '_slides_service'),
        'drive': ('drive', 'v3', '_drive_service'),
        'gmail': ('gmail', 'v1', '_gmail_service'),
    }

    name, version, cache_var = services[service_type]
    user = delegated_user or _delegated_user

    # Use delegation if a user is specified (required for Google Workspace)
    if user:
        creds = _get_credentials()
        delegated_creds = creds.with_subject(user)
        return build(name, version, credentials=delegated_creds)

    # Use cached service for non-delegated requests
    cached = globals()[cache_var]
    if cached is None:
        cached = build(name, version, credentials=_get_credentials())
        globals()[cache_var] = cached
    return cached


def _error(e: Exception, op: str) -> dict:
    """Format error response."""
    if hasattr(e, 'resp') and hasattr(e.resp, 'status'):
        status = e.resp.status
        if status == 403:
            return {"error": str(e), "status": "permission_denied", "operation": op,
                    "fix": "Enable API in Google Cloud Console and check permissions"}
        if status == 404:
            return {"error": str(e), "status": "not_found", "operation": op}
        if status == 401:
            return {"error": str(e), "status": "unauthorized", "operation": op}
    return {"error": str(e), "status": "failed", "operation": op}


def _secure_makedirs(dir_path: str) -> None:
    """Create directory with owner-only permissions (0o700)."""
    os.makedirs(dir_path, mode=0o700, exist_ok=True)


def _resolve_save_path(save_path: str) -> str:
    """Resolve a save path to be within DATA_PATH.
    If path is relative or outside DATA_PATH, place it under DATA_PATH."""
    expanded = os.path.expanduser(save_path)
    abs_data = os.path.abspath(os.path.expanduser(DATA_PATH))
    if not os.path.isabs(expanded) or not os.path.abspath(expanded).startswith(abs_data):
        return abs_data
    return os.path.abspath(expanded)


# =============================================================================
# AUTHENTICATION
# =============================================================================

@mcp.tool(
    title="Google Auth",
    description="Check Google authentication status. Returns: {status, method, credentials_file}"
)
def google_auth() -> str:
    try:
        from google.oauth2 import service_account
    except ImportError:
        return json.dumps({"error": "Install google-api-python-client", "status": "failed"})

    for source, path in [("config", _credentials_file),
                         ("env", os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')),
                         ("default", DEFAULT_CREDS_PATH)]:
        if path and os.path.exists(path):
            try:
                service_account.Credentials.from_service_account_file(path, scopes=SCOPES)
                return json.dumps({"status": "authenticated", "method": "service_account",
                                   "source": source, "credentials_file": path})
            except Exception as e:
                return json.dumps({"status": "failed", "error": str(e), "credentials_file": path})

    return json.dumps({
        "status": "not_authenticated",
        "setup": "Create service account, download JSON key, set credentials_file in config"
    })


# =============================================================================
# GOOGLE DRIVE
# =============================================================================

@mcp.tool(
    title="Drive List",
    description="""List or search files/folders in Google Drive.

Args:
- query: Search query (optional). Examples: "name contains 'report'", "mimeType='application/vnd.google-apps.folder'"
- folder_id: List contents of specific folder (optional)
- file_type: Filter by type: "folder", "doc", "sheet", "slides", "pdf", "image" (optional)
- limit: Max results (default: 100)
- user_email: Email of user to impersonate (for Google Workspace with domain-wide delegation)

Returns: {files: [{id, name, mimeType, parents, url}], count}"""
)
def drive_list(
    query: Optional[str] = None,
    folder_id: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = 100,
    user_email: Optional[str] = None
) -> str:
    try:
        drive = _get_service('drive', user_email)

        # Build query
        q_parts = ["trashed=false"]
        if folder_id:
            q_parts.append(f"'{folder_id}' in parents")
        if query:
            q_parts.append(query)

        mime_types = {
            "folder": "application/vnd.google-apps.folder",
            "doc": "application/vnd.google-apps.document",
            "sheet": "application/vnd.google-apps.spreadsheet",
            "slides": "application/vnd.google-apps.presentation",
            "pdf": "application/pdf",
            "image": "image/",
        }
        if file_type and file_type in mime_types:
            if file_type == "image":
                q_parts.append("mimeType contains 'image/'")
            else:
                q_parts.append(f"mimeType='{mime_types[file_type]}'")

        results = drive.files().list(
            q=" and ".join(q_parts),
            pageSize=min(limit, 1000),
            fields="files(id, name, mimeType, parents, webViewLink, modifiedTime)"
        ).execute()

        files = []
        for f in results.get('files', []):
            files.append({
                "id": f['id'],
                "name": f['name'],
                "mimeType": f['mimeType'],
                "parents": f.get('parents', []),
                "url": f.get('webViewLink', f"https://drive.google.com/file/d/{f['id']}"),
                "modified": f.get('modifiedTime', '')
            })

        return json.dumps({"files": files, "count": len(files)}, indent=2)
    except Exception as e:
        return json.dumps(_error(e, "drive_list"))


@mcp.tool(
    title="Drive Create Folder",
    description="""Create a folder in Google Drive.

Args:
- name: Folder name (required)
- parent_id: Parent folder ID (optional, creates in root if not specified)
- user_email: Email of user to impersonate (required for Google Workspace with domain-wide delegation)
- share_with: Email address(es) to share with, comma-separated (optional)
- share_role: Permission role: "reader", "writer" (default: "writer")

Returns: {folder_id, name, url, shared_with}"""
)
def drive_create_folder(
    name: str,
    parent_id: Optional[str] = None,
    user_email: Optional[str] = None,
    share_with: Optional[str] = None,
    share_role: str = "writer"
) -> str:
    try:
        drive = _get_service('drive', user_email)

        metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            metadata['parents'] = [parent_id]

        folder = drive.files().create(body=metadata, fields='id, name').execute()
        folder_id = folder['id']

        # Share with specified users
        shared_with = []
        if share_with:
            for email in [e.strip() for e in share_with.split(',') if e.strip()]:
                try:
                    drive.permissions().create(
                        fileId=folder_id,
                        body={'type': 'user', 'role': share_role, 'emailAddress': email},
                        sendNotificationEmail=False
                    ).execute()
                    shared_with.append(email)
                except Exception as share_err:
                    logger.warning(f"Failed to share with {email}: {share_err}")

        return json.dumps({
            "folder_id": folder_id,
            "name": folder['name'],
            "url": f"https://drive.google.com/drive/folders/{folder_id}",
            "shared_with": shared_with,
            "status": "created"
        })
    except Exception as e:
        return json.dumps(_error(e, "drive_create_folder"))


@mcp.tool(
    title="Drive Move",
    description="""Move or rename a file/folder in Google Drive.

Args:
- file_id: File or folder ID to move/rename (required)
- new_parent_id: New parent folder ID (optional)
- new_name: New name (optional)

Returns: {file_id, name, parents, status}"""
)
def drive_move(
    file_id: str,
    new_parent_id: Optional[str] = None,
    new_name: Optional[str] = None
) -> str:
    try:
        drive = _get_service('drive')

        # Get current parents if moving
        update_body = {}
        add_parents = None
        remove_parents = None

        if new_parent_id:
            file = drive.files().get(fileId=file_id, fields='parents').execute()
            remove_parents = ','.join(file.get('parents', []))
            add_parents = new_parent_id

        if new_name:
            update_body['name'] = new_name

        result = drive.files().update(
            fileId=file_id,
            body=update_body if update_body else None,
            addParents=add_parents,
            removeParents=remove_parents,
            fields='id, name, parents'
        ).execute()

        return json.dumps({
            "file_id": result['id'],
            "name": result['name'],
            "parents": result.get('parents', []),
            "status": "moved" if new_parent_id else "renamed"
        })
    except Exception as e:
        return json.dumps(_error(e, "drive_move"))


@mcp.tool(
    title="Drive Delete",
    description="""Delete or trash a file/folder in Google Drive.

Args:
- file_id: File or folder ID (required)
- permanent: If True, permanently delete; if False, move to trash (default: False)

Returns: {file_id, status}"""
)
def drive_delete(file_id: str, permanent: bool = False) -> str:
    try:
        drive = _get_service('drive')

        if permanent:
            drive.files().delete(fileId=file_id).execute()
            status = "deleted permanently"
        else:
            drive.files().update(fileId=file_id, body={'trashed': True}).execute()
            status = "moved to trash"

        return json.dumps({"file_id": file_id, "status": status})
    except Exception as e:
        return json.dumps(_error(e, "drive_delete"))


@mcp.tool(
    title="Drive Share",
    description="""Share a file/folder with users or make it public.

Args:
- file_id: File or folder ID (required)
- email: Email address to share with (optional, required if not making public)
- role: "reader", "writer", "commenter" (default: "reader")
- type: "user", "group", "domain", "anyone" (default: "user")
- notify: Send email notification (default: True)

Returns: {file_id, permission_id, role, type, status}"""
)
def drive_share(
    file_id: str,
    email: Optional[str] = None,
    role: str = "reader",
    type: str = "user",
    notify: bool = True
) -> str:
    try:
        drive = _get_service('drive')

        permission = {'type': type, 'role': role}
        if type in ['user', 'group'] and email:
            permission['emailAddress'] = email

        result = drive.permissions().create(
            fileId=file_id,
            body=permission,
            sendNotificationEmail=notify if type in ['user', 'group'] else False,
            fields='id, role, type'
        ).execute()

        # Get shareable link
        file = drive.files().get(fileId=file_id, fields='webViewLink').execute()

        return json.dumps({
            "file_id": file_id,
            "permission_id": result['id'],
            "role": result['role'],
            "type": result['type'],
            "url": file.get('webViewLink', ''),
            "status": "shared"
        })
    except Exception as e:
        return json.dumps(_error(e, "drive_share"))


@mcp.tool(
    title="Drive Download",
    description="""Download a Google Doc, Sheet, Slides, or other file to local filesystem.

Args:
- file_id: File ID to download (required)
- save_path: Directory to save to (required)
- filename: Output filename without extension (optional, uses original name if not specified)
- format: Export format (optional, auto-detected based on file type):
  - For Docs: "pdf", "docx", "txt", "html", "rtf", "odt", "epub" (default: "pdf")
  - For Sheets: "pdf", "xlsx", "csv", "ods", "tsv" (default: "xlsx")
  - For Slides: "pdf", "pptx", "odp", "txt" (default: "pdf")
  - For Drawings: "pdf", "png", "jpg", "svg" (default: "pdf")
  - For other files: downloads as-is (format ignored)
- user_email: Email of user to impersonate (for Google Workspace with domain-wide delegation)

Returns: {file_id, filename, format, saved_to, size, status}"""
)
def drive_download(
    file_id: str,
    save_path: str,
    filename: Optional[str] = None,
    format: Optional[str] = None,
    user_email: Optional[str] = None
) -> str:
    try:
        from googleapiclient.http import MediaIoBaseDownload
        import io

        drive = _get_service('drive', user_email)

        # Get file metadata
        file_meta = drive.files().get(fileId=file_id, fields='name, mimeType').execute()
        original_name = file_meta['name']
        mime_type = file_meta['mimeType']

        # Define export formats for Google Workspace files
        export_formats = {
            'application/vnd.google-apps.document': {
                'pdf': ('application/pdf', '.pdf'),
                'docx': ('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx'),
                'txt': ('text/plain', '.txt'),
                'html': ('text/html', '.html'),
                'rtf': ('application/rtf', '.rtf'),
                'odt': ('application/vnd.oasis.opendocument.text', '.odt'),
                'epub': ('application/epub+zip', '.epub'),
            },
            'application/vnd.google-apps.spreadsheet': {
                'pdf': ('application/pdf', '.pdf'),
                'xlsx': ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', '.xlsx'),
                'csv': ('text/csv', '.csv'),
                'ods': ('application/vnd.oasis.opendocument.spreadsheet', '.ods'),
                'tsv': ('text/tab-separated-values', '.tsv'),
            },
            'application/vnd.google-apps.presentation': {
                'pdf': ('application/pdf', '.pdf'),
                'pptx': ('application/vnd.openxmlformats-officedocument.presentationml.presentation', '.pptx'),
                'odp': ('application/vnd.oasis.opendocument.presentation', '.odp'),
                'txt': ('text/plain', '.txt'),
            },
            'application/vnd.google-apps.drawing': {
                'pdf': ('application/pdf', '.pdf'),
                'png': ('image/png', '.png'),
                'jpg': ('image/jpeg', '.jpg'),
                'svg': ('image/svg+xml', '.svg'),
            },
        }

        # Default formats
        default_formats = {
            'application/vnd.google-apps.document': 'pdf',
            'application/vnd.google-apps.spreadsheet': 'xlsx',
            'application/vnd.google-apps.presentation': 'pdf',
            'application/vnd.google-apps.drawing': 'pdf',
        }

        # Determine base filename
        base_name = filename if filename else os.path.splitext(original_name)[0]

        # Check if it's a Google Workspace file that needs export
        if mime_type in export_formats:
            # Use specified format or default
            fmt = format.lower() if format else default_formats.get(mime_type, 'pdf')

            if fmt not in export_formats[mime_type]:
                available = ', '.join(export_formats[mime_type].keys())
                return json.dumps({
                    "error": f"Invalid format '{fmt}' for this file type",
                    "available_formats": available,
                    "status": "failed"
                })

            export_mime, extension = export_formats[mime_type][fmt]
            output_filename = base_name + extension

            # Export the file
            request = drive.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            # Regular file - download as-is
            output_filename = original_name if not filename else filename
            request = drive.files().get_media(fileId=file_id)

        # Download to memory first
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        # Save to filesystem (confined to DATA_PATH)
        save_path = _resolve_save_path(save_path)
        _secure_makedirs(save_path)
        output_path = os.path.join(save_path, output_filename)

        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(file_buffer.getvalue())

        file_size = file_buffer.tell()

        return json.dumps({
            "file_id": file_id,
            "original_name": original_name,
            "filename": output_filename,
            "format": format or default_formats.get(mime_type, 'original'),
            "saved_to": output_path,
            "size": file_size,
            "size_human": f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB",
            "status": "downloaded"
        })

    except Exception as e:
        return json.dumps(_error(e, "drive_download"))


@mcp.tool(
    title="Drive Upload",
    description="""Upload a local file to Google Drive, or update an existing file.

IMPORTANT: Service accounts don't have storage quota. You must either:
1. Use user_email to impersonate a user (requires domain-wide delegation), OR
2. Upload to a Shared Drive folder (provide folder_id of a shared drive folder)

Args:
- file_path: Local file path to upload (required)
- file_id: Existing file ID to update/overwrite (optional, creates new file if not specified)
- folder_id: Parent folder ID to upload to (optional, uploads to root if not specified, ignored when updating)
- filename: Name for the file in Drive (optional, uses original filename if not specified)
- user_email: Email of user to impersonate (required for service accounts without shared drive)
- share_with: Email address(es) to share with, comma-separated (optional)
- share_role: Permission role: "reader", "writer" (default: "writer")

Returns: {file_id, name, url, mime_type, size, shared_with, status}"""
)
def drive_upload(
    file_path: str,
    file_id: Optional[str] = None,
    folder_id: Optional[str] = None,
    filename: Optional[str] = None,
    user_email: Optional[str] = None,
    share_with: Optional[str] = None,
    share_role: str = "writer"
) -> str:
    try:
        from googleapiclient.http import MediaFileUpload

        # Validate file exists
        if not os.path.exists(file_path):
            return json.dumps({
                "error": f"File not found: {file_path}",
                "status": "failed"
            })

        drive = _get_service('drive', user_email)

        # Determine filename and mime type
        upload_filename = filename if filename else os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or 'application/octet-stream'

        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

        if file_id:
            # Update existing file
            file_metadata = {'name': upload_filename}
            file = drive.files().update(
                fileId=file_id,
                body=file_metadata,
                media_body=media,
                fields='id, name, mimeType, size, webViewLink',
                supportsAllDrives=True
            ).execute()
            status = "updated"
        else:
            # Create new file
            file_metadata = {'name': upload_filename}
            if folder_id:
                file_metadata['parents'] = [folder_id]
            file = drive.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, mimeType, size, webViewLink',
                supportsAllDrives=True
            ).execute()
            status = "uploaded"

        result_file_id = file['id']
        file_size = int(file.get('size', 0))

        # Share with specified users (only for new files or if explicitly requested)
        shared_with = []
        if share_with:
            for email in [e.strip() for e in share_with.split(',') if e.strip()]:
                try:
                    drive.permissions().create(
                        fileId=result_file_id,
                        body={'type': 'user', 'role': share_role, 'emailAddress': email},
                        sendNotificationEmail=False,
                        supportsAllDrives=True
                    ).execute()
                    shared_with.append(email)
                except Exception as share_err:
                    logger.warning(f"Failed to share with {email}: {share_err}")

        return json.dumps({
            "file_id": result_file_id,
            "name": file['name'],
            "url": file.get('webViewLink', f"https://drive.google.com/file/d/{result_file_id}"),
            "mime_type": file['mimeType'],
            "size": file_size,
            "size_human": f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB",
            "shared_with": shared_with,
            "status": status
        })

    except Exception as e:
        error_str = str(e)
        if "storage quota" in error_str.lower() or "do not have storage" in error_str.lower():
            return json.dumps({
                "error": "Service accounts cannot upload to personal Drive folders without delegation.",
                "fix": "Add user_email parameter to impersonate a user with storage quota (e.g., user_email='user@domain.com')",
                "status": "failed",
                "operation": "drive_upload"
            })
        return json.dumps(_error(e, "drive_upload"))


# =============================================================================
# GOOGLE DOCS
# =============================================================================

@mcp.tool(
    title="Doc Create",
    description="""Create a new Google Doc.

Args:
- title: Document title (required)
- content: Initial text content (optional)
- folder_id: Folder ID to create in (optional)
- user_email: Email of user to impersonate (required for Google Workspace with domain-wide delegation)
- share_with: Email address(es) to share with, comma-separated (optional)
- share_role: Permission role: "reader", "writer", "commenter" (default: "writer")

Returns: {document_id, url, title, shared_with}"""
)
def doc_create(
    title: str,
    content: Optional[str] = None,
    folder_id: Optional[str] = None,
    user_email: Optional[str] = None,
    share_with: Optional[str] = None,
    share_role: str = "writer"
) -> str:
    try:
        docs = _get_service('docs', user_email)
        doc = docs.documents().create(body={'title': title}).execute()
        doc_id = doc['documentId']

        if content:
            docs.documents().batchUpdate(documentId=doc_id, body={
                'requests': [{'insertText': {'location': {'index': 1}, 'text': content}}]
            }).execute()

        drive = _get_service('drive', user_email)

        if folder_id:
            drive.files().update(fileId=doc_id, addParents=folder_id, fields='id').execute()

        # Share with specified users
        shared_with = []
        if share_with:
            for email in [e.strip() for e in share_with.split(',') if e.strip()]:
                try:
                    drive.permissions().create(
                        fileId=doc_id,
                        body={'type': 'user', 'role': share_role, 'emailAddress': email},
                        sendNotificationEmail=False
                    ).execute()
                    shared_with.append(email)
                except Exception as share_err:
                    logger.warning(f"Failed to share with {email}: {share_err}")

        return json.dumps({
            "document_id": doc_id,
            "url": f"https://docs.google.com/document/d/{doc_id}/edit",
            "title": title,
            "shared_with": shared_with,
            "status": "created"
        })
    except Exception as e:
        return json.dumps(_error(e, "doc_create"))


@mcp.tool(
    title="Doc Read",
    description="""Read content from a Google Doc.

Args:
- document_id: Google Doc ID (required)
- format: "text" for plain text, "json" for full structure (default: "text")
- user_email: Email of user to impersonate (for Google Workspace with domain-wide delegation)

Returns: {document_id, title, content, url}"""
)
def doc_read(document_id: str, format: str = "text", user_email: Optional[str] = None) -> str:
    try:
        docs = _get_service('docs', user_email)
        doc = docs.documents().get(documentId=document_id).execute()

        if format == "text":
            # Extract plain text
            text_content = []
            for element in doc.get('body', {}).get('content', []):
                if 'paragraph' in element:
                    for elem in element['paragraph'].get('elements', []):
                        if 'textRun' in elem:
                            text_content.append(elem['textRun'].get('content', ''))
                elif 'table' in element:
                    text_content.append("[TABLE]")
            content = ''.join(text_content)
        else:
            content = doc.get('body', {})

        return json.dumps({
            "document_id": document_id,
            "title": doc.get('title', 'Untitled'),
            "content": content,
            "url": f"https://docs.google.com/document/d/{document_id}/edit"
        }, indent=2)
    except Exception as e:
        return json.dumps(_error(e, "doc_read"))


@mcp.tool(
    title="Doc Write",
    description="""Add or modify content in a Google Doc.

Args:
- document_id: Google Doc ID (required)
- action: "append", "insert", "replace" (default: "append")
- content_type: "text", "heading", "bullets", "numbered", "table" (default: "text")
- text: Text content for text/heading
- items: List of strings for bullets/numbered lists
- table_data: 2D list for tables, e.g. [["A","B"],["1","2"]]
- heading_level: 1-6 for headings (default: 1)
- index: Insert position for "insert" action (default: end)
- find_text: Text to find for "replace" action
- user_email: Email of user to impersonate (for Google Workspace with domain-wide delegation)

Returns: {document_id, url, status}"""
)
def doc_write(
    document_id: str,
    action: str = "append",
    content_type: str = "text",
    text: Optional[str] = None,
    items: Optional[List[str]] = None,
    table_data: Optional[List[List[str]]] = None,
    heading_level: int = 1,
    index: Optional[int] = None,
    find_text: Optional[str] = None,
    user_email: Optional[str] = None
) -> str:
    try:
        docs = _get_service('docs', user_email)
        doc = docs.documents().get(documentId=document_id).execute()
        end_index = doc['body']['content'][-1]['endIndex'] - 1
        insert_index = index if index is not None else end_index

        requests = []

        if action == "replace" and find_text and text:
            requests.append({
                'replaceAllText': {
                    'containsText': {'text': find_text, 'matchCase': True},
                    'replaceText': text
                }
            })
        elif content_type == "text" and text:
            requests.append({'insertText': {'location': {'index': insert_index}, 'text': text + '\n'}})
        elif content_type == "heading" and text:
            requests.extend([
                {'insertText': {'location': {'index': insert_index}, 'text': text + '\n'}},
                {'updateParagraphStyle': {
                    'range': {'startIndex': insert_index, 'endIndex': insert_index + len(text) + 1},
                    'paragraphStyle': {'namedStyleType': f'HEADING_{min(max(heading_level, 1), 6)}'},
                    'fields': 'namedStyleType'
                }}
            ])
        elif content_type in ["bullets", "numbered"] and items:
            full_text = '\n'.join(items) + '\n'
            preset = 'BULLET_DISC_CIRCLE_SQUARE' if content_type == "bullets" else 'NUMBERED_DECIMAL_ALPHA_ROMAN'
            requests.extend([
                {'insertText': {'location': {'index': insert_index}, 'text': full_text}},
                {'createParagraphBullets': {
                    'range': {'startIndex': insert_index, 'endIndex': insert_index + len(full_text)},
                    'bulletPreset': preset
                }}
            ])
        elif content_type == "table" and table_data:
            rows, cols = len(table_data), len(table_data[0])
            requests.append({'insertTable': {'location': {'index': insert_index}, 'rows': rows, 'columns': cols}})
            docs.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()

            # Populate table cells
            doc = docs.documents().get(documentId=document_id).execute()
            table_requests = []
            for element in doc['body']['content']:
                if 'table' in element:
                    for row_idx, row in enumerate(element['table'].get('tableRows', [])):
                        for col_idx, cell in enumerate(row.get('tableCells', [])):
                            if row_idx < len(table_data) and col_idx < len(table_data[row_idx]):
                                cell_start = cell['content'][0]['startIndex']
                                table_requests.append({
                                    'insertText': {'location': {'index': cell_start}, 'text': str(table_data[row_idx][col_idx])}
                                })
                    break
            if table_requests:
                table_requests.reverse()
                docs.documents().batchUpdate(documentId=document_id, body={'requests': table_requests}).execute()

            return json.dumps({
                "document_id": document_id,
                "url": f"https://docs.google.com/document/d/{document_id}/edit",
                "status": f"added {rows}x{cols} table"
            })

        if requests:
            docs.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()

        return json.dumps({
            "document_id": document_id,
            "url": f"https://docs.google.com/document/d/{document_id}/edit",
            "status": f"{action}ed {content_type}"
        })
    except Exception as e:
        return json.dumps(_error(e, "doc_write"))


# =============================================================================
# GOOGLE SHEETS
# =============================================================================

@mcp.tool(
    title="Sheet Create",
    description="""Create a new Google Sheets spreadsheet.

Args:
- title: Spreadsheet title (required)
- sheet_name: First sheet name (default: "Sheet1")
- headers: List of column headers (optional)
- data: 2D list of data rows (optional)
- folder_id: Folder ID to create in (optional)
- user_email: Email of user to impersonate (required for Google Workspace with domain-wide delegation)
- share_with: Email address(es) to share with, comma-separated (optional)
- share_role: Permission role: "reader", "writer", "commenter" (default: "writer")

Returns: {spreadsheet_id, url, title, sheet_name, shared_with}"""
)
def sheet_create(
    title: str,
    sheet_name: str = "Sheet1",
    headers: Optional[List[str]] = None,
    data: Optional[List[List]] = None,
    folder_id: Optional[str] = None,
    user_email: Optional[str] = None,
    share_with: Optional[str] = None,
    share_role: str = "writer"
) -> str:
    try:
        sheets = _get_service('sheets', user_email)

        result = sheets.spreadsheets().create(body={
            'properties': {'title': title},
            'sheets': [{'properties': {'title': sheet_name}}]
        }).execute()
        spreadsheet_id = result['spreadsheetId']

        # Add data
        values = []
        if headers:
            values.append(headers)
        if data:
            values.extend(data)

        if values:
            sheets.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f'{sheet_name}!A1',
                valueInputOption='USER_ENTERED',
                body={'values': values}
            ).execute()

            # Format header row
            if headers:
                sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={
                    'requests': [{'repeatCell': {
                        'range': {'sheetId': 0, 'startRowIndex': 0, 'endRowIndex': 1},
                        'cell': {'userEnteredFormat': {
                            'backgroundColor': {'red': 0.27, 'green': 0.45, 'blue': 0.77},
                            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}
                        }},
                        'fields': 'userEnteredFormat(backgroundColor,textFormat)'
                    }}]
                }).execute()

        drive = _get_service('drive', user_email)

        if folder_id:
            drive.files().update(fileId=spreadsheet_id, addParents=folder_id, fields='id').execute()

        # Share with specified users
        shared_with = []
        if share_with:
            for email in [e.strip() for e in share_with.split(',') if e.strip()]:
                try:
                    drive.permissions().create(
                        fileId=spreadsheet_id,
                        body={'type': 'user', 'role': share_role, 'emailAddress': email},
                        sendNotificationEmail=False
                    ).execute()
                    shared_with.append(email)
                except Exception as share_err:
                    logger.warning(f"Failed to share with {email}: {share_err}")

        return json.dumps({
            "spreadsheet_id": spreadsheet_id,
            "url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
            "title": title,
            "sheet_name": sheet_name,
            "shared_with": shared_with,
            "status": "created"
        })
    except Exception as e:
        return json.dumps(_error(e, "sheet_create"))


@mcp.tool(
    title="Sheet Read",
    description="""Read data from a Google Sheet.

Args:
- spreadsheet_id: Spreadsheet ID (required)
- range: Cell range like "Sheet1!A1:D10" or "A1:D10" (required)
- user_email: Email of user to impersonate (for Google Workspace with domain-wide delegation)

Returns: {spreadsheet_id, range, values, row_count, column_count}"""
)
def sheet_read(spreadsheet_id: str, range: str, user_email: Optional[str] = None) -> str:
    try:
        sheets = _get_service('sheets', user_email)
        result = sheets.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range).execute()
        values = result.get('values', [])

        return json.dumps({
            "spreadsheet_id": spreadsheet_id,
            "url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
            "range": result.get('range', range),
            "values": values,
            "row_count": len(values),
            "column_count": len(values[0]) if values else 0
        }, indent=2)
    except Exception as e:
        return json.dumps(_error(e, "sheet_read"))


@mcp.tool(
    title="Sheet Write",
    description="""Write or update data in a Google Sheet.

Args:
- spreadsheet_id: Spreadsheet ID (required)
- data: 2D list of rows (required)
- range: Target range like "Sheet1!A1" (default: "A1")
- mode: "update" to overwrite, "append" to add at end (default: "update")
- user_email: Email of user to impersonate (for Google Workspace with domain-wide delegation)

Returns: {spreadsheet_id, updated_range, rows_affected}"""
)
def sheet_write(
    spreadsheet_id: str,
    data: List[List],
    range: str = "A1",
    mode: str = "update",
    user_email: Optional[str] = None
) -> str:
    try:
        sheets = _get_service('sheets', user_email)

        if mode == "append":
            result = sheets.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=range,
                valueInputOption='USER_ENTERED',
                insertDataOption='INSERT_ROWS',
                body={'values': data}
            ).execute()
            updated_range = result.get('updates', {}).get('updatedRange', '')
            rows_affected = result.get('updates', {}).get('updatedRows', 0)
        else:
            result = sheets.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range,
                valueInputOption='USER_ENTERED',
                body={'values': data}
            ).execute()
            updated_range = result.get('updatedRange', '')
            rows_affected = result.get('updatedRows', 0)

        return json.dumps({
            "spreadsheet_id": spreadsheet_id,
            "url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
            "updated_range": updated_range,
            "rows_affected": rows_affected,
            "status": f"{mode}d data"
        })
    except Exception as e:
        return json.dumps(_error(e, "sheet_write"))


# =============================================================================
# GOOGLE SLIDES
# =============================================================================

@mcp.tool(
    title="Slides Create",
    description="""Create a new Google Slides presentation.

Args:
- title: Presentation title (required)
- subtitle: Subtitle for title slide (optional)
- folder_id: Folder ID to create in (optional)
- user_email: Email of user to impersonate (required for Google Workspace with domain-wide delegation)
- share_with: Email address(es) to share with, comma-separated (optional)
- share_role: Permission role: "reader", "writer", "commenter" (default: "writer")

Returns: {presentation_id, url, title, shared_with}"""
)
def slides_create(
    title: str,
    subtitle: str = "",
    folder_id: Optional[str] = None,
    user_email: Optional[str] = None,
    share_with: Optional[str] = None,
    share_role: str = "writer"
) -> str:
    try:
        slides_svc = _get_service('slides', user_email)
        result = slides_svc.presentations().create(body={'title': title}).execute()
        presentation_id = result['presentationId']

        # Set title slide content
        presentation = slides_svc.presentations().get(presentationId=presentation_id).execute()
        if presentation.get('slides'):
            requests = []
            for element in presentation['slides'][0].get('pageElements', []):
                if 'shape' in element:
                    placeholder = element['shape'].get('placeholder', {}).get('type', '')
                    if placeholder in ['CENTERED_TITLE', 'TITLE']:
                        requests.append({'insertText': {'objectId': element['objectId'], 'text': title, 'insertionIndex': 0}})
                    elif placeholder == 'SUBTITLE' and subtitle:
                        requests.append({'insertText': {'objectId': element['objectId'], 'text': subtitle, 'insertionIndex': 0}})
            if requests:
                slides_svc.presentations().batchUpdate(presentationId=presentation_id, body={'requests': requests}).execute()

        drive = _get_service('drive', user_email)

        if folder_id:
            drive.files().update(fileId=presentation_id, addParents=folder_id, fields='id').execute()

        # Share with specified users
        shared_with = []
        if share_with:
            for email in [e.strip() for e in share_with.split(',') if e.strip()]:
                try:
                    drive.permissions().create(
                        fileId=presentation_id,
                        body={'type': 'user', 'role': share_role, 'emailAddress': email},
                        sendNotificationEmail=False
                    ).execute()
                    shared_with.append(email)
                except Exception as share_err:
                    logger.warning(f"Failed to share with {email}: {share_err}")

        return json.dumps({
            "presentation_id": presentation_id,
            "url": f"https://docs.google.com/presentation/d/{presentation_id}/edit",
            "title": title,
            "shared_with": shared_with,
            "status": "created"
        })
    except Exception as e:
        return json.dumps(_error(e, "slides_create"))


@mcp.tool(
    title="Slides Edit",
    description="""Add or edit slides in a presentation.

Args:
- presentation_id: Presentation ID (required)
- action: "add_slide" or "add_table" (default: "add_slide")
- layout: For add_slide: "title", "title_body", "blank", "section_header", "title_only" (default: "title_body")
- title: Slide title (optional)
- body: Body text or bullet points joined with newlines (optional)
- table_data: For add_table: 2D list of table data (optional)
- slide_index: For add_table: slide index to add table to (-1 = last, default: -1)

Returns: {presentation_id, url, slide_id/table_id, status}"""
)
def slides_edit(
    presentation_id: str,
    action: str = "add_slide",
    layout: str = "title_body",
    title: str = "",
    body: str = "",
    table_data: Optional[List[List[str]]] = None,
    slide_index: int = -1
) -> str:
    try:
        slides_svc = _get_service('slides')

        if action == "add_slide":
            layout_map = {
                "title": "TITLE", "title_body": "TITLE_AND_BODY", "blank": "BLANK",
                "section_header": "SECTION_HEADER", "title_only": "TITLE_ONLY"
            }
            slide_id = f"slide_{uuid.uuid4().hex[:8]}"

            slides_svc.presentations().batchUpdate(presentationId=presentation_id, body={
                'requests': [{'createSlide': {'objectId': slide_id, 'slideLayoutReference': {'predefinedLayout': layout_map.get(layout, 'TITLE_AND_BODY')}}}]
            }).execute()

            # Add content
            presentation = slides_svc.presentations().get(presentationId=presentation_id).execute()
            for slide in presentation.get('slides', []):
                if slide['objectId'] == slide_id:
                    text_requests = []
                    for element in slide.get('pageElements', []):
                        if 'shape' in element:
                            placeholder = element['shape'].get('placeholder', {}).get('type', '')
                            if placeholder in ['TITLE', 'CENTERED_TITLE'] and title:
                                text_requests.append({'insertText': {'objectId': element['objectId'], 'text': title, 'insertionIndex': 0}})
                            elif placeholder == 'BODY' and body:
                                text_requests.append({'insertText': {'objectId': element['objectId'], 'text': body, 'insertionIndex': 0}})
                            elif placeholder == 'SUBTITLE' and body and layout == "section_header":
                                text_requests.append({'insertText': {'objectId': element['objectId'], 'text': body, 'insertionIndex': 0}})
                    if text_requests:
                        slides_svc.presentations().batchUpdate(presentationId=presentation_id, body={'requests': text_requests}).execute()
                    break

            return json.dumps({
                "presentation_id": presentation_id,
                "url": f"https://docs.google.com/presentation/d/{presentation_id}/edit",
                "slide_id": slide_id,
                "status": f"added {layout} slide"
            })

        elif action == "add_table" and table_data:
            rows, cols = len(table_data), len(table_data[0])
            presentation = slides_svc.presentations().get(presentationId=presentation_id).execute()
            slides_list = presentation.get('slides', [])

            if slide_index == -1:
                target_slide_id = slides_list[-1]['objectId'] if slides_list else None
            else:
                target_slide_id = slides_list[slide_index]['objectId'] if slide_index < len(slides_list) else None

            if not target_slide_id:
                return json.dumps({"error": "No slide found"})

            table_id = f"table_{uuid.uuid4().hex[:8]}"
            slides_svc.presentations().batchUpdate(presentationId=presentation_id, body={
                'requests': [{'createTable': {
                    'objectId': table_id,
                    'elementProperties': {
                        'pageObjectId': target_slide_id,
                        'size': {'width': {'magnitude': 7500000, 'unit': 'EMU'}, 'height': {'magnitude': 4000000, 'unit': 'EMU'}},
                        'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 822000, 'translateY': 1500000, 'unit': 'EMU'}
                    },
                    'rows': rows, 'columns': cols
                }}]
            }).execute()

            # Populate cells
            cell_requests = []
            for row_idx, row_data in enumerate(table_data):
                for col_idx, cell_value in enumerate(row_data):
                    cell_requests.append({
                        'insertText': {
                            'objectId': table_id,
                            'cellLocation': {'rowIndex': row_idx, 'columnIndex': col_idx},
                            'text': str(cell_value),
                            'insertionIndex': 0
                        }
                    })
            if cell_requests:
                slides_svc.presentations().batchUpdate(presentationId=presentation_id, body={'requests': cell_requests}).execute()

            return json.dumps({
                "presentation_id": presentation_id,
                "url": f"https://docs.google.com/presentation/d/{presentation_id}/edit",
                "table_id": table_id,
                "size": f"{rows}x{cols}",
                "status": "added table"
            })

        return json.dumps({"error": "Invalid action or missing data"})
    except Exception as e:
        return json.dumps(_error(e, "slides_edit"))


@mcp.tool(
    title="Slides Read",
    description="""Get information about a Google Slides presentation.

Args:
- presentation_id: Presentation ID (required)

Returns: {presentation_id, title, url, slide_count, slides: [{index, id, title}]}"""
)
def slides_read(presentation_id: str) -> str:
    try:
        slides_svc = _get_service('slides')
        presentation = slides_svc.presentations().get(presentationId=presentation_id).execute()

        slide_list = []
        for idx, slide in enumerate(presentation.get('slides', [])):
            slide_title = ""
            for element in slide.get('pageElements', []):
                if 'shape' in element:
                    if element['shape'].get('placeholder', {}).get('type') in ['TITLE', 'CENTERED_TITLE']:
                        for text_elem in element['shape'].get('text', {}).get('textElements', []):
                            if 'textRun' in text_elem:
                                slide_title += text_elem['textRun'].get('content', '')
                        slide_title = slide_title.strip()
                        break
            slide_list.append({"index": idx, "id": slide['objectId'], "title": slide_title})

        return json.dumps({
            "presentation_id": presentation_id,
            "title": presentation.get('title', 'Untitled'),
            "url": f"https://docs.google.com/presentation/d/{presentation_id}/edit",
            "slide_count": len(slide_list),
            "slides": slide_list
        }, indent=2)
    except Exception as e:
        return json.dumps(_error(e, "slides_read"))


# =============================================================================
# GMAIL
# =============================================================================

@mcp.tool(
    title="Gmail List",
    description="""List or search emails in Gmail.

Args:
- query: Gmail search query (optional). Examples: "is:unread", "from:user@example.com", "subject:meeting", "has:attachment"
- max_results: Maximum emails to return (default: 20, max: 100)
- user_email: Email address to access (required for service account delegation)
- label: Filter by label: "INBOX", "SENT", "DRAFT", "SPAM", "TRASH", "STARRED" (default: "INBOX")

Returns: {emails: [{id, thread_id, subject, from, to, date, snippet, labels, has_attachments}], count}"""
)
def gmail_list(
    query: Optional[str] = None,
    max_results: int = 20,
    user_email: Optional[str] = None,
    label: str = "INBOX"
) -> str:
    try:
        gmail = _get_service('gmail', user_email)

        # Build query
        q = query or ""
        if label and label != "ALL":
            q = f"in:{label.lower()} {q}".strip()

        results = gmail.users().messages().list(
            userId='me',
            q=q,
            maxResults=min(max_results, 100)
        ).execute()

        messages = results.get('messages', [])
        emails = []

        for msg in messages:
            msg_data = gmail.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['From', 'To', 'Subject', 'Date']
            ).execute()

            headers = {h['name']: h['value'] for h in msg_data.get('payload', {}).get('headers', [])}
            parts = msg_data.get('payload', {}).get('parts', [])
            has_attachments = any(p.get('filename') for p in parts) if parts else False

            emails.append({
                "id": msg['id'],
                "thread_id": msg['threadId'],
                "subject": headers.get('Subject', '(no subject)'),
                "from": headers.get('From', ''),
                "to": headers.get('To', ''),
                "date": headers.get('Date', ''),
                "snippet": msg_data.get('snippet', '')[:100],
                "labels": msg_data.get('labelIds', []),
                "has_attachments": has_attachments
            })

        return json.dumps({"emails": emails, "count": len(emails)}, indent=2)
    except Exception as e:
        return json.dumps(_error(e, "gmail_list"))


@mcp.tool(
    title="Gmail Read",
    description="""Read a specific email with full content.

Args:
- message_id: Email message ID (required)
- user_email: Email address to access (required for service account delegation)
- format: "full" for complete message, "text" for plain text body only (default: "full")
- mark_as_read: Mark email as read after reading (default: True)
- add_labels: List of label IDs to add to this email (optional)
- remove_labels: List of label IDs to remove from this email (optional)

Returns: {id, thread_id, subject, from, to, cc, date, body, attachments: [{id, filename, mime_type, size}], marked_as_read, labels_added, labels_removed}"""
)
def gmail_read(
    message_id: str,
    user_email: Optional[str] = None,
    format: str = "full",
    mark_as_read: bool = True,
    add_labels: Optional[List[str]] = None,
    remove_labels: Optional[List[str]] = None
) -> str:
    try:
        gmail = _get_service('gmail', user_email)

        msg = gmail.users().messages().get(
            userId='me',
            id=message_id,
            format='full'
        ).execute()

        headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}

        # Mark as read and/or modify labels
        labels_added = []
        labels_removed = []
        was_unread = 'UNREAD' in msg.get('labelIds', [])

        modify_body = {}
        if mark_as_read and was_unread:
            modify_body['removeLabelIds'] = ['UNREAD']
            labels_removed.append('UNREAD')
        if add_labels:
            modify_body['addLabelIds'] = add_labels
            labels_added.extend(add_labels)
        if remove_labels:
            if 'removeLabelIds' in modify_body:
                modify_body['removeLabelIds'].extend(remove_labels)
            else:
                modify_body['removeLabelIds'] = remove_labels
            labels_removed.extend(remove_labels)

        if modify_body:
            gmail.users().messages().modify(
                userId='me',
                id=message_id,
                body=modify_body
            ).execute()

        # Extract body
        def get_body(payload):
            if 'body' in payload and payload['body'].get('data'):
                return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        if part['body'].get('data'):
                            return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                    elif part['mimeType'] == 'text/html' and format == "full":
                        if part['body'].get('data'):
                            return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                    elif 'parts' in part:
                        result = get_body(part)
                        if result:
                            return result
            return ""

        body = get_body(msg.get('payload', {}))

        # Extract attachments info
        attachments = []
        def get_attachments(payload):
            if 'parts' in payload:
                for part in payload['parts']:
                    if part.get('filename'):
                        attachments.append({
                            "id": part['body'].get('attachmentId', ''),
                            "filename": part['filename'],
                            "mime_type": part['mimeType'],
                            "size": part['body'].get('size', 0)
                        })
                    if 'parts' in part:
                        get_attachments(part)

        get_attachments(msg.get('payload', {}))

        result = {
            "id": msg['id'],
            "thread_id": msg['threadId'],
            "subject": headers.get('Subject', '(no subject)'),
            "from": headers.get('From', ''),
            "to": headers.get('To', ''),
            "cc": headers.get('Cc', ''),
            "date": headers.get('Date', ''),
            "labels": msg.get('labelIds', []),
            "body": body if format == "full" else body[:5000],
            "attachments": attachments,
            "marked_as_read": mark_as_read and was_unread,
            "labels_added": labels_added,
            "labels_removed": labels_removed
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps(_error(e, "gmail_read"))


@mcp.tool(
    title="Gmail Modify",
    description="""Modify email labels (add/remove labels, mark as read/unread, star/unstar).

Args:
- message_id: Email message ID (required)
- user_email: Email address to access (required for service account delegation)
- add_labels: List of label names or IDs to add (optional). Use "STARRED", "IMPORTANT", or custom label names/IDs
- remove_labels: List of label names or IDs to remove (optional). Use "UNREAD" to mark as read, "STARRED" to unstar
- mark_as_read: Shortcut to mark as read (removes UNREAD label) (default: False)
- mark_as_unread: Shortcut to mark as unread (adds UNREAD label) (default: False)
- star: Shortcut to star the email (default: False)
- unstar: Shortcut to unstar the email (default: False)

Returns: {message_id, labels_added, labels_removed, status}"""
)
def gmail_modify(
    message_id: str,
    user_email: Optional[str] = None,
    add_labels: Optional[List[str]] = None,
    remove_labels: Optional[List[str]] = None,
    mark_as_read: bool = False,
    mark_as_unread: bool = False,
    star: bool = False,
    unstar: bool = False
) -> str:
    try:
        gmail = _get_service('gmail', user_email)

        # Get all labels to map names to IDs
        all_labels = gmail.users().labels().list(userId='me').execute()
        label_map = {}  # name -> id
        for label in all_labels.get('labels', []):
            label_map[label['name'].lower()] = label['id']
            label_map[label['id'].lower()] = label['id']  # Also map ID to itself

        def resolve_label(label_input):
            """Convert label name to ID if needed."""
            # System labels are already valid IDs
            system_labels = ['INBOX', 'SENT', 'DRAFT', 'SPAM', 'TRASH', 'STARRED', 'IMPORTANT', 'UNREAD', 'CATEGORY_PERSONAL', 'CATEGORY_SOCIAL', 'CATEGORY_PROMOTIONS', 'CATEGORY_UPDATES', 'CATEGORY_FORUMS']
            if label_input.upper() in system_labels:
                return label_input.upper()
            # Look up in label map
            label_lower = label_input.lower()
            if label_lower in label_map:
                return label_map[label_lower]
            # Return as-is (might be a valid ID)
            return label_input

        # Build label modifications
        labels_to_add = [resolve_label(l) for l in add_labels] if add_labels else []
        labels_to_remove = [resolve_label(l) for l in remove_labels] if remove_labels else []

        # Apply shortcuts
        if mark_as_read:
            labels_to_remove.append('UNREAD')
        if mark_as_unread:
            labels_to_add.append('UNREAD')
        if star:
            labels_to_add.append('STARRED')
        if unstar:
            labels_to_remove.append('STARRED')

        if not labels_to_add and not labels_to_remove:
            return json.dumps({
                "error": "No label modifications specified",
                "status": "failed"
            })

        modify_body = {}
        if labels_to_add:
            modify_body['addLabelIds'] = labels_to_add
        if labels_to_remove:
            modify_body['removeLabelIds'] = labels_to_remove

        gmail.users().messages().modify(
            userId='me',
            id=message_id,
            body=modify_body
        ).execute()

        return json.dumps({
            "message_id": message_id,
            "labels_added": labels_to_add,
            "labels_removed": labels_to_remove,
            "status": "modified"
        })

    except Exception as e:
        return json.dumps(_error(e, "gmail_modify"))


@mcp.tool(
    title="Gmail Send",
    description="""Compose and send an email, optionally with attachments.

Args:
- to: Recipient email address(es), comma-separated (required)
- subject: Email subject (required)
- body: Email body text (required)
- user_email: Sender email address (required for service account delegation)
- cc: CC recipients, comma-separated (optional)
- bcc: BCC recipients, comma-separated (optional)
- html: If True, body is HTML; if False, plain text (default: False)
- attachments: List of file paths to attach (optional)
- reply_to_id: Message ID to reply to (optional)

Returns: {message_id, thread_id, status}"""
)
def gmail_send(
    to: str,
    subject: str,
    body: str,
    user_email: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    html: bool = False,
    attachments: Optional[List[str]] = None,
    reply_to_id: Optional[str] = None
) -> str:
    try:
        gmail = _get_service('gmail', user_email)
        sender = user_email or 'me'

        # Create message
        if attachments:
            message = MIMEMultipart()
            message.attach(MIMEText(body, 'html' if html else 'plain'))
        else:
            message = MIMEText(body, 'html' if html else 'plain')

        message['to'] = to
        message['subject'] = subject
        if cc:
            message['cc'] = cc
        if bcc:
            message['bcc'] = bcc

        # Handle reply
        thread_id = None
        if reply_to_id:
            original = gmail.users().messages().get(userId='me', id=reply_to_id, format='metadata',
                                                     metadataHeaders=['Message-ID', 'Subject']).execute()
            orig_headers = {h['name']: h['value'] for h in original.get('payload', {}).get('headers', [])}
            message['In-Reply-To'] = orig_headers.get('Message-ID', '')
            message['References'] = orig_headers.get('Message-ID', '')
            thread_id = original.get('threadId')

        # Add attachments
        if attachments:
            for file_path in attachments:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    mime_type, _ = mimetypes.guess_type(file_path)
                    mime_type = mime_type or 'application/octet-stream'
                    main_type, sub_type = mime_type.split('/', 1)

                    with open(file_path, 'rb') as f:
                        attachment = MIMEBase(main_type, sub_type)
                        attachment.set_payload(f.read())
                        encoders.encode_base64(attachment)
                        attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                        message.attach(attachment)

        # Encode and send
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        send_body = {'raw': raw}
        if thread_id:
            send_body['threadId'] = thread_id

        result = gmail.users().messages().send(userId='me', body=send_body).execute()

        return json.dumps({
            "message_id": result['id'],
            "thread_id": result.get('threadId', ''),
            "to": to,
            "subject": subject,
            "attachments_count": len(attachments) if attachments else 0,
            "status": "sent"
        })
    except Exception as e:
        return json.dumps(_error(e, "gmail_send"))


@mcp.tool(
    title="Gmail Attachment",
    description="""Download an email attachment or list attachments.

Args:
- message_id: Email message ID (required)
- user_email: Email address to access (required for service account delegation)
- attachment_id: Specific attachment ID to download (optional, lists all if not provided)
- save_path: Directory to save attachment (optional, returns base64 if not provided)

Returns: If listing: {attachments: [{id, filename, mime_type, size}]}
         If downloading: {filename, mime_type, size, saved_to} or {filename, mime_type, size, data_base64}"""
)
def gmail_attachment(
    message_id: str,
    user_email: Optional[str] = None,
    attachment_id: Optional[str] = None,
    save_path: Optional[str] = None
) -> str:
    try:
        gmail = _get_service('gmail', user_email)

        # Get message to find attachments
        msg = gmail.users().messages().get(userId='me', id=message_id, format='full').execute()

        # Find all attachments
        attachments = []
        def find_attachments(payload, attachments_list):
            if 'parts' in payload:
                for part in payload['parts']:
                    if part.get('filename') and part['body'].get('attachmentId'):
                        attachments_list.append({
                            "id": part['body']['attachmentId'],
                            "filename": part['filename'],
                            "mime_type": part['mimeType'],
                            "size": part['body'].get('size', 0)
                        })
                    if 'parts' in part:
                        find_attachments(part, attachments_list)

        find_attachments(msg.get('payload', {}), attachments)

        # If no attachment_id, just list them
        if not attachment_id:
            return json.dumps({
                "message_id": message_id,
                "attachments": attachments,
                "count": len(attachments)
            }, indent=2)

        # Find the specific attachment
        target = next((a for a in attachments if a['id'] == attachment_id), None)
        if not target:
            return json.dumps({"error": f"Attachment {attachment_id} not found", "status": "not_found"})

        # Download the attachment
        attachment_data = gmail.users().messages().attachments().get(
            userId='me',
            messageId=message_id,
            id=attachment_id
        ).execute()

        file_data = base64.urlsafe_b64decode(attachment_data['data'])

        if save_path:
            # Save to file (confined to DATA_PATH)
            save_path = _resolve_save_path(save_path)
            _secure_makedirs(save_path)
            file_path = os.path.join(save_path, target['filename'])
            fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'wb') as f:
                f.write(file_data)
            return json.dumps({
                "filename": target['filename'],
                "mime_type": target['mime_type'],
                "size": len(file_data),
                "saved_to": file_path,
                "status": "downloaded"
            })
        else:
            # Return as base64
            return json.dumps({
                "filename": target['filename'],
                "mime_type": target['mime_type'],
                "size": len(file_data),
                "data_base64": base64.b64encode(file_data).decode('utf-8'),
                "status": "retrieved"
            })

    except Exception as e:
        return json.dumps(_error(e, "gmail_attachment"))


@mcp.tool(
    title="Gmail Create Label",
    description="""Create a new label in Gmail, or apply existing labels to messages.

Args:
- name: Label name to create (required for creating new label)
- user_email: Email address to access (required for service account delegation)
- message_id: Message ID to apply label to (optional)
- label_ids: List of label IDs to apply to the message (optional, use with message_id)
- label_list_visibility: "labelShow", "labelShowIfUnread", "labelHide" (default: "labelShow")
- message_list_visibility: "show", "hide" (default: "show")

Returns: {label_id, name, status} or {message_id, labels, status}"""
)
def gmail_create_label(
    name: Optional[str] = None,
    user_email: Optional[str] = None,
    message_id: Optional[str] = None,
    label_ids: Optional[List[str]] = None,
    label_list_visibility: str = "labelShow",
    message_list_visibility: str = "show"
) -> str:
    try:
        gmail = _get_service('gmail', user_email)

        # If message_id and label_ids provided, apply labels to message
        if message_id and label_ids:
            gmail.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': label_ids}
            ).execute()
            return json.dumps({
                "message_id": message_id,
                "labels_applied": label_ids,
                "status": "labels_applied"
            })

        # Create new label
        if not name:
            return json.dumps({
                "error": "Label name is required when creating a new label",
                "status": "failed"
            })

        # Check if label already exists
        existing_labels = gmail.users().labels().list(userId='me').execute()
        for existing_label in existing_labels.get('labels', []):
            if existing_label['name'].lower() == name.lower():
                return json.dumps({
                    "label_id": existing_label['id'],
                    "name": existing_label['name'],
                    "type": existing_label.get('type', 'user'),
                    "status": "already_exists"
                })

        # Label doesn't exist, create it
        label_body = {
            'name': name,
            'labelListVisibility': label_list_visibility,
            'messageListVisibility': message_list_visibility
        }

        label = gmail.users().labels().create(
            userId='me',
            body=label_body
        ).execute()

        return json.dumps({
            "label_id": label['id'],
            "name": label['name'],
            "type": label.get('type', 'user'),
            "status": "created"
        })

    except Exception as e:
        return json.dumps(_error(e, "gmail_create_label"))


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 9006,
    path: str = "/workspace",
    options: dict = {}
) -> None:
    """Run the MCP server.

    Options:
        credentials_file: Path to service account JSON key
        delegated_user: Default email for Gmail domain-wide delegation
        verbose: Enable verbose logging
    """
    global _credentials_file, _delegated_user, DATA_PATH

    if 'verbose' in options:
        logger.setLevel(logging.INFO)

    if 'data_path' in options:
        DATA_PATH = options['data_path']
    _secure_makedirs(os.path.abspath(os.path.expanduser(DATA_PATH)))

    if 'credentials_file' in options:
        creds_path = os.path.expanduser(options['credentials_file'])
        if os.path.exists(creds_path):
            _credentials_file = creds_path
            logger.info(f"Using credentials: {_credentials_file}")

    if 'delegated_user' in options:
        _delegated_user = options['delegated_user']
        logger.info(f"Default delegated user for Gmail: {_delegated_user}")

    logger.info(f"Starting Google Workspace MCP Server at {host}:{port}{path}")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info("22 Tools: Auth(1), Drive(7), Docs(3), Sheets(3), Slides(3), Gmail(6)")
    mcp.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
