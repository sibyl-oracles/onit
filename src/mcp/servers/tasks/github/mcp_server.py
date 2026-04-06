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

GitHub MCP Server - Repository Operations

Create and manage GitHub repositories via the GitHub REST API.

Requires:
    pip install requests

    Environment variables:
    - GITHUB_TOKEN: GitHub personal access token (classic or fine-grained)
      Needs scopes: repo (for private repos) or public_repo (for public repos)
      For org repos: also needs admin:org or the repo scope.

1 Core Tool:
1. github_repo - Create, get, list, fork, or delete GitHub repositories
'''

import json
import os
from typing import Optional

import requests
from fastmcp import FastMCP

from src.mcp.servers.tasks.shared import validate_required as _validate_required

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("GitHub MCP Server")

GITHUB_API = "https://api.github.com"


def _get_token() -> Optional[str]:
    """Resolve GitHub token: env var > keyring."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        from src.setup import get_secret
        return get_secret("github_token")
    except Exception:
        pass
    return None


def _auth_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _repo_summary(repo: dict) -> dict:
    """Extract key fields from a GitHub repo response object."""
    return {
        "name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "url": repo.get("html_url"),
        "clone_url": repo.get("clone_url"),
        "ssh_url": repo.get("ssh_url"),
        "description": repo.get("description"),
        "private": repo.get("private"),
        "default_branch": repo.get("default_branch"),
        "created_at": repo.get("created_at"),
        "owner": repo.get("owner", {}).get("login"),
    }


@mcp.tool(
    title="GitHub Repository",
    description="""Create, get, list, fork, or delete GitHub repositories via the GitHub API.

Requires GITHUB_TOKEN environment variable (personal access token with repo scope).

Actions:
- create : Create a new repository (user or org). Returns repo details.
- get    : Get info about an existing repository.
- list   : List repositories for the authenticated user or an org.
- fork   : Fork an existing repository into the authenticated user's account or an org.
- delete : Delete a repository (requires admin access).

Args:
- action      : One of "create", "get", "list", "fork", "delete" (required)
- name        : Repository name — required for create, get, fork (owner/repo), delete (owner/repo)
- description : Repository description (create only, optional)
- private     : Make repo private (create only, default: false)
- auto_init   : Initialize with a README (create only, default: true)
- gitignore_template : e.g. "Python", "Node" (create only, optional)
- license_template   : e.g. "mit", "apache-2.0" (create only, optional)
- org         : Organization name — if set for create/list, targets the org instead of the user
- per_page    : Results per page for list (default: 30, max: 100)

Returns JSON: repo details for create/get/fork; list of repos for list; status for delete."""
)
def github_repo(
    action: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    private: bool = False,
    auto_init: bool = True,
    gitignore_template: Optional[str] = None,
    license_template: Optional[str] = None,
    org: Optional[str] = None,
    per_page: int = 30,
) -> str:
    if err := _validate_required(action=action):
        return err

    token = _get_token()
    if not token:
        return json.dumps({
            "error": "GitHub token not found. Set GITHUB_TOKEN env var or run 'onit setup'.",
            "status": "error",
        })

    headers = _auth_headers(token)

    try:
        if action == "create":
            if err := _validate_required(name=name):
                return err

            payload: dict = {
                "name": name,
                "private": private,
                "auto_init": auto_init,
            }
            if description:
                payload["description"] = description
            if gitignore_template:
                payload["gitignore_template"] = gitignore_template
            if license_template:
                payload["license_template"] = license_template

            if org:
                url = f"{GITHUB_API}/orgs/{org}/repos"
            else:
                url = f"{GITHUB_API}/user/repos"

            resp = requests.post(url, headers=headers, json=payload, timeout=30)

            if resp.status_code == 201:
                repo = resp.json()
                return json.dumps({
                    "repo": _repo_summary(repo),
                    "status": "created",
                })
            else:
                return json.dumps({
                    "error": resp.json().get("message", resp.text),
                    "errors": resp.json().get("errors"),
                    "status": "error",
                    "http_status": resp.status_code,
                })

        elif action == "get":
            if err := _validate_required(name=name):
                return err
            # name should be "owner/repo"
            url = f"{GITHUB_API}/repos/{name}"
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return json.dumps({
                    "repo": _repo_summary(resp.json()),
                    "status": "ok",
                })
            else:
                return json.dumps({
                    "error": resp.json().get("message", resp.text),
                    "status": "error",
                    "http_status": resp.status_code,
                })

        elif action == "list":
            if org:
                url = f"{GITHUB_API}/orgs/{org}/repos"
            else:
                url = f"{GITHUB_API}/user/repos"
            resp = requests.get(url, headers=headers, params={"per_page": per_page}, timeout=30)
            if resp.status_code == 200:
                repos = [_repo_summary(r) for r in resp.json()]
                return json.dumps({
                    "repos": repos,
                    "count": len(repos),
                    "status": "ok",
                })
            else:
                return json.dumps({
                    "error": resp.json().get("message", resp.text),
                    "status": "error",
                    "http_status": resp.status_code,
                })

        elif action == "fork":
            if err := _validate_required(name=name):
                return err
            # name should be "owner/repo"
            url = f"{GITHUB_API}/repos/{name}/forks"
            payload = {}
            if org:
                payload["organization"] = org
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code in (202, 200):
                return json.dumps({
                    "repo": _repo_summary(resp.json()),
                    "status": "forked",
                })
            else:
                return json.dumps({
                    "error": resp.json().get("message", resp.text),
                    "status": "error",
                    "http_status": resp.status_code,
                })

        elif action == "delete":
            if err := _validate_required(name=name):
                return err
            # name should be "owner/repo"
            url = f"{GITHUB_API}/repos/{name}"
            resp = requests.delete(url, headers=headers, timeout=30)
            if resp.status_code == 204:
                return json.dumps({"message": f"Repository '{name}' deleted.", "status": "deleted"})
            else:
                body = resp.json() if resp.content else {}
                return json.dumps({
                    "error": body.get("message", resp.text),
                    "status": "error",
                    "http_status": resp.status_code,
                })

        else:
            return json.dumps({
                "error": f"Unknown action '{action}'. Use: create, get, list, fork, delete",
                "status": "error",
            })

    except requests.RequestException as e:
        return json.dumps({"error": str(e), "status": "error"})


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18204,
    path: str = "/sse",
    options: dict = {},
) -> None:
    """Run the GitHub MCP server."""
    verbose = "verbose" in options
    level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(level)

    logger.info(f"Starting GitHub MCP Server at {host}:{port}{path}")

    if not verbose:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    mcp.run(transport=transport, host=host, port=port, path=path,
            uvicorn_config={"access_log": False, "log_level": "warning"} if not verbose else {})


if __name__ == "__main__":
    run()
