#!/usr/bin/env python3
"""Fetch recent GitHub Discussions for this repo and bake them into discussions.html.

Runs in CI (see .github/workflows/update-discussions.yml) — not in the browser.
The site remains fully static at request time; this script just keeps the
static HTML in sync with the live discussion board.
"""
import html
import json
import os
import re
import sys
import urllib.request

OWNER = "LUMS-WIT"
REPO = "climate-futures"
TARGET_FILE = "discussions.html"
START_MARKER = "<!-- DISCUSSIONS:START -->"
END_MARKER = "<!-- DISCUSSIONS:END -->"

QUERY = """
query($owner: String!, $repo: String!) {
  repository(owner: $owner, name: $repo) {
    discussions(first: 10, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        title
        url
        updatedAt
        category { name }
        comments { totalCount }
      }
    }
  }
}
"""


def fetch(token):
    body = json.dumps(
        {"query": QUERY, "variables": {"owner": OWNER, "repo": REPO}}
    ).encode()
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "climate-futures-discussions-bot",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def render(nodes):
    if not nodes:
        return (
            '<p class="text-secondary">No discussions have been posted yet. '
            "Be the first to start one on GitHub.</p>"
        )
    items = []
    for n in nodes:
        title = html.escape(n["title"])
        url = html.escape(n["url"], quote=True)
        category = html.escape(n["category"]["name"]) if n.get("category") else ""
        count = n["comments"]["totalCount"]
        reply_word = "reply" if count == 1 else "replies"
        updated = n["updatedAt"][:10]
        items.append(
            f'<li><div class="r-title"><a href="{url}" target="_blank" rel="noopener">{title}</a></div>'
            f'<div class="r-meta">{category} · {count} {reply_word} · updated {updated}</div></li>'
        )
    return '<ul class="resource-list">\n' + "\n".join(items) + "\n</ul>"


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    data = fetch(token)
    if "errors" in data:
        print(json.dumps(data["errors"]), file=sys.stderr)
        sys.exit(1)

    nodes = data["data"]["repository"]["discussions"]["nodes"]
    list_html = render(nodes)

    with open(TARGET_FILE, encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL
    )
    if not pattern.search(content):
        print(f"Markers not found in {TARGET_FILE}", file=sys.stderr)
        sys.exit(1)

    replacement = f"{START_MARKER}\n{list_html}\n{END_MARKER}"
    new_content = pattern.sub(lambda m: replacement, content)

    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)


if __name__ == "__main__":
    main()
