#!/usr/bin/env python3
"""Fetch recent GitHub Discussions for this repo and bake them into the site.

Runs in CI (see .github/workflows/update-discussions.yml) — not in the browser.
The site remains fully static at request time; this script just keeps two
spots in the static HTML in sync with the live discussion board:

  - discussions.html : the 10 most recently updated discussions, any category.
  - index.html        : the 5 most recent posts in the "Announcements" category,
                         if that category exists. Falls back to the existing
                         "no announcements yet" copy if it doesn't (e.g. the
                         category hasn't been created on GitHub yet).
"""
import html
import json
import os
import re
import sys
import urllib.request

OWNER = "LUMS-WIT"
REPO = "climate-futures"
ANNOUNCEMENTS_CATEGORY_NAME = "Announcements"

DISCUSSIONS_FILE = "discussions.html"
DISCUSSIONS_START = "<!-- DISCUSSIONS:START -->"
DISCUSSIONS_END = "<!-- DISCUSSIONS:END -->"

ANNOUNCEMENTS_FILE = "index.html"
ANNOUNCEMENTS_START = "<!-- ANNOUNCEMENTS:START -->"
ANNOUNCEMENTS_END = "<!-- ANNOUNCEMENTS:END -->"

OVERVIEW_QUERY = """
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
    discussionCategories(first: 25) {
      nodes { id name }
    }
  }
}
"""

CATEGORY_DISCUSSIONS_QUERY = """
query($owner: String!, $repo: String!, $categoryId: ID!) {
  repository(owner: $owner, name: $repo) {
    discussions(first: 5, categoryId: $categoryId, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        title
        url
        createdAt
      }
    }
  }
}
"""


def graphql(token, query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
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
        data = json.load(resp)
    if "errors" in data:
        print(json.dumps(data["errors"]), file=sys.stderr)
        sys.exit(1)
    return data["data"]


def render_discussions_list(nodes):
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


def render_announcements(nodes):
    if not nodes:
        return (
            '<p class="empty-state">No announcements yet. Check back here for '
            "updates throughout the semester.</p>"
        )
    items = []
    for n in nodes:
        title = html.escape(n["title"])
        url = html.escape(n["url"], quote=True)
        posted = n["createdAt"][:10]
        items.append(
            f'<li><div class="r-title"><a href="{url}" target="_blank" rel="noopener">{title}</a></div>'
            f'<div class="r-meta">Posted {posted}</div></li>'
        )
    return '<ul class="resource-list">\n' + "\n".join(items) + "\n</ul>"


def replace_between_markers(path, start_marker, end_marker, new_inner_html):
    with open(path, encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL
    )
    if not pattern.search(content):
        print(f"Markers not found in {path}", file=sys.stderr)
        sys.exit(1)

    replacement = f"{start_marker}\n{new_inner_html}\n{end_marker}"
    new_content = pattern.sub(lambda m: replacement, content)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    overview = graphql(
        token, OVERVIEW_QUERY, {"owner": OWNER, "repo": REPO}
    )["repository"]

    discussions_html = render_discussions_list(overview["discussions"]["nodes"])
    replace_between_markers(
        DISCUSSIONS_FILE, DISCUSSIONS_START, DISCUSSIONS_END, discussions_html
    )

    category_id = next(
        (
            c["id"]
            for c in overview["discussionCategories"]["nodes"]
            if c["name"] == ANNOUNCEMENTS_CATEGORY_NAME
        ),
        None,
    )

    if category_id:
        cat_data = graphql(
            token,
            CATEGORY_DISCUSSIONS_QUERY,
            {"owner": OWNER, "repo": REPO, "categoryId": category_id},
        )
        announcements_html = render_announcements(
            cat_data["repository"]["discussions"]["nodes"]
        )
    else:
        announcements_html = render_announcements([])

    replace_between_markers(
        ANNOUNCEMENTS_FILE, ANNOUNCEMENTS_START, ANNOUNCEMENTS_END, announcements_html
    )


if __name__ == "__main__":
    main()
