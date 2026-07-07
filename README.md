# Climate Change Governance: Science, Data & Models

Static course website for **ENGG562/ENV462 — Climate Change Governance: Science, Data & Models**, LUMS, Fall 2025–26.

Instructor: Muhammad Awais (awais.m@lums.edu.pk), Centre for Water Informatics & Technology, Syed Babar Ali School of Science & Engineering.

## Structure

```
/
├── index.html          Home
├── syllabus.html        Course description, CLOs, grading, policies
├── schedule.html        14-week / 7-module schedule
├── assignments.html     Assignment stub table
├── resources.html        Textbooks, discussion papers, tools & datasets
├── project.html          Capstone project page
├── css/
│   └── style.css         All site styles
└── README.md
```

## Tech

Plain HTML + CSS. No build step, no JavaScript, no frameworks. Fonts loaded from Google Fonts (Inter) via `<link>`.

## Local preview

Open `index.html` directly in a browser, or serve the folder locally:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Deploying to GitHub Pages

1. Push this repository to GitHub.
2. In the repo settings, go to **Pages**.
3. Set source to the `main` branch, root (`/`) folder.
4. Save — the site will publish at `https://<username>.github.io/<repo>/`.

No build configuration is required.

## Editing content

All pages share the same nav/footer markup and `css/style.css`. Update placeholder content (slides, readings, assignment PDFs, tool links) by replacing the `#` placeholders and placeholder text as the semester progresses.
