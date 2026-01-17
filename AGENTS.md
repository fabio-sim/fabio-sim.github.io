# Repository Guidelines
- This is a GitHub Pages site built on the Minimal Mistakes Jekyll theme.

## Project Structure & Module Organization
- Site configuration lives in `_config.yml`.
- Content is split into `_posts/` (blog posts) and `_pages/` (standalone pages). Posts follow `YYYY-MM-DD-title.md` naming.
- Theme structure: `_layouts/` (page templates), `_includes/` (partials), `_sass/` (theme SCSS).
- Assets live in `assets/`: `assets/css/main.scss`, `assets/js/_main.js`, compiled `assets/js/main.min.js`, and images under `assets/images/`.
- Build output is `_site/` (generated). Do not edit or commit it directly.

## Build, Test, and Development Commands
- `bundle install`: install Ruby dependencies for Jekyll/GitHub Pages.
- `bundle exec jekyll serve`: run the local site with live rebuilds.
- `bundle exec jekyll build`: generate the production site into `_site/`.
- `rake js`: rebuild `assets/js/main.min.js` after editing `assets/js/_main.js` or plugin JS.
- `rake watch_js`: watch JS sources and re-minify on changes.

## Coding Style & Naming Conventions
- Use Markdown with YAML front matter; keep YAML indented with two spaces.
- Prefer kebab-case file names (e.g., `my-new-post.md`).
- Do not hand-edit `assets/js/main.min.js`; always regenerate via `rake js`.
- Keep SCSS changes in `assets/css/main.scss` and partials under `_sass/`.

## Testing Guidelines
- Validate changes by running `bundle exec jekyll build` and reviewing the output in `_site/` or via `bundle exec jekyll serve`.

## Commit & Pull Request Guidelines
- Commit messages use Conventional Commits with scopes (e.g., `fix(mobile): …`).
- PRs should include a short summary and relevant context/links.

## Security & Configuration Tips
- Don’t commit secrets or API keys; prefer local environment configuration when needed.
- Treat `_config.yml` as public configuration files and review changes carefully.
