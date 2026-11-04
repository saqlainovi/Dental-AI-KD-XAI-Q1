import re
import html

md_path = r"outputs\DENTAL_AI_COMPREHENSIVE_REVIEW_PACKAGE.md"
with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

header = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dental AI Research: Comprehensive Review Package</title>
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #2c3e50; max-width: 1050px; margin: 40px auto; padding: 0 25px; background: #fafbfc; }
    h1 { color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 10px; font-size: 24px; }
    h2 { color: #2b6cb0; margin-top: 30px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; font-size: 19px; }
    h3 { color: #2d3748; font-size: 16px; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 6px; overflow: hidden; }
    th, td { padding: 11px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 13.5px; }
    th { background: #2b6cb0; color: #ffffff; font-weight: 600; }
    tr:hover { background: #f7fafc; }
    code { background: #edf2f7; color: #c53030; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 13px; }
    pre { padding: 15px; overflow-x: auto; color: #f7fafc; background: #2d3748; border-radius: 6px; font-family: monospace; font-size: 12.5px; }
    blockquote { border-left: 4px solid #3182ce; margin: 15px 0; padding: 12px 20px; background: #ebf8ff; color: #2b6cb0; border-radius: 0 6px 6px 0; }
    hr { border: 0; height: 1px; background: #e2e8f0; margin: 30px 0; }
</style>
</head>
<body>
"""

html_content = header
lines = text.split("\n")
in_table = False
in_pre = False

for line in lines:
    if line.startswith("```"):
        if in_pre:
            html_content += "</pre>\n"
            in_pre = False
        else:
            html_content += "<pre>"
            in_pre = True
        continue
    if in_pre:
        html_content += html.escape(line) + "\n"
        continue

    if line.startswith("# "):
        html_content += f"<h1>{html.escape(line[2:])}</h1>\n"
    elif line.startswith("## "):
        html_content += f"<h2>{html.escape(line[3:])}</h2>\n"
    elif line.startswith("### "):
        html_content += f"<h3>{html.escape(line[4:])}</h3>\n"
    elif line.startswith("> "):
        html_content += f"<blockquote>{html.escape(line[2:])}</blockquote>\n"
    elif line.startswith("---"):
        html_content += "<hr>\n"
    elif line.startswith("|") and line.endswith("|"):
        if "---" in line:
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not in_table:
            html_content += "<table>\n<thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead>\n<tbody>\n"
            in_table = True
        else:
            html_content += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>\n"
    else:
        if in_table:
            html_content += "</tbody></table>\n"
            in_table = False
        if line.strip():
            formatted = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)
            html_content += f"<p>{formatted}</p>\n"

if in_table:
    html_content += "</tbody></table>\n"
if in_pre:
    html_content += "</pre>\n"

html_content += "</body></html>"

out_html = r"outputs\DENTAL_AI_COMPREHENSIVE_REVIEW_PACKAGE.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_content)
print("HTML Review Package successfully written to:", out_html)
