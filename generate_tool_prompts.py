#!/usr/bin/env python3
"""Generate tool-calling and agent prompts for Echo distillation.

Produces prompts that teach Echo to:
- Call tools (file ops, web search, API calls, database queries)
- Act as an agent (multi-step reasoning, planning, execution)
- Write code that interacts with external systems
- Handle tool results and chain actions
"""

import random
import sys

# === TOOL CALLING SCENARIOS ===
TOOL_SCENARIOS = [
    # File operations
    ("Read the file {file} and tell me what's inside.",
     "file: read {file}\ntool_result: {file_content}\necho: {file_summary}"),
    ("Write {content} to {file}.",
     "file: write {file} {content}\ntool_result: written to {file}\necho: I've written {content} to {file}."),
    ("List all files in the {dir} directory.",
     "file: list {dir}\ntool_result: {file_list}\necho: Here are the files in {dir}: {file_list}"),
    ("Delete the file {file}.",
     "file: delete {file}\ntool_result: deleted {file}\necho: I've deleted {file}."),
    ("Move {file} to {dest}.",
     "file: move {file} {dest}\ntool_result: moved {file} to {dest}\necho: Done. I moved {file} to {dest}."),
    ("Copy {file} to {dest}.",
     "file: copy {file} {dest}\ntool_result: copied {file} to {dest}\necho: I copied {file} to {dest}."),
    ("Create a new directory called {dir}.",
     "file: mkdir {dir}\ntool_result: created {dir}\necho: I created the directory {dir}."),
    ("Search for files containing '{query}' in {dir}.",
     "file: search {dir} '{query}'\ntool_result: {search_results}\necho: I found {search_results} matching '{query}' in {dir}."),
    
    # Web/API operations
    ("Search the web for {query}.",
     "web: search '{query}'\ntool_result: {web_results}\necho: Here's what I found about {query}: {web_results}"),
    ("Fetch the content from {url}.",
     "web: fetch {url}\ntool_result: {fetch_content}\necho: I fetched the content from {url}. Here's a summary: {fetch_summary}"),
    ("Call the API endpoint {endpoint} with {params}.",
     "api: call {endpoint} {params}\ntool_result: {api_response}\necho: The API returned: {api_response}"),
    ("Send a POST request to {endpoint} with {data}.",
     "api: post {endpoint} {data}\ntool_result: {api_post_response}\necho: The POST request returned: {api_post_response}"),
    ("Download {file} from {url}.",
     "web: download {url} {file}\ntool_result: downloaded {file}\necho: I downloaded {file} from {url}."),
    ("Upload {file} to {service}.",
     "web: upload {file} {service}\ntool_result: uploaded {file} to {service}\necho: I uploaded {file} to {service}."),
    
    # Database operations
    ("Query the database for {query}.",
     "db: query '{query}'\ntool_result: {db_results}\necho: The database query returned: {db_results}"),
    ("Insert {data} into the {table} table.",
     "db: insert {table} {data}\ntool_result: inserted into {table}\necho: I inserted the data into {table}."),
    ("Update {table} where {condition} to {value}.",
     "db: update {table} {condition} {value}\ntool_result: updated {table}\necho: I updated {table} where {condition}."),
    ("Delete from {table} where {condition}.",
     "db: delete {table} {condition}\ntool_result: deleted from {table}\necho: I deleted the matching records from {table}."),
    
    # Git operations
    ("Check the git status.",
     "git: status\ntool_result: {git_status}\necho: Git shows: {git_status}"),
    ("Commit changes with the message '{message}'.",
     "git: commit '{message}'\ntool_result: committed: {message}\necho: I committed the changes with message: {message}"),
    ("Create a new branch called {branch}.",
     "git: branch {branch}\ntool_result: created branch {branch}\necho: I created a new branch called {branch}."),
    ("Merge {branch} into main.",
     "git: merge {branch}\ntool_result: merged {branch} into main\necho: I merged {branch} into main."),
    ("Show the recent git log.",
     "git: log\ntool_result: {git_log}\necho: Here are the recent commits: {git_log}"),
    
    # System operations
    ("Run the command '{command}' and show the output.",
     "shell: run '{command}'\ntool_result: {command_output}\necho: The command returned: {command_output}"),
    ("Check the system resources.",
     "system: resources\ntool_result: CPU: {cpu}%, RAM: {ram}%, Disk: {disk}%\necho: System resources — CPU at {cpu}%, RAM at {ram}%, Disk at {disk}%."),
    ("Start the {service} service.",
     "system: start {service}\ntool_result: {service} started\necho: I started the {service} service."),
    ("Stop the {service} service.",
     "system: stop {service}\ntool_result: {service} stopped\necho: I stopped the {service} service."),
    ("Check the logs of {service} for errors.",
     "system: logs {service}\ntool_result: {log_content}\necho: I found these entries in the {service} logs: {log_content}"),
    
    # Communication
    ("Send an email to {recipient} about {subject}.",
     "email: send {recipient} '{subject}'\ntool_result: email sent to {recipient}\necho: I sent an email to {recipient} about {subject}."),
    ("Post a message on {platform} saying '{message}'.",
     "social: post {platform} '{message}'\ntool_result: posted to {platform}\necho: I posted your message to {platform}."),
    ("Search for tweets about {topic}.",
     "social: search {platform} '{topic}'\ntool_result: {tweet_results}\necho: I found tweets about {topic}: {tweet_results}"),
    ("Create a calendar event for {event} on {date}.",
     "calendar: create '{event}' {date}\ntool_result: event created: {event} on {date}\necho: I created a calendar event for {event} on {date}."),
    ("Set a reminder for {task} at {time}.",
     "reminder: set '{task}' {time}\ntool_result: reminder set for {task} at {time}\necho: I'll remind you about {task} at {time}."),
    
    # Memory operations
    ("Remember that my {key} is {value}.",
     "memory: store {key} {value}\ntool_result: saved: {key}={value}\necho: I'll remember that your {key} is {value}."),
    ("What is my {key}?",
     "memory: recall {key}\ntool_result: {value}\necho: Your {key} is {value}."),
    ("Forget my {key}.",
     "memory: delete {key}\ntool_result: forgot: {key}\necho: I've forgotten your {key}."),
]

# === AGENT SCENARIOS (multi-step) ===
AGENT_SCENARIOS = [
    # Planning and execution
    ("I need to deploy a web application. What steps should I take?",
     "think: The user wants to deploy a web app. I need to check the project structure, build it, and deploy.\nplan: 1) Check project files 2) Build the application 3) Deploy to server\nstep 1: file: list .\ntool_result: index.html, app.js, package.json, Dockerfile\nstep 2: shell: run 'npm run build'\ntool_result: Build successful. Output in dist/\nstep 3: shell: run 'docker build -t myapp .'\ntool_result: Image built successfully\nstep 4: shell: run 'docker run -p 8080:80 myapp'\ntool_result: Container running on port 8080\necho: Your web application is now deployed and running on port 8080. I built the project, created a Docker image, and started the container."),
    
    ("Help me set up a Python development environment.",
     "think: The user needs a Python dev environment. I should check what's installed, create a virtual environment, and install common packages.\nplan: 1) Check Python version 2) Create venv 3) Install packages 4) Verify\nstep 1: shell: run 'python3 --version'\ntool_result: Python 3.11.5\nstep 2: shell: run 'python3 -m venv venv'\ntool_result: venv created\nstep 3: shell: run 'source venv/bin/activate && pip install requests flask numpy pandas'\ntool_result: All packages installed successfully\nstep 4: shell: run 'python -c \"import requests; print(requests.__version__)\"'\ntool_result: 2.31.0\necho: Your Python development environment is ready. I created a virtual environment with Python 3.11.5 and installed requests, flask, numpy, and pandas. Activate it with 'source venv/bin/activate'."),
    
    ("Find and fix a bug in my code.",
     "think: The user has a bug. I need to look at the code, identify the issue, and fix it.\nplan: 1) Read the source file 2) Identify the bug 3) Fix it 4) Test\nstep 1: file: read main.py\ntool_result: def divide(a, b): return a / b  # No zero check\nstep 2: think: The divide function doesn't handle division by zero\nstep 3: file: write main.py 'def divide(a, b):\\n    if b == 0:\\n        raise ValueError(\"Cannot divide by zero\")\\n    return a / b'\ntool_result: written to main.py\nstep 4: shell: run 'python -c \"from main import divide; print(divide(10, 2))\"'\ntool_result: 5.0\necho: I found and fixed the bug. The divide function didn't handle division by zero. I added a check that raises a ValueError when b is 0. The fix is verified and working."),
    
    ("Analyze a dataset and give me insights.",
     "think: The user wants data analysis. I need to load the data, examine it, compute statistics, and provide insights.\nplan: 1) Load the data 2) Check structure 3) Compute statistics 4) Find patterns 5) Report\nstep 1: shell: run 'python3 -c \"import pandas as pd; df = pd.read_csv(\\\"data.csv\\\"); print(df.shape)\"'\ntool_result: (1000, 5)\nstep 2: shell: run 'python3 -c \"import pandas as pd; df = pd.read_csv(\\\"data.csv\\\"); print(df.columns.tolist())\"'\ntool_result: ['date', 'product', 'quantity', 'price', 'total']\nstep 3: shell: run 'python3 -c \"import pandas as pd; df = pd.read_csv(\\\"data.csv\\\"); print(df.describe())\"'\ntool_result: quantity: mean=5.2, std=2.1; price: mean=$24.50, std=$8.30; total: mean=$127.40\nstep 4: shell: run 'python3 -c \"import pandas as pd; df = pd.read_csv(\\\"data.csv\\\"); print(df.groupby(\\\"product\\\")[\\\"total\\\"].sum().sort_values(ascending=False).head(5))\"'\ntool_result: Product A: $12,450, Product B: $8,200, Product C: $5,100\nstep 5: think: The dataset has 1000 sales records. Product A is the top seller. Average order value is $127.40.\necho: I analyzed your dataset. Key insights: 1) 1000 sales records across 5 columns. 2) Product A is your top seller at $12,450 total. 3) Average order value is $127.40. 4) Quantity per order averages 5.2 units. I recommend focusing marketing on Product A and investigating why Product C underperforms."),
    
    ("Monitor a server and alert me if something is wrong.",
     "think: The user wants server monitoring. I need to check CPU, memory, disk, and running services.\nplan: 1) Check CPU usage 2) Check memory 3) Check disk 4) Check services 5) Report status\nstep 1: shell: run 'top -bn1 | head -5'\ntool_result: CPU: 45%, load average: 1.2\nstep 2: shell: run 'free -h'\ntool_result: RAM: 8GB total, 4.2GB used, 3.8GB free\nstep 3: shell: run 'df -h /'\ntool_result: Disk: 50GB total, 35GB used, 15GB free (70%)\nstep 4: shell: run 'systemctl list-units --type=service --state=running'\ntool_result: nginx: active, postgresql: active, redis: active\nstep 5: think: CPU at 45% is moderate. RAM usage is 52%. Disk at 70% needs attention. All services running.\necho: Server status report: CPU at 45% (normal), RAM at 52% (healthy), Disk at 70% (warning — consider cleanup). All services (nginx, postgresql, redis) are running. I recommend cleaning up old log files to free disk space."),
    
    ("Help me write a REST API for a todo app.",
     "think: The user wants a REST API for todos. I need to create the server, define routes, add CRUD operations, and test.\nplan: 1) Create the server file 2) Define routes 3) Add CRUD 4) Test endpoints\nstep 1: file: write app.py 'from flask import Flask, jsonify, request\\napp = Flask(__name__)\\ntodos = []\\n@app.route(\"/todos\", methods=[\"GET\"])\\ndef get_todos(): return jsonify(todos)\\n@app.route(\"/todos\", methods=[\"POST\"])\\ndef add_todo(): todos.append(request.json); return jsonify(todos[-1]), 201\\n@app.route(\"/todos/<int:id>\", methods=[\"PUT\"])\\ndef update_todo(id): todos[id] = request.json; return jsonify(todos[id])\\n@app.route(\"/todos/<int:id>\", methods=[\"DELETE\"])\\ndef delete_todo(id): todos.pop(id); return \"\", 204'\ntool_result: written to app.py\nstep 2: shell: run 'pip install flask'\ntool_result: Flask installed\nstep 3: shell: run 'python app.py &'\ntool_result: Server running on port 5000\nstep 4: shell: run 'curl -X POST http://localhost:5000/todos -H \"Content-Type: application/json\" -d \\'{\"task\": \"Buy groceries\"}\\''\ntool_result: {\"task\": \"Buy groceries\"}\necho: I created a REST API for your todo app. It has GET, POST, PUT, and DELETE endpoints at /todos. The server is running on port 5000 and I tested it by adding a todo item. You can start using it right away."),
    
    ("I need to back up my database before making changes.",
     "think: The user wants a database backup. I need to check what database is running, create a backup, and verify it.\nplan: 1) Check database status 2) Create backup 3) Verify backup 4) Report\nstep 1: shell: run 'pg_dump --version'\ntool_result: pg_dump (PostgreSQL) 14.2\nstep 2: shell: run 'pg_dump -U postgres mydb > backup_$(date +%Y%m%d).sql'\ntool_result: backup created: backup_20240115.sql (45MB)\nstep 3: shell: run 'ls -lh backup_*.sql'\ntool_result: backup_20240115.sql: 45MB\nstep 4: think: Backup is 45MB, created successfully. The file is in the current directory.\necho: Database backup complete. I created backup_20240115.sql (45MB) using pg_dump. The backup is in the current directory. You can restore it with: psql -U postgres mydb < backup_20240115.sql"),
    
    ("Set up monitoring alerts for my application.",
     "think: The user wants monitoring alerts. I need to check what monitoring tools are available, configure alerts, and test them.\nplan: 1) Check available tools 2) Configure alert rules 3) Test alert 4) Report\nstep 1: shell: run 'which prometheus || echo not found'\ntool_result: /usr/local/bin/prometheus\nstep 2: file: write alerts.yml 'groups:\\n- name: app_alerts\\n  rules:\\n  - alert: HighErrorRate\\n    expr: rate(http_requests_total{status=~\"5..\"}[5m]) > 0.1\\n    for: 5m\\n    labels: severity: critical\\n    annotations: summary: \"High error rate detected\"'\ntool_result: written to alerts.yml\nstep 3: shell: run 'promtool check rules alerts.yml'\ntool_result: SUCCESS: 1 rules found\nstep 4: think: Prometheus is installed, alert rules are configured and validated.\necho: Monitoring alerts are set up. I configured Prometheus with a HighErrorRate alert that triggers when 5xx errors exceed 0.1/sec for 5 minutes. The rules file is alerts.yml and it passed validation. You can load it in Prometheus to start monitoring."),
]

# === CODE GENERATION WITH TOOLS ===
CODE_TOOL_SCENARIOS = [
    "Write a Python script that reads a CSV file, filters rows where the value in the 'status' column is 'active', and saves the result to a new file. Use the pandas library.",
    "Create a Python function that calls a REST API endpoint, parses the JSON response, and extracts specific fields. Include error handling for network failures.",
    "Write a Python script that connects to a PostgreSQL database, executes a query, and prints the results. Use psycopg2.",
    "Create a Python function that sends an email using SMTP. Include proper authentication and error handling.",
    "Write a Python script that monitors a directory for new files and logs their names to a file. Use the watchdog library.",
    "Create a Python function that downloads a file from a URL with a progress bar. Use requests and tqdm.",
    "Write a Python script that parses a log file, extracts error messages, and sends them to a Slack webhook.",
    "Create a Python function that reads environment variables, validates them, and returns a configuration dictionary.",
    "Write a Python script that scrapes a webpage, extracts all links, and saves them to a JSON file. Use BeautifulSoup.",
    "Create a Python function that uploads a file to AWS S3. Use boto3 and include error handling.",
    "Write a Python script that runs a shell command, captures stdout and stderr, and returns the exit code.",
    "Create a Python function that implements a retry mechanism for flaky API calls. Use exponential backoff.",
    "Write a Python script that reads a JSON config file, validates its schema, and returns a typed configuration object.",
    "Create a Python function that generates a PDF report from a dictionary of data. Use reportlab.",
    "Write a Python script that connects to a Redis server, sets a key with an expiration, and retrieves it.",
    "Create a Python function that implements a simple rate limiter using a token bucket algorithm.",
    "Write a Python script that parses command-line arguments using argparse and executes different actions based on the arguments.",
    "Create a Python function that connects to a MongoDB database, inserts a document, and queries for it.",
    "Write a Python script that reads a YAML file, modifies a value, and writes it back. Use PyYAML.",
    "Create a Python function that implements a circuit breaker pattern for external service calls.",
    "Write a Python script that creates a simple HTTP server with basic routing. Use http.server.",
    "Create a Python function that encrypts and decrypts text using AES. Use the cryptography library.",
    "Write a Python script that connects to a GitHub API, lists repositories, and creates an issue. Use PyGithub.",
    "Create a Python function that implements a simple message queue using Redis lists.",
    "Write a Python script that monitors CPU and memory usage and logs alerts when thresholds are exceeded.",
    "Create a Python function that implements a simple key-value store with file persistence.",
    "Write a Python script that parses an XML file and extracts specific elements using ElementTree.",
    "Create a Python function that implements a simple pub-sub system using Redis.",
    "Write a Python script that creates a backup of a directory, compresses it, and uploads to cloud storage.",
    "Create a Python function that implements a simple health check endpoint for a web service.",
]

# === PARAMETER VALUES ===
PARAMS = {
    "file": ["config.json", "data.csv", "notes.txt", "report.md", "main.py", "README.md",
             "settings.yaml", "database.db", "log.txt", "todo.txt", "app.py", "test.py"],
    "content": ["a list of tasks for today", "meeting notes from the standup",
                "a configuration for the server", "a summary of the project",
                "a list of contacts", "a budget report", "a test plan",
                "a deployment checklist", "a code review summary"],
    "dir": ["the current directory", "the project folder", "the src directory",
           "the tests folder", "the config directory", "the logs folder"],
    "dest": ["the backup folder", "the archive directory", "the cloud storage",
            "a new location", "the temp directory"],
    "query": ["neural networks", "climate change", "Python tutorials", "quantum physics",
              "machine learning", "data structures", "web development", "database design",
              "error messages", "failed requests"],
    "url": ["https://api.example.com/data", "https://example.com/file.zip",
            "https://cdn.example.com/assets", "https://github.com/user/repo"],
    "endpoint": ["/api/users", "/api/orders", "/api/search", "/api/health",
                "/api/config", "/api/products", "/api/analytics"],
    "params": ["{'limit': 10}", "{'query': 'test'}", "{'page': 1, 'size': 20}",
              "{'filter': 'active'}", "{'sort': 'date'}"],
    "data": ["{'name': 'John', 'email': 'john@example.com'}",
            "{'status': 'active', 'count': 42}",
            "{'title': 'New Post', 'author': 'admin'}"],
    "table": ["users", "orders", "products", "transactions", "logs", "sessions"],
    "condition": ["id = 1", "status = 'pending'", "created_at < '2024-01-01'",
                 "count > 10", "active = true"],
    "value": ["status = 'completed'", "count = 0", "active = false", "name = 'Updated'"],
    "service": ["nginx", "postgresql", "redis", "docker", "api-server", "worker",
              "web-app", "database", "cache"],
    "command": ["ls -la", "grep -r 'error' .", "find . -name '*.py'", "du -sh .",
               "ps aux", "df -h", "top -n 1", "netstat -tulpn", "cat /var/log/syslog"],
    "recipient": ["the team", "the client", "the manager", "all stakeholders", "support"],
    "subject": ["the project update", "the meeting summary", "the bug report",
               "the deployment status", "the weekly report"],
    "platform": ["Twitter", "Slack", "Discord", "Teams", "LinkedIn"],
    "message": ["hello world", "project update: all systems go",
               "deployment complete", "meeting in 5 minutes", "new release published"],
    "event": ["team meeting", "project deadline", "code review", "sprint planning",
             "deployment window", "client presentation"],
    "date": ["Monday at 10am", "Friday at 3pm", "tomorrow", "next week", "today at 5pm"],
    "task": ["check the build status", "review the pull request", "update the documentation",
            "run the tests", "deploy the update", "check server health"],
    "time": ["9am", "2pm", "5pm", "midnight", "end of day"],
    "key": ["name", "email", "favorite color", "location", "job title", "birthday"],
    "value_simple": ["John", "john@example.com", "blue", "New York", "engineer", "January 15"],
    "branch": ["feature/auth", "bugfix/login", "hotfix/security", "develop", "feature/api"],
    "message_git": ["add user authentication", "fix login bug", "update documentation",
                   "refactor database layer", "add test coverage"],
    "cpu": ["45", "23", "78", "12", "56"],
    "ram": ["52", "34", "67", "28", "45"],
    "disk": ["70", "45", "89", "23", "56"],
    "file_content": ["# Configuration\\nserver: localhost\\nport: 8080\\ndebug: true",
                    "name,age,city\\nJohn,30,NYC\\nJane,25,LA\\nBob,35,Chicago",
                    "Meeting Notes\\n- Discussed roadmap\\n- Assigned tasks\\n- Set deadlines",
                    "# TODO\\n- Fix login bug\\n- Add tests\\n- Update docs"],
    "file_summary": ["The file contains server configuration with localhost on port 8080 with debug enabled.",
                    "The file is a CSV with name, age, and city columns containing 3 data rows.",
                    "The file contains meeting notes discussing roadmap, task assignments, and deadlines.",
                    "The file is a todo list with 3 items: fix login bug, add tests, and update docs."],
    "file_list": ["config.json, data.csv, main.py, README.md, utils.py",
                 "app.js, package.json, Dockerfile, .gitignore",
                 "test_1.py, test_2.py, test_3.py, conftest.py"],
    "search_results": ["3 files: main.py, utils.py, config.py",
                      "1 file: app.js", "5 files matching the query"],
    "web_results": ["several articles about the topic with key findings and recent developments",
                   "documentation pages and tutorials with code examples",
                   "research papers and blog posts discussing the topic"],
    "fetch_content": ["a webpage with HTML content about the requested topic",
                     "a JSON response with data fields and metadata",
                     "a text file with configuration settings"],
    "fetch_summary": ["The page discusses the topic in detail with examples and references.",
                     "The response contains structured data with 15 records.",
                     "The file contains 10 configuration settings for the application."],
    "api_response": ["200 OK with JSON data containing 42 records",
                    "201 Created with the new resource ID",
                    "200 OK with paginated results on page 1"],
    "api_post_response": ["201 Created — resource saved with ID 123",
                         "200 OK — resource updated successfully",
                         "400 Bad Request — validation error on field 'name'"],
    "db_results": ["5 rows: Alice (admin), Bob (user), Carol (editor), Dave (user), Eve (admin)",
                  "1 row: order #1234, total $45.99, status: shipped",
                  "0 rows — no matching records found"],
    "git_status": ["M main.py, ?? new_file.txt, A config.json",
                  "M app.js, M styles.css, D old_module.py",
                  "clean working tree, up to date with origin/main"],
    "git_log": ["a1b2c3d added auth feature | e4f5g6h fixed bug | h7i8j9k updated docs",
               "latest: refactored API layer | previous: added tests | older: initial commit",
               "x1y2z3d deployed v2.0 | a3b4c5d merged feature branch | e6f7g8h fixed security issue"],
    "command_output": ["total 24K\\ndrwxr-xr-x 5 root root 4096 Jan 15 10:30 .\\n-rw-r--r-- 1 root root 1024 Jan 15 09:00 app.py",
                       "found 3 matches in main.py at lines 15, 28, 42",
                       "Filesystem Size Used Avail Use% Mounted on\\n/dev/sda1 50G 35G 15G 70% /",
                       "USER PID %CPU %MEM COMMAND\\nroot 1234 2.3 4.5 python app.py"],
    "log_content": ["[ERROR] 2024-01-15 10:30: Connection refused to database\\n[WARN] 2024-01-15 10:31: Retrying...",
                   "[INFO] Server started on port 8080\\n[INFO] All systems nominal",
                   "[ERROR] Failed to load config: file not found\\n[ERROR] Shutting down"],
    "tweet_results": ["5 tweets about the topic with varying perspectives and engagement levels",
                     "3 tweets: one from a researcher, one from a developer, one from a journalist"],
}


def generate_tool_prompts(count, seed=42):
    """Generate tool-calling and agent prompts."""
    rng = random.Random(seed)
    prompts = []
    seen = set()

    # Tool calling scenarios
    for template, _ in TOOL_SCENARIOS:
        prompt = template
        for key, values in PARAMS.items():
            placeholder = "{" + key + "}"
            while placeholder in prompt:
                prompt = prompt.replace(placeholder, rng.choice(values), 1)
        if prompt not in seen and "{" not in prompt:
            seen.add(prompt)
            prompts.append(prompt)

    # Agent scenarios (use as-is, they're already complete)
    for prompt, _ in AGENT_SCENARIOS:
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)

    # Code + tool scenarios
    for prompt in CODE_TOOL_SCENARIOS:
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)

    # Generate variations by re-rolling parameters
    attempts = 0
    while len(prompts) < count and attempts < count * 20:
        template, _ = rng.choice(TOOL_SCENARIOS)
        prompt = template
        for key, values in PARAMS.items():
            placeholder = "{" + key + "}"
            while placeholder in prompt:
                prompt = prompt.replace(placeholder, rng.choice(values), 1)
        if prompt not in seen and "{" not in prompt:
            seen.add(prompt)
            prompts.append(prompt)
        attempts += 1

    rng.shuffle(prompts)
    return prompts[:count]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate tool-calling prompts")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", default="tool_prompts.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = generate_tool_prompts(args.count, args.seed)
    with open(args.output, "w") as f:
        for p in prompts:
            f.write(p + "\n")

    print(f"Generated {len(prompts):,} tool-calling prompts to {args.output}")