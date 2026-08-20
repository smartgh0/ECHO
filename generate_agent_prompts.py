#!/usr/bin/env python3
"""Generate agent and tool-calling prompts that teach Echo to act as an AI agent.

These prompts ask the teacher to respond in a tool-call format so Echo
learns the pattern of: think → call tool → read result → respond.
"""

import random

# === SYSTEM PROMPT to instruct the teacher ===
SYSTEM_PROMPT = """You are an AI assistant that uses tools to help users. When a user asks you to do something, respond in this exact format:

think: <brief reasoning about what to do>
tool: <tool_name> <arguments>
tool_result: <simulated result of the tool call>
echo: <your response to the user based on the tool result>

If multiple steps are needed, chain them:
think: <reasoning>
tool: <tool_name> <args>
tool_result: <result>
think: <next reasoning>
tool: <tool_name> <args>
tool_result: <result>
echo: <final response>

Available tools: file_read, file_write, file_list, file_search, file_delete, web_search, web_fetch, api_call, db_query, db_insert, db_update, db_delete, shell_run, git_status, git_commit, git_branch, git_merge, email_send, calendar_create, reminder_set, memory_store, memory_recall, mcp_call, http_get, http_post, code_execute, code_write, package_install, service_start, service_stop, service_status, log_read, config_get, config_set, backup_create, backup_restore, deploy_app, test_run, lint_run, format_code, doc_generate, data_analyze, data_visualize, encrypt_data, decrypt_data, hash_compute, jwt_generate, jwt_verify, queue_push, queue_pop, cache_set, cache_get, search_index, vector_search, schedule_task, monitor_add, alert_create, webhook_register, notification_send

Always use the tool format. Never just answer directly — show the tool call."""

# === PROMPTS that force tool-calling behavior ===
PROMPTS = [
    # File operations
    "Read the file config.json and tell me what's in it.",
    "Write 'Hello World' to a file called output.txt.",
    "List all files in the current directory.",
    "Search for all Python files containing 'import torch' in the src directory.",
    "Delete the file temp_data.csv.",
    "Create a new file called notes.txt with the content 'Meeting at 3pm'.",
    "Read the file app.py and check for any syntax errors.",
    "Append 'TODO: fix login' to the file todo.txt.",
    "Show me the first 10 lines of the file log.txt.",
    "Count the number of lines in data.csv.",
    "Find all .json files in the config directory.",
    "Read the file package.json and tell me the dependencies.",
    "Write a Python dictionary to a file called settings.json.",
    "Move the file old_config.yaml to config.yaml.",
    "Copy the file database.db to backup_database.db.",
    "Check if the file .env exists and show its contents.",
    "Read the file Dockerfile and explain what it does.",
    "Create a directory called logs if it doesn't exist.",
    "Read the file requirements.txt and list all packages.",
    "Write a list of 5 tasks to tasks.txt, one per line.",

    # Web and API operations
    "Search the web for 'Python async best practices'.",
    "Fetch the content from https://api.github.com/users/torvalds.",
    "Call the API endpoint /api/users with a GET request.",
    "Send a POST request to /api/orders with body {'product': 'laptop', 'qty': 2}.",
    "Download the file from https://example.com/data.csv.",
    "Search the web for 'how to implement JWT authentication'.",
    "Fetch the README from https://github.com/pytorch/pytorch.",
    "Call the API /api/health and check if the service is up.",
    "Make a PUT request to /api/users/42 with body {'name': 'Alice'}.",
    "Send a DELETE request to /api/posts/123.",
    "Search for 'React vs Vue performance comparison'.",
    "Fetch the weather data from https://api.weather.com/today.",
    "Call the API /api/products?category=electronics&limit=10.",
    "Download a file from https://cdn.example.com/assets/logo.png.",
    "Search the web for 'best practices for Docker security'.",
    "Fetch the latest release info from https://api.github.com/repos/python/cpython/releases.",
    "Call the API /api/analytics with params {'start': '2024-01-01', 'end': '2024-12-31'}.",
    "Send a webhook to https://hooks.slack.com/services/xxx with payload {'text': 'Build passed'}.",
    "Search for 'how to optimize PostgreSQL queries'.",
    "Fetch the content of https://httpbin.org/json and parse it.",

    # Database operations
    "Query the database for all users where status is 'active'.",
    "Insert a new user with name 'John', email 'john@example.com' into the users table.",
    "Update the orders table where id=42 to set status='shipped'.",
    "Delete from the logs table where created_at < '2024-01-01'.",
    "Run a query to count total orders by status.",
    "Query the database for the top 10 products by sales.",
    "Insert a new order with product_id=5, quantity=3, total=99.99.",
    "Update all users where last_login < '2024-01-01' to set status='inactive'.",
    "Query the database for the average order value in the last 30 days.",
    "Delete duplicate rows from the products table.",
    "Run a JOIN query between users and orders tables.",
    "Query the database schema for the users table.",
    "Insert multiple records into the tags table: ['python', 'ai', 'ml'].",
    "Update the products table where category='electronics' to set discount=0.15.",
    "Query for users who haven't logged in for 90 days.",

    # Shell and system operations
    "Run the command 'ls -la' and show the output.",
    "Run 'git status' and tell me what changed.",
    "Execute 'python -m pytest' and show the test results.",
    "Run 'docker ps' and list all running containers.",
    "Execute 'npm install' and show the output.",
    "Run 'pip list' and show all installed packages.",
    "Execute 'git log --oneline -5' and show recent commits.",
    "Run 'df -h' and check disk usage.",
    "Execute 'top -n 1' and show system resources.",
    "Run 'cat /etc/os-release' and tell me the OS version.",
    "Execute 'curl -s http://localhost:8080/health' and check the response.",
    "Run 'git diff' and show what changed since last commit.",
    "Execute 'find . -name \"*.py\" -type f' and list all Python files.",
    "Run 'grep -r \"TODO\" .' and find all TODO comments.",
    "Execute 'ps aux | grep python' and show Python processes.",

    # Git operations
    "Check the git status of the repository.",
    "Commit all changes with the message 'fix: resolve login bug'.",
    "Create a new branch called 'feature/user-auth'.",
    "Merge the branch 'feature/api' into main.",
    "Show the last 5 git commits.",
    "Create a new branch called 'hotfix/security-patch' and switch to it.",
    "Commit only the file app.py with message 'refactor: clean up app module'.",
    "Show the diff between main and develop branches.",
    "Merge 'feature/payment' into develop.",
    "Create a tag called 'v1.2.0' with message 'Release 1.2.0'.",

    # Email and communication
    "Send an email to the team about the project update.",
    "Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Don't forget the meeting at 10am'.",
    "Post a message on Slack saying 'Deployment complete, all systems go'.",
    "Send a notification to the #devops channel about the build failure.",
    "Post on Twitter: 'Excited to announce our new AI model is now open source!'",

    # Calendar and reminders
    "Create a calendar event for 'Team Meeting' on Monday at 10am.",
    "Set a reminder for 'review the pull request' at 3pm today.",
    "Create a calendar event for 'Sprint Planning' on Friday at 2pm for 1 hour.",
    "Set a reminder for 'deploy to production' tomorrow at 9am.",
    "Create a recurring event for 'Weekly Standup' every Monday at 9am.",

    # Memory operations
    "Remember that my name is Alice.",
    "Remember that I prefer Python over JavaScript.",
    "What is my name?",
    "Remember that I work at TechCorp as a software engineer.",
    "What programming language do I prefer?",
    "Remember that my favorite editor is VS Code.",
    "What do I do for work?",
    "Remember that I'm learning machine learning.",
    "What editor do I use?",
    "Forget my name.",

    # MCP (Model Context Protocol) operations
    "Call the MCP server 'filesystem' with method 'read_file' and args {'path': '/config/app.json'}.",
    "Call the MCP server 'database' with method 'query' and args {'sql': 'SELECT * FROM users LIMIT 5'}.",
    "Call the MCP server 'web-search' with method 'search' and args {'query': 'Python tutorials'}.",
    "Call the MCP server 'git' with method 'status' and args {'repo': '/workspace/myproject'}.",
    "Call the MCP server 'shell' with method 'execute' and args {'command': 'ls -la'}.",
    "Call the MCP server 'http' with method 'get' and args {'url': 'https://api.example.com/data'}.",
    "Call the MCP server 'cache' with method 'get' and args {'key': 'user_session_42'}.",
    "Call the MCP server 'queue' with method 'push' and args {'queue': 'tasks', 'message': 'process order 1234'}.",
    "Call the MCP server 'vector-db' with method 'search' and args {'collection': 'docs', 'query': 'how to deploy', 'limit': 5}.",
    "Call the MCP server 'notification' with method 'send' and args {'channel': 'email', 'recipient': 'team@company.com', 'subject': 'Alert', 'body': 'Server down'}.",
    "List all available MCP servers.",
    "Call the MCP server 'filesystem' with method 'list_directory' and args {'path': '/workspace'}.",
    "Call the MCP server 'database' with method 'insert' and args {'table': 'logs', 'data': {'level': 'INFO', 'message': 'Server started'}}.",
    "Call the MCP server 'web-search' with method 'fetch_page' and args {'url': 'https://docs.python.org/3/'}.",
    "Call the MCP server 'git' with method 'log' and args {'repo': '/workspace/myproject', 'limit': 10}.",

    # Multi-step agent scenarios
    "I need to deploy my web app. Can you help me through the steps?",
    "Help me debug why my API is returning 500 errors.",
    "I want to set up a CI/CD pipeline for my project. Walk me through it.",
    "My database is slow. Can you help me diagnose the issue?",
    "I need to migrate my app from Heroku to AWS. What should I do?",
    "Help me set up monitoring for my microservices.",
    "I think there's a memory leak in my Python app. Can you investigate?",
    "I need to refactor a large monolith into microservices. Where do I start?",
    "Help me implement authentication for my REST API.",
    "My Docker container keeps crashing. Can you help me figure out why?",
    "I need to analyze my server logs for errors from the last 24 hours.",
    "Help me create a backup strategy for my PostgreSQL database.",
    "I want to add rate limiting to my API. How do I do it?",
    "My tests are flaky and sometimes fail. Can you help me debug?",
    "I need to set up SSL for my web server. Walk me through it.",
    "Help me optimize my database queries that are running slow.",
    "I want to implement a caching layer for my API. What should I do?",
    "My Kubernetes pod won't start. Can you help me troubleshoot?",
    "I need to implement a webhook system for my app. Help me plan it.",
    "Help me set up a load balancer for my web application.",

    # Code execution and analysis
    "Run this Python code: print([x**2 for x in range(10)])",
    "Execute this code and tell me the output: import json; print(json.dumps({'a': 1, 'b': 2}, indent=2))",
    "Write a Python function to calculate fibonacci numbers and run it to verify it works.",
    "Execute this SQL query: SELECT COUNT(*) FROM users WHERE active = true",
    "Run this shell command and show the output: echo 'Hello' | tr a-z A-Z",
    "Write and execute a Python script that reads a CSV file and prints the first 5 rows.",
    "Run this JavaScript code: console.log(Array.from({length: 5}, (_, i) => i * 2))",
    "Execute this Python code and check for errors: def add(a, b): return a + b; print(add(1, 2))",
    "Write a function that sorts a list and test it with [3, 1, 4, 1, 5, 9, 2, 6].",
    "Run this code: import os; print(os.listdir('.'))",

    # Configuration and deployment
    "Get the current configuration of the api-server service.",
    "Set the environment variable DATABASE_URL to postgresql://localhost/mydb.",
    "Get the value of the API_KEY environment variable.",
    "Set the configuration parameter 'max_connections' to 100 for the database service.",
    "Deploy the application to the staging environment.",
    "Deploy version 2.1.0 to production with a rolling update.",
    "Roll back the deployment to the previous version.",
    "Check the deployment status of the web-app service.",
    "Scale the api-server to 5 instances.",
    "Get the configuration of the nginx service.",

    # Monitoring and alerting
    "Check the health of the api-server service.",
    "Add a monitoring alert for CPU usage above 80%.",
    "Check the logs of the web-app for any errors in the last hour.",
    "Create an alert for high memory usage on the database server.",
    "Monitor the response time of the /api/users endpoint.",
    "Check the status of all running services.",
    "Set up a health check endpoint for the web service.",
    "Create an alert for failed login attempts above 10 per minute.",
    "Check the error rate of the api-server in the last 24 hours.",
    "Monitor the disk space on the database server.",

    # Data analysis
    "Analyze the data in sales.csv and give me a summary.",
    "Read the file users.json and count how many users are active.",
    "Load the data from orders.csv and find the top 5 customers by total spending.",
    "Analyze the log file app.log and find the most common error.",
    "Read the file metrics.json and create a summary of the key metrics.",
    "Load data.csv, group by category, and show the average price per category.",
    "Analyze the file transactions.csv and detect any anomalies.",
    "Read the file survey_results.csv and calculate the satisfaction score.",
    "Load the data from inventory.json and find items with stock below 10.",
    "Analyze the file access_log.csv and find the peak traffic hours.",

    # Security operations
    "Encrypt the string 'Hello World' using AES.",
    "Decrypt this encrypted message: U2FsdGVkX1+abc123...",
    "Generate a JWT token for user_id=42 with admin role.",
    "Verify this JWT token: eyJhbGciOiJIUzI1NiIs...",
    "Hash the password 'mypassword123' using bcrypt.",
    "Generate a random API key of 32 characters.",
    "Check if the password 'abc123' meets security requirements.",
    "Generate an SSH key pair.",
    "Encrypt the file config.json.",
    "Verify the integrity of the file data.csv using a checksum.",

    # Package management
    "Install the package 'requests' using pip.",
    "Install the packages 'flask' and 'sqlalchemy' using pip.",
    "List all installed npm packages.",
    "Install the package 'pandas' with version 2.0.0.",
    "Uninstall the package 'old-package' using pip.",
    "Update all pip packages to their latest versions.",
    "Install the package 'torch' with CUDA support.",
    "Check if the package 'numpy' is installed.",
    "Install the development dependencies from requirements-dev.txt.",
    "Create a requirements.txt from the currently installed packages.",

    # Testing and quality
    "Run the test suite and show the results.",
    "Run the linter on the src directory.",
    "Format the code in the project using black.",
    "Run the tests in the file test_app.py.",
    "Generate documentation for the project.",
    "Run the type checker on the codebase.",
    "Check code coverage for the test suite.",
    "Run the security scanner on the project dependencies.",
    "Format the file main.py using autopep8.",
    "Generate a test file for the module utils.py.",

    # Backup and recovery
    "Create a backup of the database.",
    "Create a backup of the /workspace directory.",
    "Restore the database from the backup file backup_20240115.sql.",
    "Create a compressed backup of the config directory.",
    "Restore the configuration from the backup config_backup.tar.gz.",
    "Create a snapshot of the current system state.",
    "Backup all log files to the backup directory.",
    "Restore the application from the last known good backup.",
    "Create an incremental backup of the data directory.",
    "Verify the integrity of the backup file.",

    # Queue and async operations
    "Push a message 'process order 1234' to the 'orders' queue.",
    "Pop the next message from the 'tasks' queue.",
    "Check the length of the 'notifications' queue.",
    "Push 5 messages to the 'emails' queue.",
    "Clear all messages from the 'temp' queue.",
    "Schedule a task to run every 5 minutes.",
    "Schedule a task to run at midnight tonight.",
    "Cancel the scheduled task with ID 'task_42'.",
    "List all scheduled tasks.",
    "Schedule a backup to run every Sunday at 2am.",

    # Cache operations
    "Set the cache key 'user_session_42' to {'user_id': 42, 'role': 'admin'} with a 1 hour expiration.",
    "Get the value of the cache key 'user_session_42'.",
    "Delete the cache key 'temp_data'.",
    "Clear all cache entries matching the pattern 'session_*'.",
    "Check if the cache key 'config' exists.",
    "Set the cache key 'api_response' to {'status': 'ok'} with 5 minute expiration.",
    "Get all cache keys matching the pattern 'user_*'.",
    "Increment the cache counter 'page_views' by 1.",
    "Set the cache key 'feature_flags' to {'new_ui': true, 'dark_mode': false}.",
    "Get the remaining TTL for the cache key 'session_token'.",

    # Vector search and AI operations
    "Search the vector database for documents similar to 'how to deploy a web app'.",
    "Add a document to the vector database with content 'Python is great for data science'.",
    "Search the 'knowledge_base' collection for 'machine learning basics'.",
    "Find similar code snippets to 'def quicksort(arr): ...' in the code collection.",
    "Search the documentation index for 'authentication setup'.",
    "Add a new embedding to the 'products' collection with metadata {'category': 'electronics'}.",
    "Search for similar images to the uploaded file.",
    "Find the most relevant FAQ entries for the question 'how to reset password'.",
    "Search the 'articles' collection for 'climate change impact' and return top 5.",
    "Index the document 'API Reference v2.0' into the search engine.",

    # Notification and webhook
    "Send a notification to the user about their order being shipped.",
    "Register a webhook for the event 'order.created' pointing to https://myapp.com/webhook.",
    "Send a push notification to device 'device_123' with message 'Your package has arrived'.",
    "Notify all admins about the system maintenance window.",
    "Send a notification to the #alerts channel about the high CPU usage.",
    "Register a webhook for GitHub push events.",
    "Send an SMS notification to +1234567890 with message 'Your code is 123456'.",
    "Notify the team that the build has passed.",
    "Register a webhook for Stripe payment events.",
    "Send a notification about the new comment on the pull request.",

    # Complex multi-tool scenarios
    "Read the file config.json, update the 'port' field to 8080, and write it back.",
    "Search the web for 'Python Flask tutorial', fetch the first result, and save it to a file.",
    "Query the database for all users, export to CSV, and upload to cloud storage.",
    "Read the log file, find all ERROR entries, and send an email alert about them.",
    "Check git status, commit any changes, and push to the remote repository.",
    "Read the file data.csv, analyze it, create a visualization, and save it as a PNG.",
    "Query the API for user data, store it in the database, and cache the result.",
    "Read the file app.py, run the linter, fix any issues, and run the tests.",
    "Fetch the weather data, store it in the database, and send a notification if it's raining.",
    "Read the file config.yaml, validate it, encrypt it, and save the encrypted version.",
    "Search the vector database for similar documents, read the top result, and summarize it.",
    "Check the server health, read the error logs, and create an alert if there are critical errors.",
    "Query the database for slow queries, optimize them, and update the indexes.",
    "Read the file package.json, check for outdated dependencies, and update them.",
    "Fetch the latest news from the API, filter for tech news, and save to a file.",
    "Read the file users.csv, validate the emails, and insert valid ones into the database.",
    "Check the disk usage, find large files, and move old ones to backup storage.",
    "Read the file test_results.json, analyze pass/fail rates, and send a report via email.",
    "Query the API for all products, compare prices with competitors, and update the database.",
    "Read the file access_log.csv, detect suspicious activity, and create security alerts.",
]


def generate_agent_prompts(count, seed=42):
    """Generate agent and tool-calling prompts."""
    rng = random.Random(seed)
    prompts = list(PROMPTS)
    rng.shuffle(prompts)
    return prompts[:count]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate agent tool-calling prompts")
    parser.add_argument("--count", type=int, default=len(PROMPTS))
    parser.add_argument("--output", default="agent_prompts.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = generate_agent_prompts(args.count, args.seed)
    with open(args.output, "w") as f:
        for p in prompts:
            f.write(p + "\n")

    print(f"Generated {len(prompts)} agent tool-calling prompts to {args.output}")