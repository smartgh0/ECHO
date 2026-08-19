#!/usr/bin/env python3
"""Generate balanced domain-specific prompts for Echo distillation.

Domains: coding (25%), tools (15%), science (20%), math (15%),
environmental (10%), history (10%), general (5%)
"""

import random
import sys

# === CODING PROMPTS (25%) ===
CODING_TEMPLATES = [
    "Write a Python function that {task}.",
    "How do you {task} in Python?",
    "Write a Python function to {task}. Include error handling.",
    "Explain how to {task} with a code example.",
    "Write a Python script that {task}.",
    "Implement a function that {task}. Explain the logic.",
    "Write code to {task}. Add comments explaining each step.",
    "How would you {task} in JavaScript?",
    "Write a Python class that {task}.",
    "Create a function that {task}. Handle edge cases.",
    "Write a recursive function to {task}.",
    "How do you {task} using a dictionary in Python?",
    "Write a Python function that {task}. What is the time complexity?",
    "Implement {task} using object-oriented programming.",
    "Write a Python function that {task}. Write unit tests for it.",
    "How do you {task} efficiently? Show the code.",
    "Write a Python decorator that {task}.",
    "Create a generator function that {task}.",
    "Write a Python function that {task} using list comprehension.",
    "How do you {task} using pandas?",
    "Write a Python function that {task} using numpy.",
    "Implement a binary search algorithm to {task}.",
    "Write a Python function that {task} using regex.",
    "How do you {task} in SQL?",
    "Write a Python function that {task}. Include type hints.",
    "Create a REST API endpoint that {task}.",
    "Write a Python function that {task} using async/await.",
    "How do you {task} using Git commands?",
    "Write a Python script that {task} and saves the result to a file.",
    "Implement a data structure that {task}.",
]

CODING_TASKS = [
    "reverses a string", "checks if a number is prime", "sorts a list of dictionaries by a key",
    "finds the maximum element in a list", "merges two sorted lists", "counts word frequency in a text",
    "removes duplicates from a list", "finds the GCD of two numbers", "converts Celsius to Fahrenheit",
    "validates an email address", "calculates factorial recursively", "generates Fibonacci numbers",
    "checks if a string is a palindrome", "finds all prime numbers up to N", "compresses a string using run-length encoding",
    "implements a stack with push and pop", "finds the longest common substring", "rotates a matrix 90 degrees",
    "implements a queue using two stacks", "finds the first non-repeating character",
    "validates a binary search tree", "implements a hash table from scratch", "finds the shortest path in a graph",
    "detects a cycle in a linked list", "balances parentheses in a string", "converts infix to postfix notation",
    "implements merge sort", "finds the k-th largest element", "checks if two strings are anagrams",
    "calculates the sum of digits in a number", "finds all permutations of a string",
    "implements a binary tree traversal", "finds the intersection of two lists",
    "validates a credit card number", "implements a simple calculator",
    "finds the longest increasing subsequence", "checks if a number is a power of two",
    "converts a decimal number to binary", "implements a priority queue",
    "finds the median of two sorted arrays", "counts the number of vowels in a string",
    "reverses the words in a sentence", "finds the missing number in an array",
    "implements a circular buffer", "finds the majority element in a list",
    "validates a JSON string", "implements a simple hash function",
    "finds the longest palindromic substring", "implements a trie data structure",
    "checks if a tree is balanced", "finds the diameter of a binary tree",
    "implements quicksort", "finds the minimum spanning tree",
    "detects if a graph is bipartite", "implements topological sort",
    "finds the longest path in a DAG", "implements Dijkstra's algorithm",
    "finds all subsets of a set", "implements a bloom filter",
    "finds the longest common prefix", "implements an LRU cache",
    "checks if a Sudoku board is valid", "solves the N-queens problem",
    "implements a rate limiter", "finds the k-most frequent elements",
    "validates a password strength", "implements a simple blockchain",
    "finds the longest word in a sentence", "implements a Caesar cipher",
    "counts islands in a 2D grid", "finds the shortest common supersequence",
    "implements a sliding window maximum", "finds the first unique character",
    "checks if a number is an Armstrong number", "implements a simple neural network layer",
    "finds the longest chain in a matrix", "implements a text editor with undo",
    "calculates matrix multiplication", "implements a simple web scraper",
    "finds the longest consecutive sequence", "implements a basic interpreter",
    "creates a simple key-value store", "implements a connection pool",
    "finds the maximum subarray sum", "implements a simple file system",
    "creates a basic HTTP server", "implements a pub-sub system",
    "finds the longest increasing path in a matrix", "implements a simple regex engine",
    "creates a basic chat server", "implements a distributed lock",
    "finds the minimum window substring", "implements a simple database",
    "creates a basic load balancer", "implements a circuit breaker pattern",
    "finds the longest valid parentheses", "implements a retry mechanism",
    "creates a simple message queue", "implements a basic compiler",
    "finds the maximum product subarray", "implements a simple virtual machine",
    "creates a basic search engine", "implements a code formatter",
    "finds the longest substring without repeating characters", "implements a syntax highlighter",
    "creates a simple game engine", "implements a basic profiler",
    "finds the minimum path sum in a grid", "implements a memory allocator",
    "creates a basic file compressor", "implements a simple debugger",
    "finds the longest palindromic subsequence", "implements a code linter",
    "creates a simple package manager", "implements a basic test framework",
    "finds the maximum depth of nested parentheses", "implements a simple build system",
    "creates a basic version control system", "implements a code analyzer",
    "finds the shortest unsorted subarray", "implements a simple IDE plugin",
    "creates a basic terminal emulator", "implements a code generator",
    "finds the longest mountain in an array", "implements a simple type checker",
    "creates a basic shell", "implements a simple linker",
    "finds the maximum frequency stack", "implements a simple assembler",
    "creates a basic text editor", "implements a simple parser",
    "finds the longest well-performing interval", "implements a simple lexer",
    "creates a basic calculator with memory", "implements a simple interpreter for a toy language",
    "finds the maximum sum of a circular subarray", "implements a simple virtual filesystem",
    "creates a basic task scheduler", "implements a simple process manager",
    "finds the longest line segment in a 2D plane", "implements a simple window manager",
    "creates a basic image processor", "implements a simple audio processor",
    "finds the maximum rectangle in a histogram", "implements a simple video processor",
    "creates a basic network simulator", "implements a simple protocol parser",
    "finds the longest path in a tree", "implements a simple data serializer",
    "creates a basic compression tool", "implements a simple encryption tool",
    "finds the maximum sum path in a binary tree", "implements a simple hash table",
    "creates a basic sorting visualizer", "implements a simple graph visualizer",
    "finds the longest sequence in a matrix", "implements a simple tree visualizer",
    "creates a basic algorithm visualizer", "implements a simple data structure visualizer",
]

# === TOOL CALLING PROMPTS (15%) ===
TOOL_TEMPLATES = [
    "Read the file {file} and {action}.",
    "Write to {file} with the content {content}.",
    "List all files in the {directory} directory.",
    "Search for {query} in {location}.",
    "Send an email to {recipient} about {subject}.",
    "Create a calendar event for {event} on {date}.",
    "Set a reminder for {task} at {time}.",
    "Look up the weather in {city}.",
    "Search the web for {query}.",
    "Post a message on {platform} saying {message}.",
    "Create a new file called {filename} with {content}.",
    "Delete the file {file}.",
    "Move {file} to {destination}.",
    "Copy {file} to {destination}.",
    "Check the git status of the repository.",
    "Commit changes with the message {message}.",
    "Create a new branch called {branch}.",
    "Merge branch {branch} into main.",
    "Run the test suite and report results.",
    "Deploy the application to {environment}.",
    "Check the database for {query}.",
    "Insert a new record into {table} with {data}.",
    "Update {table} where {condition} to {value}.",
    "Delete records from {table} where {condition}.",
    "Call the API endpoint {endpoint} with {parameters}.",
    "Fetch data from {url} and parse the response.",
    "Upload {file} to {service}.",
    "Download {file} from {url}.",
    "Compress {files} into {archive}.",
    "Extract {archive} to {directory}.",
    "Start the {service} service.",
    "Stop the {service} service.",
    "Check the logs of {service} for {error}.",
    "Monitor the system resources.",
    "Create a backup of {directory}.",
    "Restore from backup {backup}.",
    "Set an environment variable {variable} to {value}.",
    "Run {command} and capture the output.",
    "Schedule {task} to run every {interval}.",
    "Send a notification about {event}.",
    "Create a ticket for {issue}.",
    "Assign {task} to {person}.",
    "Check the status of {service}.",
    "Restart the {service} service.",
    "Scale {service} to {count} instances.",
    "Roll back the deployment to version {version}.",
    "Check the health of {service}.",
    "Get the configuration of {service}.",
    "Update the configuration of {service} with {changes}.",
]

TOOL_PARAMS = {
    "file": ["config.json", "data.csv", "notes.txt", "report.md", "main.py", "README.md",
             "settings.yaml", "database.db", "log.txt", "todo.txt", "index.html",
             "app.js", "styles.css", "requirements.txt", "Dockerfile"],
    "action": ["summarize the contents", "count the lines", "find all errors",
               "extract the key information", "check for duplicates", "validate the format",
               "convert to JSON", "parse the data", "show the first 10 lines"],
    "content": ["a list of tasks", "meeting notes", "a configuration template",
                "a summary report", "a list of contacts", "a budget spreadsheet",
                "a project plan", "a code review checklist", "a test plan"],
    "directory": ["current", "project", "documents", "downloads", "src",
                  "tests", "config", "data", "logs", "backup"],
    "query": ["neural networks", "climate change", "Python tutorials", "quantum physics",
              "machine learning", "data structures", "web development", "database design"],
    "location": ["all files", "the project", "the database", "the web", "the documentation"],
    "recipient": ["the team", "the client", "the manager", "all stakeholders", "support"],
    "subject": ["the project update", "the meeting summary", "the bug report",
                "the deployment status", "the weekly report"],
    "event": ["team meeting", "project deadline", "code review", "sprint planning",
              "deployment window", "client presentation"],
    "date": ["Monday at 10am", "Friday at 3pm", "tomorrow", "next week", "today at 5pm"],
    "task": ["check the build status", "review the pull request", "update the documentation",
             "run the tests", "deploy the update"],
    "time": ["9am", "2pm", "5pm", "midnight", "end of day"],
    "city": ["New York", "London", "Tokyo", "Sydney", "Cairo", "Mumbai", "Sao Paulo"],
    "platform": ["Twitter", "Slack", "Discord", "Teams", "email"],
    "message": ["hello world", "project update", "deployment complete", "meeting in 5 minutes"],
    "filename": ["todo.txt", "config.yaml", "notes.md", "data.json", "output.csv"],
    "destination": ["the backup folder", "the cloud", "a new directory", "the archive"],
    "branch": ["feature/auth", "bugfix/login", "hotfix/security", "develop"],
    "environment": ["staging", "production", "development", "testing"],
    "table": ["users", "orders", "products", "transactions", "logs"],
    "data": ["name='John' email='john@example.com'", "status='active' count=42",
             "title='New Post' author='admin'", "price=29.99 stock=100"],
    "condition": ["id=1", "status='pending'", "created_at < '2024-01-01'", "count > 10"],
    "value": ["status='completed'", "count=0", "active=false", "name='Updated'"],
    "endpoint": ["/api/users", "/api/orders", "/api/search", "/api/health", "/api/config"],
    "parameters": ["{'limit': 10}", "{'query': 'test'}", "{'page': 1, 'size': 20}",
                   "{'filter': 'active'}", "{'sort': 'date'}"],
    "url": ["https://api.example.com/data", "https://example.com/file.zip",
            "https://cdn.example.com/assets", "https://github.com/repo"],
    "service": ["nginx", "postgres", "redis", "docker", "api-server", "worker"],
    "files": ["all logs", "the config files", "the source code", "the test files"],
    "archive": ["backup.zip", "release.tar.gz", "data.rar", "project.tar"],
    "command": ["ls -la", "grep -r 'error' .", "find . -name '*.py'", "du -sh .",
                "ps aux", "df -h", "top -n 1", "netstat -tulpn"],
    "interval": ["hour", "day", "week", "5 minutes", "month"],
    "issue": ["server not responding", "database connection timeout",
              "memory leak in worker", "SSL certificate expired"],
    "person": ["Alice", "Bob", "the team lead", "the developer"],
    "count": ["2", "5", "10", "3"],
    "version": ["v1.2.3", "v2.0.0", "the previous version", "v1.0.0"],
    "variable": ["DATABASE_URL", "API_KEY", "DEBUG_MODE", "PORT", "SECRET_KEY"],
    "backup": ["backup_2024_01_01.tar.gz", "snapshot_001", "weekly_backup.zip"],
    "changes": ["{'timeout': 30}", "{'workers': 4}", "{'debug': true}", "{'port': 8080}"],
}

# === SCIENCE PROMPTS (20%) ===
SCIENCE_TOPICS = [
    "gravity", "photosynthesis", "evolution", "DNA", "quantum mechanics", "relativity",
    "thermodynamics", "electromagnetism", "the solar system", "black holes", "atoms",
    "molecules", "chemical reactions", "the periodic table", "nuclear fission",
    "nuclear fusion", "radioactivity", "isotopes", "chemical bonds", "acids and bases",
    "oxidation", "catalysts", "polymers", "crystals", "states of matter",
    "phase transitions", "surface tension", "viscosity", "density", "pressure",
    "buoyancy", "sound waves", "light waves", "electromagnetic spectrum",
    "reflection", "refraction", "diffraction", "interference", "polarization",
    "magnetism", "electric current", "voltage", "resistance", "capacitance",
    "inductance", "semiconductors", "transistors", "diodes", "integrated circuits",
    "the immune system", "viruses", "bacteria", "cells", "proteins", "enzymes",
    "neurons", "the brain", "the nervous system", "the circulatory system",
    "the respiratory system", "the digestive system", "the endocrine system",
    "the skeletal system", "the muscular system", "the reproductive system",
    "genetics", "natural selection", "mutation", "gene expression", "stem cells",
    "mitosis", "meiosis", "chromosomes", "RNA", "protein synthesis",
    "the water cycle", "weather patterns", "ocean currents", "plate tectonics",
    "earthquakes", "volcanoes", "minerals", "rocks", "soil formation",
    "the atmosphere", "the ozone layer", "the greenhouse effect", "clouds",
    "lightning", "tornadoes", "hurricanes", "floods", "droughts",
    "ecosystems", "food chains", "biodiversity", "symbiosis", "parasitism",
    "the carbon cycle", "the nitrogen cycle", "the phosphorus cycle",
    "photosynthesis", "cellular respiration", "fermentation", "metabolism",
    "the scientific method", "hypothesis testing", "experimental design",
    "peer review", "scientific theories", "scientific laws", "falsifiability",
]

# === MATH PROMPTS (15%) ===
MATH_TOPICS = [
    "algebra", "linear equations", "quadratic equations", "polynomials",
    "factoring", "the distributive property", "the order of operations",
    "inequalities", "absolute value", "functions", "domain and range",
    "composite functions", "inverse functions", "exponential functions",
    "logarithms", "sequences", "series", "arithmetic sequences",
    "geometric sequences", "limits", "derivatives", "integrals",
    "the chain rule", "the product rule", "the quotient rule",
    "partial derivatives", "gradient descent", "optimization",
    "linear algebra", "matrices", "matrix multiplication", "determinants",
    "eigenvalues", "eigenvectors", "vector spaces", "linear independence",
    "basis", "dimension", "linear transformations", "inner products",
    "orthogonality", "projection", "least squares", "SVD",
    "probability", "conditional probability", "Bayes' theorem",
    "random variables", "expected value", "variance", "standard deviation",
    "normal distribution", "binomial distribution", "Poisson distribution",
    "the central limit theorem", "hypothesis testing", "confidence intervals",
    "p-values", "correlation", "regression", "linear regression",
    "logistic regression", "overfitting", "cross-validation",
    "discrete mathematics", "set theory", "logic", "proofs",
    "induction", "contradiction", "combinatorics", "permutations",
    "combinations", "the pigeonhole principle", "graph theory",
    "trees", "Euler paths", "Hamiltonian paths", "graph coloring",
    "number theory", "prime numbers", "modular arithmetic",
    "the Euclidean algorithm", "the fundamental theorem of arithmetic",
    "Diophantine equations", "Fermat's little theorem", "RSA encryption",
    "geometry", "triangles", "circles", "polygons", "area",
    "perimeter", "volume", "the Pythagorean theorem", "trigonometry",
    "sine, cosine, and tangent", "the unit circle", "trigonometric identities",
    "complex numbers", "the imaginary unit", "Euler's formula",
    "differential equations", "first-order ODEs", "second-order ODEs",
    "Laplace transforms", "Fourier series", "the Fourier transform",
    "vector calculus", "gradients", "divergence", "curl",
    "line integrals", "surface integrals", "Green's theorem",
    "Stokes' theorem", "the divergence theorem",
]

# === ENVIRONMENTAL SCIENCE PROMPTS (10%) ===
ENV_TOPICS = [
    "climate change", "global warming", "the greenhouse effect", "carbon emissions",
    "renewable energy", "solar power", "wind energy", "hydroelectric power",
    "geothermal energy", "biomass energy", "nuclear energy", "fossil fuels",
    "coal", "oil", "natural gas", "fracking", "carbon capture",
    "deforestation", "reforestation", "biodiversity loss", "endangered species",
    "habitat destruction", "overfishing", "ocean acidification", "coral bleaching",
    "plastic pollution", "microplastics", "e-waste", "air pollution",
    "water pollution", "soil contamination", "noise pollution", "light pollution",
    "the ozone layer", "acid rain", "smog", "particulate matter",
    "the water cycle", "groundwater depletion", "drought", "desertification",
    "erosion", "salinization", "composting", "recycling",
    "waste management", "landfills", "incineration", "biodegradable materials",
    "sustainable agriculture", "organic farming", "pesticides", "fertilizers",
    "GMOs", "monoculture", "crop rotation", "irrigation",
    "the carbon cycle", "the nitrogen cycle", "the phosphorus cycle",
    "the sulfur cycle", "ecosystems", "food webs", "trophic levels",
    "biomes", "tropical rainforests", "deserts", "tundra", "grasslands",
    "wetlands", "coral reefs", "deep sea ecosystems", "polar ecosystems",
    "conservation", "wildlife reserves", "national parks", "marine protected areas",
    "ecological footprint", "carbon footprint", "sustainability",
    "environmental impact assessment", "life cycle analysis",
    "the Paris Agreement", "carbon trading", "carbon tax",
    "environmental policy", "the EPA", "clean air act", "clean water act",
    "biodiversity hotspots", "invasive species", "ecological succession",
    "carrying capacity", "population dynamics", "ecological resilience",
]

# === HISTORY PROMPTS (10%) ===
HISTORY_TOPICS = [
    "ancient Egypt", "ancient Greece", "the Roman Empire", "the Byzantine Empire",
    "the Middle Ages", "the Renaissance", "the Industrial Revolution",
    "the French Revolution", "the American Revolution", "the Russian Revolution",
    "World War I", "World War II", "the Cold War", "the Vietnam War",
    "the Korean War", "the Gulf War", "the Iraq War", "the War in Afghanistan",
    "ancient China", "the Ming Dynasty", "the Qing Dynasty", "the Silk Road",
    "the Mongol Empire", "the Ottoman Empire", "the Mughal Empire",
    "ancient India", "the Maurya Empire", "the Gupta Empire",
    "ancient Mesopotamia", "the Babylonian Empire", "the Persian Empire",
    "the Phoenicians", "the Carthaginian Empire", "the Vikings",
    "the Norman Conquest", "the Crusades", "the Black Death",
    "the Age of Discovery", "colonialism", "the slave trade",
    "the abolition of slavery", "the Civil Rights Movement",
    "the women's suffrage movement", "the labor movement",
    "the Reformation", "the Counter-Reformation", "the Enlightenment",
    "the Scientific Revolution", "the Age of Reason",
    "the Great Depression", "the New Deal", "the Marshall Plan",
    "the Berlin Wall", "the Cuban Missile Crisis", "the Space Race",
    "the fall of the Soviet Union", "the European Union",
    "the United Nations", "the League of Nations",
    "ancient Africa", "the Kingdom of Kush", "the Mali Empire",
    "the Songhai Empire", "the Great Zimbabwe",
    "ancient Americas", "the Maya civilization", "the Aztec Empire",
    "the Inca Empire", "the Native American tribes",
    "the Australian Aboriginal history", "the Maori history",
    "the Industrial Revolution in Britain", "the Victorian Era",
    "the Edwardian Era", "the Roaring Twenties", "the Great Depression",
    "the Swinging Sixties", "the Digital Revolution",
    "the history of writing", "the history of printing", "the history of the wheel",
    "the history of agriculture", "the history of medicine",
    "the history of science", "the history of mathematics",
    "the history of philosophy", "the history of art",
    "the history of music", "the history of literature",
    "the history of education", "the history of democracy",
    "the history of religion", "the history of trade",
    "the history of warfare", "the history of navigation",
    "the history of architecture", "the history of engineering",
]

# === GENERAL KNOWLEDGE PROMPTS (5%) ===
GENERAL_TOPICS = [
    "the meaning of life", "how to be happy", "how to manage stress",
    "how to be productive", "how to learn effectively", "how to communicate well",
    "how to be a good leader", "how to negotiate", "how to give a presentation",
    "how to write well", "how to think critically", "how to solve problems",
    "how to make decisions", "how to be creative", "how to stay motivated",
    "how to build habits", "how to manage time", "how to set goals",
    "how to handle failure", "how to deal with conflict",
    "the importance of empathy", "the value of education", "the power of habits",
    "the art of conversation", "the science of sleep", "the benefits of exercise",
    "the psychology of color", "the economics of happiness",
    "the philosophy of stoicism", "the concept of mindfulness",
    "how to cook a basic meal", "how to manage personal finances",
    "how to start investing", "how to save money", "how to build credit",
    "how to write a resume", "how to prepare for an interview",
    "how to network effectively", "how to start a business",
    "how to manage a team", "how to delegate tasks",
]


def generate_prompts(count, seed=42):
    """Generate balanced domain-specific prompts."""
    rng = random.Random(seed)

    # Target distribution
    targets = {
        "coding": int(count * 0.25),
        "tools": int(count * 0.15),
        "science": int(count * 0.20),
        "math": int(count * 0.15),
        "environmental": int(count * 0.10),
        "history": int(count * 0.10),
        "general": int(count * 0.05),
    }

    prompts = []
    seen = set()

    # Coding prompts
    while len([p for p in prompts if p.startswith("Write") or p.startswith("How") or p.startswith("Implement") or p.startswith("Create") or p.startswith("Explain")]) < targets["coding"]:
        template = rng.choice(CODING_TEMPLATES)
        task = rng.choice(CODING_TASKS)
        prompt = template.format(task=task)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
        if len(seen) > count * 3:
            break

    # Tool prompts
    tool_count = 0
    while tool_count < targets["tools"]:
        template = rng.choice(TOOL_TEMPLATES)
        # Fill in template parameters
        prompt = template
        for key, values in TOOL_PARAMS.items():
            placeholder = "{" + key + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, rng.choice(values), 1)
        if prompt not in seen and "{" not in prompt:
            seen.add(prompt)
            prompts.append(prompt)
            tool_count += 1
        if len(seen) > count * 4:
            break

    # Science prompts
    science_templates = [
        "What is {topic}?", "Explain {topic}.", "How does {topic} work?",
        "Why is {topic} important?", "Describe {topic} in simple terms.",
        "What are the key concepts in {topic}?", "How would you explain {topic} to a beginner?",
        "What are common misconceptions about {topic}?", "What are the practical applications of {topic}?",
        "How has our understanding of {topic} evolved?",
    ]
    science_count = 0
    while science_count < targets["science"]:
        template = rng.choice(science_templates)
        topic = rng.choice(SCIENCE_TOPICS)
        prompt = template.format(topic=topic)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            science_count += 1

    # Math prompts
    math_templates = [
        "What is {topic}?", "Explain {topic} with an example.",
        "How do you solve problems using {topic}?", "What are the key formulas in {topic}?",
        "Explain {topic} step by step.", "What is the intuition behind {topic}?",
        "How is {topic} used in real life?", "What are the common mistakes in {topic}?",
        "Explain {topic} like you are teaching a student.",
        "What is the relationship between {topic} and calculus?",
    ]
    math_count = 0
    while math_count < targets["math"]:
        template = rng.choice(math_templates)
        topic = rng.choice(MATH_TOPICS)
        prompt = template.format(topic=topic)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            math_count += 1

    # Environmental prompts
    env_templates = [
        "What is {topic}?", "Explain {topic} and its impact.",
        "How does {topic} affect the environment?", "What are the causes of {topic}?",
        "What are the solutions to {topic}?", "How can individuals help with {topic}?",
        "What are the consequences of {topic}?", "Explain {topic} for policymakers.",
        "What is the science behind {topic}?", "How does {topic} relate to climate change?",
    ]
    env_count = 0
    while env_count < targets["environmental"]:
        template = rng.choice(env_templates)
        topic = rng.choice(ENV_TOPICS)
        prompt = template.format(topic=topic)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            env_count += 1

    # History prompts
    history_templates = [
        "What was {topic}?", "Explain the significance of {topic}.",
        "What caused {topic}?", "What were the consequences of {topic}?",
        "Who were the key figures in {topic}?", "How did {topic} change the world?",
        "What can we learn from {topic}?", "Describe {topic} in detail.",
        "How is {topic} relevant today?", "What are common myths about {topic}?",
    ]
    history_count = 0
    while history_count < targets["history"]:
        template = rng.choice(history_templates)
        topic = rng.choice(HISTORY_TOPICS)
        prompt = template.format(topic=topic)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            history_count += 1

    # General prompts
    general_templates = [
        "What is {topic}?", "How do you {topic}?", "Why is {topic} important?",
        "Explain {topic}.", "What are the benefits of {topic}?",
    ]
    general_count = 0
    while general_count < targets["general"]:
        template = rng.choice(general_templates)
        topic = rng.choice(GENERAL_TOPICS)
        prompt = template.format(topic=topic)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            general_count += 1

    rng.shuffle(prompts)
    return prompts[:count]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate domain-balanced prompts")
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--output", default="distill_prompts.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = generate_prompts(args.count, args.seed)

    with open(args.output, "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    print(f"Generated {len(prompts):,} prompts to {args.output}")
    print(f"Distribution:")
    categories = {"coding": 0, "tools": 0, "science": 0, "math": 0,
                  "environmental": 0, "history": 0, "general": 0}
    for p in prompts:
        if any(w in p.lower() for w in ["write", "implement", "create", "function", "code", "python", "javascript", "sql", "git", "algorithm"]):
            categories["coding"] += 1
        elif any(w in p.lower() for w in ["file", "read", "write to", "list", "search", "send", "email", "calendar", "deploy", "database", "api", "upload", "download", "service", "backup", "log"]):
            categories["tools"] += 1
        elif any(w in p.lower() for w in ["gravity", "dna", "quantum", "cell", "atom", "molecule", "chemical", "wave", "magnet", "neuron", "brain", "gene", "ecosystem"]):
            categories["science"] += 1
        elif any(w in p.lower() for w in ["algebra", "calculus", "matrix", "probability", "derivative", "integral", "theorem", "equation", "geometry", "vector"]):
            categories["math"] += 1
        elif any(w in p.lower() for w in ["climate", "carbon", "renewable", "pollution", "ecosystem", "biodiversity", "deforestation", "sustainable", "ozone", "greenhouse"]):
            categories["environmental"] += 1
        elif any(w in p.lower() for w in ["empire", "war", "revolution", "dynasty", "ancient", "medieval", "renaissance", "civilization", "colonial"]):
            categories["history"] += 1
        else:
            categories["general"] += 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / len(prompts) * 100
        print(f"  {cat:15s} {count:>6,} ({pct:.1f}%)")