# ============================================================
# ECHO AGENT — Tool calling system for Echo RNN
# Echo generates tool calls as text. This system parses them,
# executes them, and feeds results back to Echo.
#
# Echo doesn't need to be big. It needs to learn the PATTERN:
#   user asks X → echo outputs "tool: arguments"
# The tool system does the real work.
#
# Available tools:
#   calc: <expression>          — Math evaluation
#   time: now                   — Current time
#   date: today                 — Current date
#   search: <query>             — Web search
#   fetch: <url>                — Fetch a web page
#   email: check                — Check inbox
#   email: send <to> <subject>  — Draft an email
#   file: read <path>           — Read a workspace file
#   file: write <path> <text>   — Write a workspace file
#   file: list <path>            — List directory
#   remember: <key> <value>      — Store in memory
#   recall: <key>                — Retrieve from memory
#   git: status                 — Git status
#   git: log                    — Recent commits
#   twitter: post <text>        — Post to X/Twitter
#   twitter: search <query>     — Search tweets
#   facebook: post <text>       — Post to Facebook page
#   facebook: pages             — List managed pages
#   think: <topic>              — Extended generation
#   status: brain                — Echo brain status
#   help: tools                  — List available tools
# ============================================================

import os
import sys
import json
import math
import time
import datetime
import subprocess
from echo_brain import EchoBrain

# --- Memory store (simple key-value) ---
MEMORY = {}

# --- Tool definitions ---
TOOLS = {
    'calc':     'calc: <expression> — Evaluate a math expression',
    'time':     'time: now — Get current time',
    'date':     'date: today — Get current date',
    'search':   'search: <query> — Search the web',
    'fetch':    'fetch: <url> — Fetch a web page',
    'email':    'email: check | email: send <to> <subject> — Email operations',
    'file':     'file: read|write|list <args> — File operations',
    'remember': 'remember: <key> <value> — Store something in memory',
    'recall':   'recall: <key> — Retrieve something from memory',
    'git':      'git: status|log — Git operations',
    'twitter':  'twitter: post|search <args> — X/Twitter operations',
    'facebook': 'facebook: post|pages — Facebook operations',
    'think':    'think: <topic> — Extended thinking on a topic',
    'status':   'status: brain — Show Echo brain status',
    'help':     'help: tools — List all available tools',
}

class ToolResult:
    """Result of a tool execution."""
    def __init__(self, success=True, output="", error=""):
        self.success = success
        self.output = output
        self.error = error

    def to_text(self):
        if self.success:
            return f"tool_result: {self.output}"
        else:
            return f"tool_error: {self.error}"

    def __str__(self):
        return self.to_text()


def parse_tool_call(text):
    """Check if Echo's output contains a tool call.
    Returns (tool_name, tool_args) or None.

    A tool call looks like:
        calc: 2 + 2
        time: now
        search: quantum mechanics
        remember: name echo
    """
    text = text.strip()

    # Find the first colon
    if ':' not in text:
        return None

    # Try to match known tool prefixes
    for tool_name in TOOLS:
        prefix = f"{tool_name}:"
        if text.startswith(prefix):
            args = text[len(prefix):].strip()
            return (tool_name, args)

    return None


def execute_tool(tool_name, args, brain=None, mcp_tools=None):
    """Execute a tool call. Returns ToolResult.

    brain: EchoBrain instance (for status and think tools)
    mcp_tools: dict of DeeRichAI MCP tool functions (for web, email, social)
    """
    if mcp_tools is None:
        mcp_tools = {}

    try:
        # --- CALC ---
        if tool_name == 'calc':
            return _tool_calc(args)

        # --- TIME ---
        elif tool_name == 'time':
            now = datetime.datetime.now()
            return ToolResult(True, now.strftime('%H:%M:%S'))

        # --- DATE ---
        elif tool_name == 'date':
            today = datetime.date.today()
            return ToolResult(True, today.strftime('%Y-%m-%d (%A)'))

        # --- SEARCH (web) ---
        elif tool_name == 'search':
            if 'web_search' in mcp_tools:
                try:
                    results = mcp_tools['web_search'](args)
                    if results:
                        lines = []
                        for r in results[:5]:
                            lines.append(f"{r.get('title','')}: {r.get('snippet','')}")
                        return ToolResult(True, " | ".join(lines))
                    return ToolResult(True, "no results found")
                except Exception as e:
                    return ToolResult(False, error=f"search error: {e}")
            return ToolResult(False, error="web search not available")

        # --- FETCH (web page) ---
        elif tool_name == 'fetch':
            if 'web_fetch' in mcp_tools:
                try:
                    page = mcp_tools['web_fetch'](args)
                    if page:
                        text = page.get('text', '') or page.get('markdown', '')
                        return ToolResult(True, text[:500])
                    return ToolResult(False, error="could not fetch page")
                except Exception as e:
                    return ToolResult(False, error=f"fetch error: {e}")
            return ToolResult(False, error="web fetch not available")

        # --- EMAIL ---
        elif tool_name == 'email':
            if args.startswith('check'):
                if 'email_check' in mcp_tools:
                    try:
                        messages = mcp_tools['email_check']()
                        if messages:
                            lines = [f"from: {m.get('from','')} subject: {m.get('subject','')}"
                                     for m in messages[:5]]
                            return ToolResult(True, "\n".join(lines))
                        return ToolResult(True, "inbox is empty")
                    except Exception as e:
                        return ToolResult(False, error=f"email error: {e}")
                return ToolResult(False, error="email not available")
            elif args.startswith('send'):
                parts = args.split(None, 3)
                if len(parts) >= 3:
                    to = parts[1]
                    subject = parts[2]
                    return ToolResult(True, f"email drafted to {to} subject {subject} (needs confirmation to send)")
                return ToolResult(False, error="usage: email: send <to> <subject>")
            return ToolResult(False, error="usage: email: check | email: send <to> <subject>")

        # --- FILE ---
        elif tool_name == 'file':
            if args.startswith('read '):
                path = args[5:].strip()
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    return ToolResult(True, content[:500])
                except Exception as e:
                    return ToolResult(False, error=f"cannot read {path}: {e}")
            elif args.startswith('write '):
                parts = args[6:].split(None, 1)
                if len(parts) >= 2:
                    path, content = parts[0], parts[1]
                    try:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        return ToolResult(True, f"written to {path}")
                    except Exception as e:
                        return ToolResult(False, error=f"cannot write: {e}")
                return ToolResult(False, error="usage: file: write <path> <content>")
            elif args.startswith('list'):
                path = args[4:].strip() or '.'
                try:
                    entries = os.listdir(path)
                    return ToolResult(True, ", ".join(entries[:20]))
                except Exception as e:
                    return ToolResult(False, error=f"cannot list {path}: {e}")
            return ToolResult(False, error="usage: file: read|write|list <args>")

        # --- REMEMBER ---
        elif tool_name == 'remember':
            parts = args.split(None, 1)
            if len(parts) >= 2:
                key, value = parts[0], parts[1]
                MEMORY[key] = value
                return ToolResult(True, f"saved: {key}")
            return ToolResult(False, error="usage: remember: <key> <value>")

        # --- RECALL ---
        elif tool_name == 'recall':
            key = args.strip()
            if key in MEMORY:
                return ToolResult(True, MEMORY[key])
            return ToolResult(False, error=f"nothing stored for '{key}'")

        # --- GIT ---
        elif tool_name == 'git':
            if args.startswith('status'):
                try:
                    result = subprocess.run(['git', 'status', '--short'],
                                          capture_output=True, text=True, timeout=5)
                    return ToolResult(True, result.stdout or "clean")
                except Exception as e:
                    return ToolResult(False, error=f"git error: {e}")
            elif args.startswith('log'):
                try:
                    result = subprocess.run(['git', 'log', '--oneline', '-5'],
                                          capture_output=True, text=True, timeout=5)
                    return ToolResult(True, result.stdout or "no commits")
                except Exception as e:
                    return ToolResult(False, error=f"git error: {e}")
            return ToolResult(False, error="usage: git: status|log")

        # --- TWITTER / X ---
        elif tool_name == 'twitter':
            if args.startswith('post '):
                text = args[5:].strip()
                if 'twitter_post' in mcp_tools:
                    return ToolResult(True, f"tweet drafted: {text} (needs confirmation to post)")
                return ToolResult(False, error="twitter not connected")
            elif args.startswith('search '):
                query = args[7:].strip()
                if 'twitter_search' in mcp_tools:
                    try:
                        tweets = mcp_tools['twitter_search'](query)
                        if tweets:
                            return ToolResult(True, str(tweets)[:500])
                        return ToolResult(True, "no tweets found")
                    except Exception as e:
                        return ToolResult(False, error=f"twitter error: {e}")
                return ToolResult(False, error="twitter not connected")
            return ToolResult(False, error="usage: twitter: post|search <args>")

        # --- FACEBOOK ---
        elif tool_name == 'facebook':
            if args.startswith('post '):
                text = args[5:].strip()
                if 'facebook_post' in mcp_tools:
                    return ToolResult(True, f"post drafted: {text} (needs confirmation to post)")
                return ToolResult(False, error="facebook not connected")
            elif args.startswith('pages'):
                if 'facebook_pages' in mcp_tools:
                    try:
                        pages = mcp_tools['facebook_pages']()
                        return ToolResult(True, str(pages)[:500])
                    except Exception as e:
                        return ToolResult(False, error=f"facebook error: {e}")
                return ToolResult(False, error="facebook not connected")
            return ToolResult(False, error="usage: facebook: post|pages <args>")

        # --- THINK (extended generation) ---
        elif tool_name == 'think':
            if brain and brain.model:
                topic = args.strip()
                # Generate a longer response on the topic
                context = f"user: tell me more about {topic}\necho: "
                response = brain.respond(topic, temperature=0.7, length=200)
                return ToolResult(True, response)
            return ToolResult(False, error="brain not available")

        # --- STATUS ---
        elif tool_name == 'status':
            if args.startswith('brain') and brain:
                info = brain.info()
                return ToolResult(True, info)
            return ToolResult(False, error="usage: status: brain")

        # --- HELP ---
        elif tool_name == 'help':
            if args.startswith('tools'):
                lines = [f"  {desc}" for desc in TOOLS.values()]
                return ToolResult(True, "\n".join(lines))
            return ToolResult(False, error="usage: help: tools")

        else:
            return ToolResult(False, error=f"unknown tool: {tool_name}")

    except Exception as e:
        return ToolResult(False, error=f"execution error: {e}")


def _tool_calc(expression):
    """Safely evaluate a math expression."""
    # Only allow safe characters
    allowed = set('0123456789+-*/.() ,')
    clean = expression.replace('^', '**')
    if not all(c in allowed for c in clean if c != ' '):
        return ToolResult(False, error="invalid characters in expression")

    try:
        result = eval(clean, {"__builtins__": {}}, {"math": math, "abs": abs, "sqrt": math.sqrt})
        return ToolResult(True, str(result))
    except Exception as e:
        return ToolResult(False, error=f"calc error: {e}")


# ============================================================
# AGENT LOOP — Echo generates, we parse, execute, feed back
# ============================================================

def agent_respond(brain, user_input, mcp_tools=None, temperature=0.5,
                  max_tool_rounds=3):
    """Full agent loop: Echo generates → parse tool → execute → feed back.

    Returns the final text response and any tool calls made.
    """
    history = []  # Track what happened

    # Build context from recent conversation
    context_text = user_input
    result_text = ""

    for round_num in range(max_tool_rounds):
        # Generate Echo's response
        echo_output = brain.respond(context_text, temperature=temperature, length=150)

        # Check if Echo is trying to call a tool
        tool_call = parse_tool_call(echo_output)

        if tool_call is None:
            # No tool call — this is the final response
            result_text = echo_output
            history.append({'type': 'response', 'text': echo_output})
            break

        tool_name, tool_args = tool_call
        history.append({'type': 'tool_call', 'tool': tool_name, 'args': tool_args})

        # Execute the tool
        result = execute_tool(tool_name, tool_args, brain=brain, mcp_tools=mcp_tools)
        history.append({'type': 'tool_result', 'result': result.to_text()})

        # Feed the result back to Echo for a natural language response
        context_text = f"{result.to_text()}\necho: "
        result_text = echo_output  # Fallback if next round fails

    return result_text, history


def list_tools():
    """Return formatted list of all available tools."""
    lines = ["=== ECHO AGENT TOOLS ==="]
    for name, desc in TOOLS.items():
        lines.append(f"  {desc}")
    return "\n".join(lines)