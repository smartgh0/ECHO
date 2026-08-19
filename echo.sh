#!/bin/bash
# ============================================================
# ECHO — Main entry point
# A self-contained neural mind that learns from your words.
# Pure Python. Zero dependencies. No PyTorch. No TensorFlow.
# ============================================================

set -e

# --- Resolve script directory ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Colors ---
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
MAGENTA='\033[35m'
CYAN='\033[36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

# --- Check Python ---
check_python() {
    if command -v python3 &>/dev/null; then
        PYTHON=python3
    elif command -v python &>/dev/null; then
        PYTHON=python
    else
        echo -e "${RED}Error: Python is not installed.${RESET}"
        echo -e "  Install Python 3 from https://python.org"
        exit 1
    fi
}

# --- Check if all files exist ---
check_files() {
    local missing=0
    for f in echo_matrix.py echo_rnn.py echo_brain.py echo_chat.py echo_dream.py echo_domain_chat.py; do
        if [ ! -f "$SCRIPT_DIR/$f" ]; then
            echo -e "${RED}Missing: $f${RESET}"
            missing=1
        fi
    done
    if [ $missing -eq 1 ]; then
        echo -e "${RED}Some Echo files are missing. Reinstall Echo.${RESET}"
        exit 1
    fi
}

# --- Run self-test ---
run_selftest() {
    echo -e "${DIM}[echo] Running self-test...${RESET}"
    $PYTHON echo_matrix.py 2>&1 | tail -1
    echo ""
}

# --- Help ---
show_help() {
    echo -e "${MAGENTA}${BOLD}ECHO${RESET} — A mind that grows from your words"
    echo ""
    echo -e "  ${CYAN}./echo.sh${RESET}              Start chatting with Echo (quantum layer mode)"
    echo -e "  ${CYAN}./echo.sh chat${RESET}         Same as above"
    echo -e "  ${CYAN}./echo.sh classical${RESET}    Start in classical mode (no quantum)"
    echo -e "  ${CYAN}./echo.sh quantum${RESET}      Start in full quantum mode (per-weight)"
    echo -e "  ${CYAN}./echo.sh transformer${RESET}  Start in quantum transformer mode (requires PyTorch)"
    echo -e "  ${CYAN}./echo.sh domain${RESET}       Start the subword domain model from domain_brain/"
    echo -e "  ${CYAN}./echo.sh dream${RESET}        Start dream mode (background training)"
    echo -e "  ${CYAN}./echo.sh test${RESET}         Run self-test"
    echo -e "  ${CYAN}./echo.sh info${RESET}         Show brain status"
    echo -e "  ${CYAN}./echo.sh help${RESET}         Show this help"
    echo -e "  ${DIM}Default mode: quantum layer (efficient + dropout){RESET}"
    echo ""
    echo -e "  ${DIM}Zero dependencies. Pure Python. No libraries.${RESET}"
    echo ""
}

# --- Info ---
show_info() {
    if [ ! -f "$SCRIPT_DIR/brain/model.json" ]; then
        echo -e "${DIM}[echo] No brain found. Start chatting first: ./echo.sh${RESET}"
        exit 0
    fi
    $PYTHON -c "
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('$SCRIPT_DIR')), '.'))
from echo_brain import EchoBrain
brain = EchoBrain.load('$SCRIPT_DIR/brain')
if brain:
    print(brain.info())
else:
    print('No brain found.')
"
}

# --- Main ---
check_python
check_files

case "${1:-chat}" in
    chat|"")
        run_selftest
        ECHO_MODE=quantum_layer exec $PYTHON echo_chat.py
        ;;
    classical)
        run_selftest
        ECHO_MODE=classical exec $PYTHON echo_chat.py
        ;;
    quantum)
        run_selftest
        ECHO_MODE=quantum exec $PYTHON echo_chat.py
        ;;
    transformer)
        run_selftest
        ECHO_MODE=quantum_transformer exec $PYTHON echo_chat.py
        ;;
    domain)
        exec $PYTHON echo_domain_chat.py
        ;;
    dream)
        exec $PYTHON echo_dream.py
        ;;
    test)
        $PYTHON echo_matrix.py
        ;;
    info)
        show_info
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo -e "${YELLOW}Unknown command: $1${RESET}"
        show_help
        exit 1
        ;;
esac