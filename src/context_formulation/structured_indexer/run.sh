#!/bin/bash
set -e  

DATASET=""
REPOS_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_BASE_DIR="$ROOT_DIR/data/parser_output"

usage() {
    echo "Usage: $0 --dataset [RepoExec|DevEval] [--repos_dir path]"
    echo ""
    echo "Arguments:"
    echo "  --dataset     Dataset type: RepoExec or DevEval (required)"
    echo "  --repos_dir   Path to repositories directory (optional)"
    echo ""
    echo "Default repos_dir:"
    echo "  RepoExec: benchmark/RepoExec/test-apps"
    echo "  DevEval:  benchmark/DevEval/Source_Code"
    echo ""
    echo "Examples:"
    echo "  $0 --dataset RepoExec"
    echo "  $0 --dataset DevEval --repos_dir /path/to/deveval/repos"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --repos_dir)
            REPOS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset argument is required"
    usage
fi

if [[ "$DATASET" != "RepoExec" && "$DATASET" != "DevEval" ]]; then
    echo "Error: Dataset must be either 'RepoExec' or 'DevEval'"
    usage
fi


if [[ -z "$REPOS_DIR" ]]; then
    if [[ "$DATASET" == "RepoExec" ]]; then
        REPOS_DIR="$ROOT_DIR/benchmark/RepoExec/test-apps"
    else
        REPOS_DIR="$ROOT_DIR/benchmark/DevEval/Source_Code"
    fi
fi

REPOS_DIR="$(cd "$REPOS_DIR" 2>/dev/null && pwd)" || {
    echo "Error: Repository directory '$REPOS_DIR' does not exist"
    exit 1
}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Repos directory: $REPOS_DIR"
echo "  Output directory: $OUTPUT_BASE_DIR/$DATASET"
echo ""

OUTPUT_DIR="$OUTPUT_BASE_DIR/$DATASET"
mkdir -p "$OUTPUT_DIR"

run_ast_parser() {
    local repo_path="$1"
    local repo_name="$2"
    local output_path="$3"
    
    echo "Processing repository: $repo_name"
    echo "  Source: $repo_path"
    echo "  Output: $output_path"
    
    mkdir -p "$output_path"
    
    cd "$SCRIPT_DIR"
    python ast_parser.py --repo_path "$repo_path" \
        --output_dir "$output_path" || {
        echo "  Error: Failed to process $repo_name"
        return 1
    }
    
    echo "   Completed: $repo_name"
    echo ""
}

if [[ "$DATASET" == "RepoExec" ]]; then
    echo "Processing RepoExec repositories..."
    echo "Looking for repositories in: $REPOS_DIR"
    for repo_dir in "$REPOS_DIR"/*; do
        if [[ -d "$repo_dir" ]]; then
            repo_name="$(basename "$repo_dir")"
            output_path="$OUTPUT_DIR/$repo_name"
            
            run_ast_parser "$repo_dir" "$repo_name" "$output_path"
        fi
    done
    
elif [[ "$DATASET" == "DevEval" ]]; then
    echo "Processing DevEval repositories..."
    echo "Looking for topics in: $REPOS_DIR"
    
    for topic_dir in "$REPOS_DIR"/*; do
        if [[ -d "$topic_dir" ]]; then
            topic_name="$(basename "$topic_dir")"
            echo "Processing topic: $topic_name"
            
            for repo_dir in "$topic_dir"/*; do
                if [[ -d "$repo_dir" ]]; then
                    repo_name="$(basename "$repo_dir")"
                    output_path="$OUTPUT_DIR/$repo_name"
                    
                    run_ast_parser "$repo_dir" "$repo_name" "$output_path"
                fi
            done
        fi
    done
fi

echo "All repositories processed successfully!"
echo "Output saved to: $OUTPUT_DIR"
