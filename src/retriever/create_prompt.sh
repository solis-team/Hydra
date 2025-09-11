#!/bin/bash

BENCHMARK=""
CONTEXT=""
RETRIEVER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --context)
            CONTEXT="$2"
            shift 2
            ;;
        --retriever)
            RETRIEVER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$BENCHMARK" || -z "$CONTEXT" || -z "$RETRIEVER" ]]; then
    echo "Usage: $0 --benchmark <RepoExec|DevEval> --context <chunking|structured> --retriever <hybrid|dar|bm25|unixcoder>"
    exit 1
fi

if [[ "$BENCHMARK" != "RepoExec" && "$BENCHMARK" != "DevEval" ]]; then
    echo "Invalid benchmark. Must be RepoExec or DevEval"
    exit 1
fi

if [[ "$CONTEXT" != "chunking" && "$CONTEXT" != "structured" ]]; then
    echo "Invalid context. Must be chunking or structured"
    exit 1
fi

if [[ "$RETRIEVER" != "hybrid" && "$RETRIEVER" != "dar" && "$RETRIEVER" != "bm25" && "$RETRIEVER" != "unixcoder" ]]; then
    echo "Invalid retriever. Must be hybrid, dar, bm25, or unixcoder"
    exit 1
fi

if [[ "$CONTEXT" == "chunking" ]]; then
    if [[ "$RETRIEVER" == "dar" || "$RETRIEVER" == "hybrid" ]]; then
        echo "DAR and hybrid retrievers are not supported for chunking context"
        exit 1
    fi
    
    echo "Running make_window.py for chunking context..."
    python src/context_formulation/chunking/make_window.py --benchmark "$BENCHMARK"
    
    if [[ "$RETRIEVER" == "bm25" ]]; then
        echo "Running BM25 retriever..."
        python src/retriever/similar_context/bm25.py --benchmark "$BENCHMARK" --imported_context
    elif [[ "$RETRIEVER" == "unixcoder" ]]; then
        echo "Running UniXcoder retriever..."
        python src/retriever/similar_context/unixcoder.py --benchmark "$BENCHMARK" --imported_context
    fi
    
elif [[ "$CONTEXT" == "structured" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

    export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

    echo "Running load_benchmark.py for structured context..."
    python "$SCRIPT_DIR/load_benchmark.py" --benchmark "$BENCHMARK"
    
    echo "Running retriever.py with $RETRIEVER..."
    python "$SCRIPT_DIR/retriever.py" --benchmark "$BENCHMARK" --retriever "$RETRIEVER"
fi


echo "Prompt creation completed for $BENCHMARK with $CONTEXT context using $RETRIEVER retriever"
