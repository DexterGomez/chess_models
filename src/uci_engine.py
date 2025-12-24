# THIS CODE WAS GENERATED COMPLETELY BY GEMINI

import sys
import os
import chess
import argparse

try:
    from src.game_engine import ChessEngine
except ImportError as e:
    # This will show up in debug_log.txt if it fails
    sys.stderr.write(f"Error importing engine: {e}\n")
    sys.exit(1)

def main():
    # Use absolute path for the model to be safe
    model_path = os.path.join("checkpoints", "final_model.safetensors")
    
    try:
        engine = ChessEngine(model_path=model_path)
    except Exception as e:
        sys.stderr.write(f"Error loading model: {e}\n")
        sys.exit(1)
        
    board = chess.Board()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
        except KeyboardInterrupt:
            break

        if line == "uci":
            print("id name MLX_Chess_Alpha", flush=True)
            print("id author You", flush=True)
            print("uciok", flush=True) # <--- Critical flush
        
        elif line == "isready":
            print("readyok", flush=True)
        
        elif line == "ucinewgame":
            board.reset()

        elif line.startswith("position"):
            # (Copy your parse_position logic here or import it)
            # For brevity, reusing the logic from previous response
            args = line.split()
            if "startpos" in args:
                board.reset()
                moves_idx = args.index("moves") + 1 if "moves" in args else len(args)
            elif "fen" in args:
                fen_idx = args.index("fen") + 1
                if "moves" in args:
                    moves_idx = args.index("moves")
                    fen_part = " ".join(args[fen_idx:moves_idx])
                    moves_idx += 1
                else:
                    fen_part = " ".join(args[fen_idx:])
                    moves_idx = len(args)
                board.set_fen(fen_part)
            else:
                board.reset()
                moves_idx = len(args)

            if moves_idx < len(args):
                for move_uci in args[moves_idx:]:
                    try:
                        board.push_uci(move_uci)
                    except: pass

        elif line.startswith("go"):
            best_move, win_prob = engine.get_best_move(board, temperature=0.1)
            if best_move:
                print(f"bestmove {best_move.uci()}", flush=True)
            else:
                print("bestmove (none)", flush=True)

        elif line == "quit":
            break

if __name__ == "__main__":
    main()