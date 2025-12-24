import mlx.core as mx
import numpy as np
import chess

from src.mlx_model import ChessNet

class ChessEngine:
    def __init__(self, model_path:str):
        print(f"Loading model from {model_path}...")
        self.model = ChessNet()
        self.model.load_weights(model_path)
        self.model.eval() # switches model to inference mode
        print("Model loaded!")

    def _get_mapper(self):
        '''
        Maps pieces to tensor channels (0-11)
        0-5 from players
        6-11 from opponent
        Pieces:
        P, N, B, R, Q, K
        '''
        return {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

    def board_to_tensor(self, board):
        '''
        Parses the chesss.Board to a tensor (1, 14, 8, 8) conserving the cannonic structure.
        '''
        
        # initializating a zeros tensor
        x = np.zeros((14, 8, 8), dtype=np.float32)
        
        # get perspective
        us = board.turn
        them = not board.turn
        
        # map pieces
        piece_map = self._get_mapper()

        # populates pieces channels (0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # determine if piece is ours or from opponent
                if piece.color == us:
                    channel = piece_map[piece.piece_type]
                else:
                    # if it is from opponent, add 6 to get to its opponent channel (6-11)
                    channel = piece_map[piece.piece_type] + 6

                # get obsulute coordinates (e4 is rank=3, file=4)
                rank, file = chess.square_rank(square), chess.square_file(square)
                
                if us == chess.BLACK:
                    # if playing black, rotate 180 degrees the board
                    # to achieve this we need to parse:
                    # rank 0 -> 7, rank 7 -> 0
                    rank = 7 - rank
                    file = 7 - file

                x[channel, rank, file] = 1.0

        # channel 12. castling rights
        can_castle = False
        if board.has_kingside_castling_rights(us) or board.has_queenside_castling_rights(us):
             can_castle = True
        
        if can_castle:
            x[12, :, :] = 1.0

        # Channel 13 En Passant
        if board.ep_square is not None:
            rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
            if us == chess.BLACK:
                rank = 7 - rank
                file = 7 - file
            x[13, rank, file] = 1.0

        # cast to mlx array and add dimension batch
        return mx.array(x).astype(mx.float32)[None, ...] # (1, 14, 8, 8)

    def encode_move(self, move, turn):
        '''
        Encodes an UCI movement to the index (0-4095) using the formula index = (origen * 64) destiny
        If the board is rotated (turn for black), the movement is also rotated
        '''
        if turn == chess.BLACK:
            from_sq = 63 - move.from_square
            to_sq = 63 - move.to_square
        else:
            from_sq = move.from_square
            to_sq = move.to_square
            
        return (from_sq * 64) + to_sq

    def get_best_move(self, board, temperature=0.1):
        '''
        Does inference, masks illegal moves, selects a movement
        '''
        x = self.board_to_tensor(board)
        
        # inference
        logits_p, value = self.model(x)
        
        # logits to numpy
        logits = np.array(logits_p[0]) # Shape (4096,)
        
        # model can predict illegal moves so the model outputs
        # should be masked to contain only legal movements.
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None, value.item()
            
        legal_indices = []
        move_map = {} # index map -> real move object
        
        for move in legal_moves:
            idx = self.encode_move(move, board.turn)
            if 0 <= idx < 4096:
                legal_indices.append(idx)
                move_map[idx] = move
        
        # array of real moves probabilities
        legal_logits = logits[legal_indices]
        
        # softmax over legal movements
        # Subtract max for stability
        exp_logits = np.exp(legal_logits - np.max(legal_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # selects a movements
        if temperature == 0:
            # gets best model legal movement
            best_idx_in_subset = np.argmax(probs)
        else:
            # flattens the probabilties distribution to allow the
            # selection of suboptimal movements.
            # temperature adjustemnt: logits / temp
            # 'to roll a weighted dice'
            probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            best_idx_in_subset = np.random.choice(len(probs), p=probs)
            
        chosen_idx = legal_indices[best_idx_in_subset]
        chosen_move = move_map[chosen_idx]
        
        # win estimation (-1 to 1)
        win_prob = value.item()
        
        return chosen_move, win_prob