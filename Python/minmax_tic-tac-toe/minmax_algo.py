import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    get next player
    """

    count_x = sum(row.count(X) for row in board)
    count_o = sum(row.count(O) for row in board)

    if count_x > count_o:
        return O
    elif count_o > count_x:
        return X
    else:
        return X


def actions(board):
    """
    get possible action coordinates
    """

    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] is EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    apply action to the board and return resulting board
    """
    current_player = player(board)
    i, j = action

    if i < 0 or i > 2 or j < 0 or j > 2 or board[i][j] is not EMPTY:
        raise Exception("Invalid move")

    copy_board = copy.deepcopy(board)
    copy_board[i][j] = current_player

    return copy_board


def winner(board):
    """
    check if there is a winner and return winner
    """

    for i in range(3):
        if (board[i][0] is not EMPTY) and (board[i][0] == board[i][1] == board[i][2]):
            return board[i][0]

    for j in range(3):
        if (board[0][j] is not EMPTY) and (board[0][j] == board[1][j] == board[2][j]):
            return board[0][j]

    if (board[0][0] is not EMPTY) and (board[0][0] == board[1][1] == board[2][2]):
        return board[0][0]

    if (board[0][2] is not EMPTY) and (board[0][2] == board[1][1] == board[2][0]):
        return board[0][2]

    return None


def terminal(board):
    """
    check if game over
    """

    if winner(board) is not None:
        return True

    none_count = sum(row.count(EMPTY) for row in board)
    if none_count == 0:
        return True
    else:
        return False


def utility(board):
    """
    check who has won
    """

    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    get the optimal move for any board based on minmaxing algorithm
    """
    if terminal(board):
        return None

    current_player = player(board)

    if current_player == X:
        best_val = -math.inf
        best_move = None
        for action in sorted(actions(board)):
            resulting_board = result(board, action)
            move_val = min_value(resulting_board)

            if move_val > best_val:
                best_val = move_val
                best_move = action
    else:
        best_val = math.inf
        best_move = None
        for action in sorted(actions(board)):
            resulting_board = result(board, action)
            move_val = max_value(resulting_board)
            if move_val < best_val:
                best_val = move_val
                best_move = action

    return best_move


def max_value(board):
    if terminal(board):
        return utility(board)

    v = -math.inf
    for action in sorted(actions(board)):
        resulting_board = result(board, action)
        next_player_value = min_value(resulting_board)
        v = max(v, next_player_value)
    return v


def min_value(board):
    if terminal(board):
        return utility(board)

    v = math.inf
    for action in sorted(actions(board)):
        resulting_board = result(board, action)
        next_player_value = max_value(resulting_board)
        v = min(v, next_player_value)
    return v
