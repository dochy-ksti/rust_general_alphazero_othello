import logging

from numpy.typing import NDArray
from numpy import float32

import numpy as np

from .intf_py_communicator import PyCommunicator

from .nnet import NNetWrapper

from .intf_self_player import SelfPlayer

log = logging.getLogger(__name__)


class Arena:
    def __init__(self):
        pass

    def play_game(self, sp: SelfPlayer, net1: NNetWrapper, net2: NNetWrapper) -> NDArray[float32]:
        turn = 0
        cur_player = -1
        while True:
            turn += 1
            cur_player *= -1
            rnum = sp.prepare_next(cur_player)
            if rnum == 0:
                continue
            elif rnum == 1:
                boards = sp.get_boards_for_prediction(cur_player)
                if cur_player == 1:
                    pis, win_rates = net1.predict(boards)
                else:
                    pis, win_rates = net2.predict(boards)
                sp.receive_prediction(pis, win_rates, cur_player)
            elif rnum == 2:
                results = sp.get_results_for_counting()
                return results

    def play_game_and_count_wons(self, sp: SelfPlayer, net1: NNetWrapper, net2: NNetWrapper) -> tuple[int, int, int]:
        array = self.play_game(sp, net1, net2)
        return (np.count_nonzero(array == 1.0), np.count_nonzero(array == -1.0), np.count_nonzero(array == 0.0))

    def play_games(self, pc: PyCommunicator, nnet: NNetWrapper, pnet: NNetWrapper) -> tuple[int, int, int]:

        n_won, p_won, draws = self.play_game_and_count_wons(
            pc.create_self_player(2), nnet, pnet)

        p_won2, n_won2, draws2 = self.play_game_and_count_wons(
            pc.create_self_player(2), pnet, nnet)

        log.info(
            f"NNET WIN RATE FIRST {n_won/(n_won+p_won+draws)} SECOND {n_won2/(n_won2+p_won2+draws2)}")

        return n_won + n_won2, p_won + p_won2, draws + draws2
